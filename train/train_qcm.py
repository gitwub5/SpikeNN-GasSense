import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sys

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.spiking_net import SpikingNet

# --- Configuration (설정) ---
# 학습에 필요한 하이퍼파라미터 및 경로 설정
CONFIG = {
    "data_dir": Path("../data/csv"),                # CSV 데이터 파일 경로
    "metadata_path": Path("../data/qcm_all_metadata.csv"), # 메타데이터 파일 경로
    "batch_size": 16,                               # 배치 크기
    "num_epochs": 50,                               # 학습 에포크 수
    "learning_rate": 2e-3,                          # 학습률
    "time_steps": 100,                              # 시계열 데이터 길이 (모든 입력을 이 길이로 맞춤)
    "input_features": 2,                            # 입력 특징 수 (dF/Hz, dR/ohm)
    "hidden_size": 64,                              # 은닉층 뉴런 수
    "beta": 0.9,                                    # LIF 뉴런의 전압 감쇠율 (Decay rate)
    "device": "cuda" if torch.cuda.is_available() else "cpu", # 학습 장치 (GPU/CPU)
    "seed": 42                                      # 재현성을 위한 시드값
}

torch.manual_seed(CONFIG["seed"])

# --- Dataset (데이터셋) ---
# QCM 데이터를 로드하고 전처리하는 클래스
# npx_data_manager.py의 역할을 수행 (데이터 로드, 라벨 인코딩, 포맷팅)
class QCMDataset(Dataset):
    def __init__(self, data_dir, metadata_path, time_steps):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_path)
        self.time_steps = time_steps
        
        # 파일 존재 여부 확인 및 유효한 샘플 필터링
        self.samples = []
        self.labels = []
        self.valid_indices = []
        
        # 라벨 인코딩 (Label Encoder)
        # 문자열로 된 화합물 이름(target_chemical)을 숫자(0, 1, 2...)로 변환
        self.metadata = self.metadata.dropna(subset=['target_chemical'])
        self.label_encoder = LabelEncoder()
        self.metadata['encoded_label'] = self.label_encoder.fit_transform(self.metadata['target_chemical'])
        
        print(f"Found classes: {self.label_encoder.classes_}")
        
        for idx, row in self.metadata.iterrows():
            # 확장자 불일치 해결 (.txt -> .csv)
            filename = Path(row['file_name']).stem + ".csv"
            file_path = self.data_dir / filename
            
            if file_path.exists():
                self.samples.append(file_path)
                self.labels.append(row['encoded_label'])
            else:
                # print(f"Warning: File not found {file_path}")
                pass
                
        print(f"Loaded {len(self.samples)} valid samples out of {len(self.metadata)} metadata entries.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # CSV 파일 읽기
        try:
            df = pd.read_csv(file_path)
            # 특징 선택 (dF/Hz, dR/ohm)
            features = df[['dF/Hz', 'dR/ohm']].values.astype(np.float32)
            
            # 시계열 길이 고정 (Interpolation/Resizing)
            # 입력 데이터의 길이를 time_steps(100)으로 맞춤
            # Input shape: [Time, Features] -> [Features, Time] (interpolate를 위해 transpose)
            features_t = torch.tensor(features).T.unsqueeze(0) # [1, Feat, Time]
            
            # 선형 보간법을 사용하여 길이를 조절
            features_interp = torch.nn.functional.interpolate(
                features_t, 
                size=self.time_steps, 
                mode='linear', 
                align_corners=False
            )
            
            # 다시 원래 형태로 변환: [Time, Features]
            # SNN 입력은 보통 [Time, Batch, Features] 형태를 띔 (배치는 DataLoader가 추가)
            data = features_interp.squeeze(0).T # [Time, Features]
            
            return data, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((self.time_steps, 2)), label

# --- Training (학습) ---
# 학습 루프 및 평가 (npx_trainer.py 역할)
def train_model():
    # 데이터 로드
    dataset = QCMDataset(CONFIG["data_dir"], CONFIG["metadata_path"], CONFIG["time_steps"])
    
    # Train/Test 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=True)
    
    # 모델 초기화
    num_classes = len(dataset.label_encoder.classes_)
    net = SpikingNet(CONFIG["input_features"], CONFIG["hidden_size"], num_classes, CONFIG["beta"]).to(CONFIG["device"])
    
    # 최적화 도구 (Adam) 및 손실 함수 (MSE Count Loss)
    optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    
    print(f"\n[TRAIN] Starting training on {CONFIG['device']} for {CONFIG['num_epochs']} epochs...")
    print(f"Input features: {CONFIG['input_features']}, Classes: {num_classes} ({dataset.label_encoder.classes_})")
    
    loss_hist = []
    acc_hist = []
    
    # 학습 루프 시작
    for epoch in range(CONFIG["num_epochs"]):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # 데이터 차원 변경: [Batch, Time, Feat] -> [Time, Batch, Feat]
            # SNN은 시간 축이 가장 앞에 와야 처리하기 편함
            data = data.to(CONFIG["device"]) 
            data = data.permute(1, 0, 2)
            targets = targets.to(CONFIG["device"])
            
            # 순전파 (Forward pass)
            spk_rec, mem_rec = net(data)
            
            # 손실 계산 (Loss)
            # 전체 시간 동안 발생한 스파이크의 빈도 등을 정답과 비교
            loss_val = loss_fn(spk_rec, targets)
            
            # 역전파 (Backpropagation) 및 가중치 갱신
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # 기록 저장
            train_loss += loss_val.item()
            
            # 정확도 계산
            acc = SF.accuracy_rate(spk_rec, targets)
            correct += acc * data.size(1) # 배치 크기만큼 곱해줌
            total += data.size(1)
            
        avg_loss = train_loss / len(train_loader)
        avg_acc = correct / total
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} \tLoss: {avg_loss:.4f} \tAcc: {100*avg_acc:.2f}%")
        
        # 검증 (Validation) - 5 에포크마다 실행
        if (epoch + 1) % 5 == 0:
            test_acc = test_model(net, test_loader, CONFIG["device"])
            print(f"--> Validation Acc: {100*test_acc:.2f}%")

    print("Training Complete.")
    return net, loss_hist, acc_hist

# 테스트 함수
def test_model(net, dataloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device).permute(1, 0, 2)
            targets = targets.to(device)
            spk_rec, _ = net(data)
            correct += SF.accuracy_rate(spk_rec, targets) * data.size(1)
            total += data.size(1)
    
    return correct / total

if __name__ == "__main__":
    train_model()
