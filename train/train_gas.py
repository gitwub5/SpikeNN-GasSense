import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn as nn
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset.gas_dataset import GasRegressionDataset
from model.spiking_net import SpikingNet

# --- Configuration (설정) ---
CONFIG = {
    "data_dir": Path("data/gas_sensor"),         # Gas 데이터가 저장될 경로
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "time_steps": 500,                              # 윈도우 길이 (5초)
    "input_features": 16,                           # Gas Dataset의 센서 개수
    "hidden_size": 128,                             
    "num_outputs": 2,                               # 회귀 출력 개수 (Gas1, Ethylene)
    "beta": 0.9,
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42
}

torch.manual_seed(CONFIG["seed"])

def train_gas_model():
    print("Initializing Gas Regression Dataset...")
    
    # GasRegressionDataset 은 기본적으로 ethylene_CO.pt, ethylene_methane.pt 두 개의 파일을 로드합니다.
    dataset = GasRegressionDataset(data_dir=CONFIG["data_dir"])
    
    if len(dataset) == 0:
        print("No data found. Please place .pt files in the data/gas_sensor directory.")
        return

    # Train/Test 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=True)
    
    # --- 핵심 포인트: 모델 구조 그대로 사용! ---
    # 클래스 개수(num_classes) 대신 출력 차원(num_outputs) 2를 넘겨줍니다. 
    net = SpikingNet(
        num_inputs=CONFIG["input_features"], 
        num_hidden=CONFIG["hidden_size"], 
        num_outputs=CONFIG["num_outputs"], 
        beta=CONFIG["beta"]
    ).to(CONFIG["device"])
    
    optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])
    
    # --- 핵심 포인트: Regression 용 Loss 사용 ---
    # 분류일 때는 SF.mse_count_loss 나 CrossEntropy를 사용했지만,
    # 회귀일 때는 연속적인 농도 값을 예측해야 하므로 PyTorch의 순수 MSELoss를 사용합니다.
    loss_fn = nn.MSELoss()
    
    print(f"\n[TRAIN] Starting Gas Regression on {CONFIG['device']} for {CONFIG['num_epochs']} epochs...")
    print(f"Total dataset size: {len(dataset)}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    for epoch in range(CONFIG["num_epochs"]):
        net.train()
        train_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # [Batch, Time, Feat] -> [Time, Batch, Feat]
            data = data.to(CONFIG["device"]).permute(1, 0, 2)
            targets = targets.to(CONFIG["device"])
            
            # Forward
            spk_rec, mem_rec = net(data)
            
            # --- 핵심 포인트: Membrane Potential 로 예측 ---
            # spk_rec 대신 마지막 시간 단계의 Membrane Potential (전압) 값 자체를 예측값으로 사용합니다.
            # Output Shape of mem_rec: [Time, Batch, num_outputs]
            # mem_rec[-1] 의 Shape: [Batch, num_outputs]
            predictions = mem_rec[-1]
            
            # Loss & Backprop
            loss_val = loss_fn(predictions, targets)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            train_loss += loss_val.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss = test_gas_model(net, test_loader, CONFIG["device"], loss_fn)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} \tTrain MSE Loss: {avg_train_loss:.4f} \tVal MSE Loss: {val_loss:.4f}")
        
    # --- 학습 완료 후 모델 저장 ---
    os.makedirs("model_weights", exist_ok=True)
    save_path = "model_weights/snn_gas_regression.pth"
    torch.save(net.state_dict(), save_path)
    print(f"\n✅ Training complete! Model saved to {save_path}")

def test_gas_model(net, dataloader, device, loss_fn):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device).permute(1, 0, 2)
            targets = targets.to(device)
            
            spk_rec, mem_rec = net(data)
            
            # 검증 시에도 마지막 스텝의 Membrane Potential 을 타겟 농도와 비교 (MSE)
            predictions = mem_rec[-1]
            loss_val = loss_fn(predictions, targets)
            
            val_loss += loss_val.item()
            
    return val_loss / len(dataloader)

if __name__ == "__main__":
    train_gas_model()
