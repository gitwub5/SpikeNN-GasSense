import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
from snntorch import functional as SF
from snntorch import utils

# Add project root to sys.path to import config and dataset
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_MODEL_WEIGHTS_DIR
from dataset.drift_gas_dataset import get_dataloaders

class GasSensorSNN(nn.Module):
    """
    NPX Trainer 호환 Gas Sensor SNN 모델.
    
    NPX Trainer 방식과 동일하게:
    - init_hidden=True: 뉴런이 내부적으로 membrane potential 관리
    - spike_grad=None: snntorch 기본 surrogate gradient 사용
    - bias=False: NPX 양자화 호환
    - forward()는 단일 timestep만 처리, 외부에서 timestep 루프를 돌림
    """
    def __init__(self, beta=0.95):
        super().__init__()
        
        # 128 features as input
        self.fc1 = nn.Linear(128, 256, bias=False) 
        self.lif1 = snn.Leaky(
            beta=beta, 
            spike_grad=None,        # NPX Trainer 방식: 기본 surrogate gradient
            init_hidden=True,       # NPX Trainer 방식: 내부 상태 관리
            reset_mechanism='subtract',
        )
        
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.lif2 = snn.Leaky(
            beta=beta, 
            spike_grad=None,
            init_hidden=True,
            reset_mechanism='subtract',
        )
        
        # 6 gas classes as output (0-5)
        self.fc3 = nn.Linear(128, 6, bias=False) 
        self.lif3 = snn.Leaky(
            beta=beta, 
            spike_grad=None,
            init_hidden=True,
            reset_mechanism='subtract',
        )

    def forward(self, x):
        """
        단일 timestep forward pass (NPX Trainer의 NpxModule.forward()와 동일한 패턴).
        외부에서 timestep 루프를 돌리고, 각 step마다 이 메서드를 호출.
        
        Args:
            x: [batch, 128] — 단일 timestep의 입력
        Returns:
            spk3: 마지막 LIF 뉴런의 spike 출력
        """
        cur1 = self.fc1(x)
        spk1 = self.lif1(cur1)
        
        cur2 = self.fc2(spk1)
        spk2 = self.lif2(cur2)
        
        cur3 = self.fc3(spk2)
        spk3 = self.lif3(cur3)
        
        return spk3


def forward_pass(net, data, num_steps):
    """
    NPX Trainer의 forward_pass()와 동일한 패턴.
    
    MATRIX3D 방식: 동일한 입력을 num_steps 동안 반복 주입 (constant current).
    이는 Gas Sensor의 정적 데이터에 적합한 인코딩 방식.
    """
    spk_rec = []
    utils.reset(net)  # 모든 LIF 뉴런의 hidden state 초기화

    for step in range(num_steps):
        spk_out = net(data)  # 동일한 입력을 반복 주입
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


def save_checkpoint(net, optimizer, path: Path):
    """NPX Trainer 방식의 checkpoint 저장."""
    checkpoint = {
        'npx_module': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(net, optimizer, path: Path):
    """NPX Trainer 방식의 checkpoint 로드."""
    checkpoint = torch.load(path, weights_only=False)
    net.load_state_dict(checkpoint['npx_module'])
    if optimizer and checkpoint['optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    num_steps = 50  # Number of time steps for SNN

    # Dataset (Train on Batch 1, Test on Batch 2 to see drift effect)
    train_loader, test_loader, scaler = get_dataloaders(train_batches=[1], test_batches=[2], batch_size=batch_size)

    # Initialize Model, Loss, Optimizer
    net = GasSensorSNN().to(device)

    # NPX Trainer 방식: 기본 Adam optimizer (lr, betas 모두 기본값)
    optimizer = torch.optim.Adam(net.parameters())

    # NPX Trainer 방식: mse_count_loss with correct/incorrect rate
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    # Create weight directory
    DRIFT_GAS_MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # --- Training ---
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            # NPX Trainer 방식의 forward pass
            spk_rec = forward_pass(net, data, num_steps)
            
            # calculate loss
            loss = loss_fn(spk_rec, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record
            train_loss += loss.item()
            
            # Accuracy metric: predicted class is the one with highest spike count across T steps
            acc = SF.accuracy_rate(spk_rec, targets)
            correct += acc * targets.size(0)
            total += targets.size(0)

        train_acc = correct / total

        # --- Testing ---
        net.eval()
        test_loss = 0
        correct_t = 0
        total_t = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device)
                targets = targets.to(device)

                spk_rec = forward_pass(net, data, num_steps)
                loss = loss_fn(spk_rec, targets)

                test_loss += loss.item()
                acc_t = SF.accuracy_rate(spk_rec, targets)
                correct_t += acc_t * targets.size(0)
                total_t += targets.size(0)

        test_acc = correct_t / total_t

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss/len(test_loader):.4f} Acc: {test_acc*100:.2f}%")

        # Save best model (NPX Trainer checkpoint 형식)
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = DRIFT_GAS_MODEL_WEIGHTS_DIR / 'best_drift_gas_snn.pth'
            save_checkpoint(net, optimizer, save_path)
            print(f"-> Saved new best model to {save_path}!")

    print(f"\n--- Training Complete. Best Batch 2 Test Acc: {best_acc*100:.2f}% ---")
    print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    train()
