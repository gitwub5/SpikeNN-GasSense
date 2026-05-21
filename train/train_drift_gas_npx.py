import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

# Add project root to sys.path to import config and dataset
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_MODEL_WEIGHTS_DIR
from dataset.drift_gas_dataset import get_dataloaders

class GasSensorSNN_NPX(nn.Module):
    """
    NPX Trainer 호환 Gas Sensor SNN 모델.
    - app/drift_gas_app.cfg와 정확히 일치하는 구조
    """
    def __init__(self, beta=0.95):
        super().__init__()
        self.fc1 = nn.Linear(128, 256, bias=False) 
        self.lif1 = snn.Leaky(beta=beta, spike_grad=None, init_hidden=True, reset_mechanism='subtract')
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=None, init_hidden=True, reset_mechanism='subtract')
        self.fc3 = nn.Linear(128, 6, bias=False) 
        self.lif3 = snn.Leaky(beta=beta, spike_grad=None, init_hidden=True, reset_mechanism='subtract')

    def forward(self, x):
        cur1 = self.fc1(x)
        spk1 = self.lif1(cur1)
        cur2 = self.fc2(spk1)
        spk2 = self.lif2(cur2)
        cur3 = self.fc3(spk2)
        spk3 = self.lif3(cur3)
        return spk3

def forward_pass(net, data, num_steps):
    spk_rec = []
    utils.reset(net)
    for step in range(num_steps):
        spk_out = net(data)
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

def save_checkpoint(net, optimizer, path: Path):
    checkpoint = {
        'npx_module': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, path)

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Hyperparameters from app/drift_gas_app.cfg
    batch_size = 64
    num_epochs = 20
    num_steps = 50

    # Dataset: Train on Batches 1~8, Test on Batches 9~10 (mimicking npx_trainer split)
    print("Loading datasets... (Train: Batch 1-8, Test: Batch 9-10)")
    train_batches = list(range(1, 9))
    test_batches = [9, 10]
    train_loader, test_loader, scaler = get_dataloaders(train_batches=train_batches, test_batches=test_batches, batch_size=batch_size)

    net = GasSensorSNN_NPX().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    
    DRIFT_GAS_MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(num_epochs): # 20 Epochs
        # Training
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = forward_pass(net, data, num_steps)
            loss = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = SF.accuracy_rate(spk_rec, targets)
            correct += acc * targets.size(0)
            total += targets.size(0)

        train_acc = correct / total

        # Testing (Validation over entirely unseen Batches 9-10)
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
              f"Test(9-10) Loss: {test_loss/len(test_loader):.4f} Acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = DRIFT_GAS_MODEL_WEIGHTS_DIR / 'best_drift_gas_snn_npx.pth'
            save_checkpoint(net, optimizer, save_path)
            print(f"-> Saved new best model to {save_path}!")

    print(f"\n--- Training Complete. Best Test(9-10) Acc: {best_acc*100:.2f}% ---")
    print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    train()
