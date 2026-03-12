import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
from snntorch import functional as SF

# Add project root to sys.path to import config and dataset
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_MODEL_WEIGHTS_DIR
from dataset.drift_gas_dataset import get_dataloaders

class GasSensorSNN(nn.Module):
    def __init__(self, beta=0.95):
        super().__init__()
        
        # 128 features as input
        self.fc1 = nn.Linear(128, 256) 
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid()) 
        
        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        # 6 gas classes as output (0-5)
        self.fc3 = nn.Linear(128, 6) 
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        # x.shape = [batch, 128]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []
        mem3_rec = []

        # T 스텝에 걸친 forward pass
        for step in range(num_steps):
            # 동일한 입력(Constant current) 주입
            cur1 = self.fc1(x) 
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3
    num_steps = 50  # Number of time steps for SNN

    # Dataset (Train on Batch 1, Test on Batch 2 to see drift effect)
    train_loader, test_loader, scaler = get_dataloaders(train_batches=[1], test_batches=[2], batch_size=batch_size)

    # Initialize Model, Loss, Optimizer
    net = GasSensorSNN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # We use CrossEntropy on total spike count
    loss_fn = SF.ce_count_loss() 

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

            # forward pass
            spk_rec, mem_rec = net(data, num_steps)
            
            # calculate loss (using snntorch SF)
            loss = loss_fn(spk_rec, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record
            train_loss += loss.item()
            
            # Accuracy metric: predicted class is the one with highest spike count across T steps
            acc = SF.accuracy_rate(spk_rec, targets)
            correct += acc * targets.size(0) # snntorch accuracy returns ratio, convert to count
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

                spk_rec, mem_rec = net(data, num_steps)
                loss = loss_fn(spk_rec, targets)

                test_loss += loss.item()
                acc_t = SF.accuracy_rate(spk_rec, targets)
                correct_t += acc_t * targets.size(0)
                total_t += targets.size(0)

        test_acc = correct_t / total_t

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss/len(test_loader):.4f} Acc: {test_acc*100:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = DRIFT_GAS_MODEL_WEIGHTS_DIR / 'best_drift_gas_snn.pth'
            torch.save(net.state_dict(), save_path)
            print(f"-> Saved new best model to {save_path}!")

    print(f"\n--- Training Complete. Best Batch 2 Test Acc: {best_acc*100:.2f}% ---")
    print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    train()
