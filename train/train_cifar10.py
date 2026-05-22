import sys
from pathlib import Path
import torch
import torch.nn as nn
from snntorch import functional as SF

# Add project root to path so we can import from other directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import CIFAR10_MODEL_WEIGHTS_DIR
from dataset.cifar10_dataset import get_cifar10_dataloaders
from model.cifar10_spiking_net import CIFAR10SpikingNet

def train():
    # Parameters
    batch_size = 128
    num_epochs = 50
    num_steps = 25
    learning_rate = 1e-3
    beta = 0.9
    
    # Ensure save directory exists
    CIFAR10_MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    
    # Model
    net = CIFAR10SpikingNet(beta=beta).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = SF.ce_rate_loss()  # Cross Entropy rate loss for SNNs
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            spk_rec, _ = net(data, num_steps)
            
            # Loss calculation
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = spk_rec.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_acc = 100. * correct / total
        
        # Testing
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                spk_rec, _ = net(data, num_steps)
                
                loss = loss_fn(spk_rec, targets)
                test_loss += loss.item()
                
                _, predicted = spk_rec.sum(dim=0).max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        test_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), CIFAR10_MODEL_WEIGHTS_DIR / "best_cifar10_snn.pth")
            print("Model saved.")

if __name__ == "__main__":
    train()
