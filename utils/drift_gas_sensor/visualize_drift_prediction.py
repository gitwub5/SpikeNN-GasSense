import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# Add project root to sys.path to import config and dataset
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_MODEL_WEIGHTS_DIR, DRIFT_GAS_ANALYZE_DIR
from dataset.drift_gas_dataset import get_dataloaders

class GasSensorSNN(nn.Module):
    """NPX Trainer 호환 모델 (train_drift_gas.py와 동일)."""
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
    """NPX Trainer 방식의 forward pass."""
    spk_rec = []
    utils.reset(net)
    for step in range(num_steps):
        spk_out = net(data)
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)


def visualize_drift_predictions():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} for inference")
    
    # Configuration
    batch_size = 64
    num_steps = 50
    model_path = DRIFT_GAS_MODEL_WEIGHTS_DIR / 'best_drift_gas_snn.pth'
    
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        print("Please run train/train_drift_gas.py first.")
        return
        
    # Load Model (NPX Trainer checkpoint 형식)
    net = GasSensorSNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['npx_module'])
    net.eval()
    
    # Evaluate over all batches (Time: Batch 1 to 10)
    print("Evaluating Model Accuracy across all Batches (1 to 10)...")
    batch_numbers = list(range(1, 11))
    accuracies = []
    
    for eval_batch in batch_numbers:
        # Load the specific batch as test set 
        # (scaler was fitted on batch 1 during training, so we must load it exactly the same)
        _, eval_loader, _ = get_dataloaders(train_batches=[1], test_batches=[eval_batch], batch_size=batch_size)
        
        correct_e = 0
        total_e = 0
        
        with torch.no_grad():
            for data, targets in eval_loader:
                data = data.to(device)
                targets = targets.to(device)

                spk_rec = forward_pass(net, data, num_steps)
                
                acc_e = SF.accuracy_rate(spk_rec, targets)
                correct_e += acc_e * targets.size(0)
                total_e += targets.size(0)
                
        eval_acc = correct_e / total_e
        accuracies.append(eval_acc * 100)
        print(f"Batch {eval_batch} Acc: {eval_acc*100:.2f}%")
        
    # Plotting
    DRIFT_GAS_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DRIFT_GAS_ANALYZE_DIR / 'model_drift_accuracy_profile.png'
    
    plt.figure(figsize=(10, 6))
    
    # Create the line plot
    plt.plot(batch_numbers, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8, color='#FF5722')
    
    # Fill the area under the curve slightly for aesthetics
    plt.fill_between(batch_numbers, accuracies, alpha=0.1, color='#FF5722')
    
    # Formatting
    plt.title('SNN Classification Accuracy Drop over Time (Sensor Drift)', fontsize=14, pad=15)
    plt.xlabel('Batch Number (Time Progression ->)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(batch_numbers)
    plt.ylim(0, 105)
    
    # Highlight training batch
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
    plt.text(1.2, 10, 'Trained only on Batch 1', color='gray', rotation=90)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Visualization saved at: {save_path}")

if __name__ == "__main__":
    visualize_drift_predictions()
