import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

# Add project root to sys.path to import config and dataset
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_MODEL_WEIGHTS_DIR, DRIFT_GAS_ANALYZE_DIR
from dataset.drift_gas_dataset import get_dataloaders
from train.train_drift_gas_npx import GasSensorSNN_NPX, forward_pass

def visualize_drift_predictions_npx():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device} for inference")
    
    batch_size = 64
    num_steps = 50
    model_path = DRIFT_GAS_MODEL_WEIGHTS_DIR / 'best_drift_gas_snn_npx.pth'
    
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        print("Please run train/train_drift_gas_npx.py first.")
        return
        
    net = GasSensorSNN_NPX().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['npx_module'])
    net.eval()
    
    print("Evaluating NPX-style Model Accuracy across all Batches (1 to 10)...")
    batch_numbers = list(range(1, 11))
    accuracies = []
    
    for eval_batch in batch_numbers:
        # Crucial: The scaler MUST be fitted precisely on Batches 1 to 8, just like training
        _, eval_loader, _ = get_dataloaders(train_batches=list(range(1, 9)), test_batches=[eval_batch], batch_size=batch_size)
        
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
        
    DRIFT_GAS_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DRIFT_GAS_ANALYZE_DIR / 'model_drift_accuracy_profile_npx.png'
    
    plt.figure(figsize=(11, 6))
    
    # Create the line plot
    plt.plot(batch_numbers, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8, color='#0066cc')
    
    # Fill the area under the curve slightly for aesthetics
    plt.fill_between(batch_numbers, accuracies, alpha=0.1, color='#0066cc')
    
    # Formatting
    plt.title('SNN Accuracy Under Sensor Drift (NPX Trainer Model, Train: B1-B8)', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Batch Number (Time Progression ->)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(batch_numbers)
    plt.ylim(0, 105)
    
    # Highlight training subset (Batches 1 to 8)
    plt.axvspan(0.8, 8.2, color='green', alpha=0.1, label='Training Region (Batches 1-8)')
    # Highlight Testing subset (Batches 9 to 10)
    plt.axvspan(8.8, 10.2, color='red', alpha=0.1, label='Testing / Drift Region (Batches 9-10)')
    
    plt.legend(loc='lower left', fontsize=11)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Visualization saved at: {save_path}")

if __name__ == "__main__":
    visualize_drift_predictions_npx()
