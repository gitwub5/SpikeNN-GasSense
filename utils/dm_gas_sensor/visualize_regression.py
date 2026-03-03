import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset.gas_dataset import GasRegressionDataset
from model.spiking_net import SpikingNet
import config

CONFIG = {
    "data_dir": config.DM_GAS_DATA_DIR,
    "model_path": config.DM_GAS_MODEL_WEIGHTS_DIR / "snn_gas_regression.pth",
    "batch_size": 32,
    "input_features": 16,
    "hidden_size": 128,
    "num_outputs": 2, # Gas1 conc, Ethylene conc
    "beta": 0.9,
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
}

def evaluate_and_visualize():
    print("Loading test dataset...")
    # Load dataset. Here we just take the first file 'ethylene_CO.pt' or both
    dataset = GasRegressionDataset(data_dir=CONFIG["data_dir"], files=["ethylene_CO.pt", "ethylene_methane.pt"])
    
    # Select a test subset (Same 20% slice from the random_split in train_gas if seed=42)
    torch.manual_seed(42) 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # We use batch_size 1 or a single large batch to gather sequential predictions for plotting
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Loading SpikingNet from {CONFIG['model_path']}...")
    net = SpikingNet(
        num_inputs=CONFIG["input_features"], 
        num_hidden=CONFIG["hidden_size"], 
        num_outputs=CONFIG["num_outputs"], 
        beta=CONFIG["beta"]
    ).to(CONFIG["device"])
    
    if not CONFIG["model_path"].exists():
        print(f"WARNING: Model weight not found at {CONFIG['model_path']}. Using untrained weights.")
    else:
        net.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
        
    net.eval()
    
    all_targets = []
    all_preds = []
    
    print("Running Inference over the full Test Set...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(CONFIG["device"]).permute(1, 0, 2)
            targets = targets.to(CONFIG["device"])
            
            # Forward
            _, mem_rec = net(data)
            
            # Predictions at the last timestep
            preds = mem_rec[-1]
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
                
    # Concatenate all gathered arrays
    y_true_full = np.concatenate(all_targets, axis=0)
    y_pred_full = np.concatenate(all_preds, axis=0)
    
    # Calculate Regression Metrics
    mse = np.mean((y_true_full - y_pred_full) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_full - y_pred_full), axis=0)
    
    # R2 Score calculation
    ss_res = np.sum((y_true_full - y_pred_full) ** 2, axis=0)
    ss_tot = np.sum((y_true_full - np.mean(y_true_full, axis=0)) ** 2, axis=0)
    # Prevent division by zero if target variance is 0
    r2 = 1 - (ss_res / np.where(ss_tot == 0, 1, ss_tot))

    print("\n" + "="*50)
    print("📊 Regression Evaluation Metrics (Full Test Set)")
    print("="*50)
    
    print("📌 Gas 1 (CO/Methane)")
    print(f"  - MSE  : {mse[0]:.4f}")
    print(f"  - RMSE : {rmse[0]:.4f} ppm")
    print(f"  - MAE  : {mae[0]:.4f} ppm")
    print(f"  - R²   : {r2[0]:.4f}")
    
    print("\n📌 Gas 2 (Ethylene)")
    print(f"  - MSE  : {mse[1]:.4f}")
    print(f"  - RMSE : {rmse[1]:.4f} ppm")
    print(f"  - MAE  : {mae[1]:.4f} ppm")
    print(f"  - R²   : {r2[1]:.4f}")
    print("="*50 + "\n")
    
    # Limit for plotting to keep the graph readable
    num_samples_to_plot = 300
    y_true = y_true_full[:num_samples_to_plot]
    y_pred = y_pred_full[:num_samples_to_plot]
    
    # Plotting
    print("Generating Regression Plot...")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time_axis = np.arange(len(y_true))
    
    # Gas 1 (CO or Methane depending on the slice, but here it's mixed from the random split)
    axs[0].plot(time_axis, y_true[:, 0], label="True Gas1 (CO/Methane)", color="blue", linewidth=1.5)
    axs[0].plot(time_axis, y_pred[:, 0], label="Predicted", color="red", linestyle="--", linewidth=1.5)
    axs[0].set_ylabel("Concentration (ppm)")
    axs[0].set_title("Gas 1 Concentration Prediction")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Ethylene
    axs[1].plot(time_axis, y_true[:, 1], label="True Ethylene", color="green", linewidth=1.5)
    axs[1].plot(time_axis, y_pred[:, 1], label="Predicted", color="orange", linestyle="--", linewidth=1.5)
    axs[1].set_xlabel("Sample Index (Test Split)")
    axs[1].set_ylabel("Concentration (ppm)")
    axs[1].set_title("Ethylene Concentration Prediction")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save Plot
    out_dir = Path("analyze")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "regression_inference_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully at: {save_path}")

if __name__ == "__main__":
    evaluate_and_visualize()
