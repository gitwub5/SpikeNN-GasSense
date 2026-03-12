import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_DATA_RAW_DIR, DRIFT_GAS_ANALYZE_DIR

def load_batch(batch_num):
    """Loads a specific batch file."""
    file_path = DRIFT_GAS_DATA_RAW_DIR / f'batch{batch_num}.dat'
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return None, None
    X, y = load_svmlight_file(str(file_path))
    return X.toarray(), y

def plot_mean_features_by_class(X, y, save_path):
    """Plot the mean feature values for each gas class in Batch 1."""
    classes = np.unique(y)
    plt.figure(figsize=(15, 8))
    
    # Gas names based on dataset description
    gas_names = {
        1: "Ethanol",
        2: "Ethylene",
        3: "Ammonia",
        4: "Acetaldehyde",
        5: "Acetone",
        6: "Toluene"
    }

    for c in classes:
        mean_features = X[y == c].mean(axis=0)
        plt.plot(mean_features, label=gas_names.get(int(c), f"Class {int(c)}"), alpha=0.8)
    
    plt.title('Mean Feature Responses per Gas Class (Batch 1)')
    plt.xlabel('Feature Index (0-127)')
    plt.ylabel('Mean Value')
    plt.legend(title='Gas Type', loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_all_batches(save_path):
    """Applies PCA to a subset of all batches to visualize drift over time."""
    all_X = []
    all_batches = []
    
    # Load all batches
    print("Loading all batches for PCA drift analysis...")
    for i in range(1, 11):
        X, _ = load_batch(i)
        if X is not None:
            # Subsample to keep plot readable (e.g., max 500 samples per batch)
            idx = np.random.choice(X.shape[0], min(500, X.shape[0]), replace=False)
            all_X.append(X[idx])
            all_batches.extend([i] * len(idx))
            
    if not all_X:
        print("No batches loaded.")
        return
        
    X_concat = np.vstack(all_X)
    y_batch = np.array(all_batches)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_concat)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_batch, cmap='viridis', alpha=0.6, s=15)
    
    plt.colorbar(scatter, label='Batch Number (Time ->)')
    plt.title('PCA of Sensor Data: Visualizing Drift Over Batches (Time)')
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_pca_single_batch(X, y, batch_num, save_path):
    """Applies PCA to a single batch to show class separation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    gas_names = {
        1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
        4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"
    }
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.8, s=30)
    
    # Create legend handles manually
    handles, _ = scatter.legend_elements()
    labels = [gas_names.get(int(lbl), f"Class {int(lbl)}") for lbl in np.unique(y)]
    plt.legend(handles, labels, title="Gas Type")
    
    plt.title(f'PCA of Sensor Data: Gas Class Separation (Batch {batch_num})')
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print(f"Creating output directory: {DRIFT_GAS_ANALYZE_DIR}")
    DRIFT_GAS_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading Batch 1...")
    X1, y1 = load_batch(1)
    
    if X1 is not None:
        print("Generating mean features plot for Batch 1...")
        plot_mean_features_by_class(X1, y1, DRIFT_GAS_ANALYZE_DIR / 'batch1_mean_features.png')
        
        print("Generating PCA gas class separation plot for Batch 1...")
        plot_pca_single_batch(X1, y1, 1, DRIFT_GAS_ANALYZE_DIR / 'batch1_pca_classes.png')
        
    print("Generating PCA drift plot over all batches...")
    plot_pca_all_batches(DRIFT_GAS_ANALYZE_DIR / 'all_batches_pca_drift.png')
    
    print("Visualization complete! Check the analyze/drift_gas_sensor directory.")

if __name__ == "__main__":
    main()
