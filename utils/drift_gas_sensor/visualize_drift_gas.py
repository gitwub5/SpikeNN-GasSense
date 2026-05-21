import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_DATA_RAW_DIR, DRIFT_GAS_ANALYZE_DIR

# --- Constants ---
GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene"
}

# Color palette for 6 gas classes
GAS_COLORS = {
    1: "#e6194b",  # Red
    2: "#3cb44b",  # Green
    3: "#4363d8",  # Blue
    4: "#f58231",  # Orange
    5: "#911eb4",  # Purple
    6: "#42d4f4",  # Cyan
}

BATCH_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']

NUM_BATCHES = 10
NUM_SENSORS = 16
# Each sensor has 8 features: [DR steady, DR max, DR area, ..., EMA_alpha0.001]
FEATURES_PER_SENSOR = 8
FEATURE_NAMES = [
    "Steady-state (DR)",
    "Max response",
    "Area under curve",
    "y_max / y_min",
    "EMA α=0.1",
    "EMA α=0.01",
    "EMA α=0.001",
    "Rise time"
]


def load_batch(batch_num):
    """Loads a specific batch file."""
    file_path = DRIFT_GAS_DATA_RAW_DIR / f'batch{batch_num}.dat'
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return None, None
    X, y = load_svmlight_file(str(file_path))
    return X.toarray(), y


def load_all_batches(max_samples_per_batch=None):
    """Loads all 10 batches, optionally subsampling."""
    all_X, all_y, all_batch_ids = [], [], []
    for i in range(1, NUM_BATCHES + 1):
        X, y = load_batch(i)
        if X is not None:
            if max_samples_per_batch and X.shape[0] > max_samples_per_batch:
                idx = np.random.choice(X.shape[0], max_samples_per_batch, replace=False)
                X, y = X[idx], y[idx]
            all_X.append(X)
            all_y.append(y)
            all_batch_ids.extend([i] * len(y))
    return np.vstack(all_X), np.concatenate(all_y), np.array(all_batch_ids)


# =============================================================================
# 1) All-Batch PCA Subplot Grid
# =============================================================================
def plot_pca_all_batches_grid(save_path):
    """
    Generates a 2x5 subplot grid, each showing PCA of one batch 
    with gas class coloring. Makes it easy to compare class separation
    across batches visually.
    """
    print("=" * 60)
    print("[1/3] Generating All-Batch PCA Subplot Grid...")

    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    fig.suptitle('PCA Class Separation per Batch (Batch 1 → 10)',
                 fontsize=18, fontweight='bold', y=1.01)

    # First: fit PCA on ALL data (with scaling) so axes are comparable
    print("  Loading all batches for unified PCA...")
    X_all, y_all, batch_ids = load_all_batches()
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_all)
    pca = PCA(n_components=2)
    pca.fit(X_scaled_all)
    X_pca_all = pca.transform(X_scaled_all)

    # Calculate percentiles to remove top/bottom 0.5% extreme outliers for better visualization
    p_low, p_high = 0.5, 99.5
    x_min, x_max = np.percentile(X_pca_all[:, 0], [p_low, p_high])
    y_min, y_max = np.percentile(X_pca_all[:, 1], [p_low, p_high])
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    for idx, batch_num in enumerate(range(1, NUM_BATCHES + 1)):
        row, col = idx // 5, idx % 5
        ax = axes[row, col]

        # Get data for this batch from the already-loaded data
        mask = batch_ids == batch_num
        X_batch = X_scaled_all[mask]
        y_batch = y_all[mask]

        X_pca = pca.transform(X_batch)
        
        # Apply outlier filter mask per batch
        valid_mask = (X_pca[:, 0] >= x_min) & (X_pca[:, 0] <= x_max) & \
                     (X_pca[:, 1] >= y_min) & (X_pca[:, 1] <= y_max)
                     
        X_pca_valid = X_pca[valid_mask]
        y_batch_valid = y_batch[valid_mask]

        classes = np.unique(y_batch_valid)
        for c in classes:
            c_mask = y_batch_valid == c
            ax.scatter(X_pca_valid[c_mask, 0], X_pca_valid[c_mask, 1],
                       c=GAS_COLORS[int(c)], label=GAS_NAMES[int(c)],
                       alpha=0.6, s=12, edgecolors='none')

        ax.set_title(f'Batch {batch_num} (n={valid_mask.sum()})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               fontsize=11, title='Gas Type', title_fontsize=12,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# =============================================================================
# 2) PCA Drift with Gas-Type Colors + Batch Markers
# =============================================================================
def plot_pca_drift_by_gas_and_batch(save_path):
    """
    PCA scatter of all batches combined:
      - Color = Gas type (6 colors)
      - Marker = Batch number (10 markers)
    Shows both class separation AND drift simultaneously.
    """
    print("=" * 60)
    print("[2/3] Generating PCA Drift Plot (Gas Colors + Batch Markers)...")

    X_all, y_all, batch_ids = load_all_batches(max_samples_per_batch=500)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate percentiles to remove top/bottom 0.5% extreme outliers so the clusters don't squish
    p_low, p_high = 0.5, 99.5
    x_min, x_max = np.percentile(X_pca[:, 0], [p_low, p_high])
    y_min, y_max = np.percentile(X_pca[:, 1], [p_low, p_high])
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    
    valid_mask = (X_pca[:, 0] >= x_min) & (X_pca[:, 0] <= x_max) & \
                 (X_pca[:, 1] >= y_min) & (X_pca[:, 1] <= y_max)

    fig, ax = plt.subplots(figsize=(14, 11))

    # Plot each (gas, batch) combination
    for batch_num in range(1, NUM_BATCHES + 1):
        for gas_id in sorted(GAS_NAMES.keys()):
            mask = (batch_ids == batch_num) & (y_all == gas_id) & valid_mask
            if mask.sum() == 0:
                continue
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=GAS_COLORS[gas_id],
                marker=BATCH_MARKERS[batch_num - 1],
                alpha=0.55,
                s=25,
                edgecolors='white',
                linewidths=0.3,
                label=f'{GAS_NAMES[gas_id]} B{batch_num}'
            )

    ax.set_title('PCA Drift Visualization: Gas Type × Batch',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=13)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.grid(True, alpha=0.2)

    # Create two separate legends
    # Legend 1: Gas types (color)
    gas_handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=GAS_COLORS[g], markersize=10,
                              label=GAS_NAMES[g])
                   for g in sorted(GAS_NAMES.keys())]
    leg1 = ax.legend(handles=gas_handles, title='Gas Type', loc='upper left',
                     fontsize=10, title_fontsize=11)
    ax.add_artist(leg1)

    # Legend 2: Batch markers (shape)
    batch_handles = [plt.Line2D([0], [0], marker=BATCH_MARKERS[i], color='w',
                                markerfacecolor='gray', markersize=9,
                                label=f'Batch {i + 1}')
                     for i in range(NUM_BATCHES)]
    ax.legend(handles=batch_handles, title='Batch (Time →)', loc='upper right',
              fontsize=9, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# =============================================================================
# 3) Sensor Feature Mean Time-Series
# =============================================================================
def plot_feature_mean_drift(save_path):
    """
    Shows how each sensor's steady-state feature (feature 0 of each sensor)
    changes across batches 1→10, grouped by gas type.
    Directly demonstrates sensor drift over time.
    """
    print("=" * 60)
    print("[3/3] Generating Sensor Feature Mean Drift Time-Series...")

    # Collect per-batch, per-class means for the steady-state feature
    # Feature indices: sensor i -> feature i*8 (steady-state)
    # Select 4 representative sensors to keep the plot readable
    selected_sensors = [0, 4, 8, 12]  # Sensors 1, 5, 9, 13

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sensor Feature Drift Over Time (Steady-State Response per Gas)',
                 fontsize=16, fontweight='bold', y=1.01)

    batch_range = list(range(1, NUM_BATCHES + 1))

    for plot_idx, sensor_idx in enumerate(selected_sensors):
        row, col = plot_idx // 2, plot_idx % 2
        ax = axes[row, col]
        feature_idx = sensor_idx * FEATURES_PER_SENSOR  # Steady-state feature

        for gas_id in sorted(GAS_NAMES.keys()):
            means = []
            stds = []
            valid_batches = []

            for batch_num in batch_range:
                X, y = load_batch(batch_num)
                if X is None:
                    continue
                gas_mask = y == gas_id
                if gas_mask.sum() == 0:
                    continue
                vals = X[gas_mask, feature_idx]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_batches.append(batch_num)

            if not means:
                continue

            means = np.array(means)
            stds = np.array(stds)
            valid_batches = np.array(valid_batches)

            ax.plot(valid_batches, means,
                    marker='o', markersize=5, linewidth=2,
                    color=GAS_COLORS[gas_id], label=GAS_NAMES[gas_id])
            ax.fill_between(valid_batches, means - stds, means + stds,
                            alpha=0.1, color=GAS_COLORS[gas_id])

        ax.set_title(f'Sensor {sensor_idx + 1} (Feature #{feature_idx})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Batch Number (Time →)', fontsize=11)
        ax.set_ylabel('Mean Steady-State Response', fontsize=11)
        ax.set_xticks(batch_range)
        ax.grid(True, alpha=0.3, linestyle=':')

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               fontsize=11, title='Gas Type', title_fontsize=12,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Output directory: {DRIFT_GAS_ANALYZE_DIR}")
    DRIFT_GAS_ANALYZE_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # 1) All-batch PCA grid
    plot_pca_all_batches_grid(
        DRIFT_GAS_ANALYZE_DIR / 'all_batches_pca_grid.png'
    )

    # 2) PCA drift with gas colors + batch markers
    plot_pca_drift_by_gas_and_batch(
        DRIFT_GAS_ANALYZE_DIR / 'pca_drift_gas_and_batch.png'
    )

    # 3) Feature mean drift over time
    plot_feature_mean_drift(
        DRIFT_GAS_ANALYZE_DIR / 'feature_mean_drift_timeseries.png'
    )

    print("\n" + "=" * 60)
    print("✅ All visualizations complete!")
    print(f"   Output: {DRIFT_GAS_ANALYZE_DIR}")


if __name__ == "__main__":
    main()
