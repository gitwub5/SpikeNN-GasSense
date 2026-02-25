import torch
import argparse
from pathlib import Path

def verify_dataset(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\nVerifying {file_path.name}...")
    
    # Safe loading using weights_only if supported
    try:
        data = torch.load(file_path, weights_only=True)
    except TypeError:
        data = torch.load(file_path)

    # Validate output dictionary structure
    required_keys = ["X", "y", "sampling_rate", "window_size"]
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing key `{key}` in the dictionary")
            return
            
    X = data["X"]
    y = data["y"]
    sr = data["sampling_rate"]
    ws = data["window_size"]

    print(f"Keys present: {list(data.keys())}")
    print(f"Sampling rate: {sr} Hz")
    print(f"Window size: {ws} timesteps")
    
    print(f"X shape: {X.shape} (Expected: (N, 500, 16))")
    print(f"y shape: {y.shape} (Expected: (N, 2))")
    
    # Validate dimensions and expectations strictly
    assert X.dim() == 3, f"X should be 3-dimensional, got {X.dim()}"
    assert X.shape[1] == 500, f"X expected sequence length 500, got {X.shape[1]}"
    assert X.shape[2] == 16, f"X expected 16 features, got {X.shape[2]}"
    
    assert y.dim() == 2, f"y should be 2-dimensional, got {y.dim()}"
    assert y.shape[1] == 2, f"y expected 2 targets, got {y.shape[1]}"
    assert X.shape[0] == y.shape[0], f"Batch size mismatch: X({X.shape[0]}) != y({y.shape[0]})"
    
    print(f"X value range (min, max): {X.min().item():.4f}, {X.max().item():.4f}")
    print(f"y value range (min, max): {y.min().item():.4f}, {y.max().item():.4f}")
    
    print(f"✅ {file_path.name} verified successfully!")


def main():
    parser = argparse.ArgumentParser(description="Verify Preprocessed Gas Sensor Data for Regression")
    parser.add_argument("--data_dir", type=str, default="/Users/gwshin/Dev/Nanolatis/spike_nn/data/gas_sensor",
                        help="Path to the directory containing .pt files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    verify_dataset(data_dir / "ethylene_CO.pt")
    verify_dataset(data_dir / "ethylene_methane.pt")


if __name__ == "__main__":
    main()
