import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def process_and_save(file_path: Path, output_dir: Path, out_filename: str):
    print(f"Loading data from {file_path}...")
    
    # Read the data. Columns: 'Time', 'Gas1_conc', 'Ethylene_conc', R1..R16
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=None)
    
    # Verify we have exactly 19 columns: (1 time + 2 targets + 16 sensors)
    if df.shape[1] != 19:
        raise ValueError(f"Expected 19 columns, but got {df.shape[1]} in {file_path.name}")

    # Extract targets (concentrations) and features (sensors)
    targets = df.iloc[:, 1:3].values.astype(np.float32)  # [gas1, ethylene]
    features = df.iloc[:, 3:].values.astype(np.float32)   # 16 sensors
    
    print(f"Loaded {len(df)} rows. Normalizing features with StandardScaler...")
    
    # Normalize features using StandardScaler (fit per file)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Sliding window segmentation specifications
    sampling_rate = 100
    window_size = 500    # 5 seconds at 100 Hz
    stride = 100         # 1 second stride
    
    num_samples = len(df)
    windows_x = []
    windows_y = []
    
    print(f"Segmenting into windows (size={window_size}, stride={stride})...")
    
    for start_idx in range(0, num_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Features for the sequence window
        x_window = features[start_idx:end_idx, :]
        
        # Target is the concentration at the last timestep of the window
        # (end_idx is exclusive, so the last valid index is end_idx - 1)
        y_target = targets[end_idx - 1, :]
        
        windows_x.append(x_window)
        windows_y.append(y_target)
        
    X_tensor = torch.tensor(np.array(windows_x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(windows_y), dtype=torch.float32)
    
    # Create the output dictionary mapping format
    output_dict = {
        "X": X_tensor,
        "y": y_tensor,
        "sampling_rate": sampling_rate,
        "window_size": window_size
    }
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir / out_filename
    
    print(f"Saving to {out_path}...")
    print(f" - X shape (N, Time, Feat): {X_tensor.shape}")
    print(f" - y shape (N, Targets):    {y_tensor.shape}")
    
    torch.save(output_dict, out_path)
    print(f"Successfully saved {out_filename}.\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Gas Sensor Dataset for Regression")
    parser.add_argument("--data_dir", type=str, default="/Users/gwshin/Dev/Nanolatis/spike_nn/data_raw/gas_sensor",
                        help="Path to the directory containing raw txt files")
    parser.add_argument("--out_dir", type=str, default="/Users/gwshin/Dev/Nanolatis/spike_nn/data/gas_sensor",
                        help="Path to the directory where .pt files will be saved")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # 1. Process ethylene_CO.txt
    file_co = data_dir / "ethylene_CO.txt"
    if file_co.exists():
        process_and_save(file_co, out_dir, "ethylene_CO.pt")
    else:
        print(f"Warning: {file_co} not found.")

    # 2. Process ethylene_methane.txt
    file_methane = data_dir / "ethylene_methane.txt"
    if file_methane.exists():
        process_and_save(file_methane, out_dir, "ethylene_methane.pt")
    else:
        print(f"Warning: {file_methane} not found.")

if __name__ == "__main__":
    main()
