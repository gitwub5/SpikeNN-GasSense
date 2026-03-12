import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import DRIFT_GAS_DATA_RAW_DIR

class DriftGasDataset(Dataset):
    """
    PyTorch Dataset for the Gas Sensor Array Drift Dataset.
    """
    def __init__(self, batch_numbers=[1], scaler=None, fit_scaler=False):
        """
        Args:
            batch_numbers (list of int): List of batch numbers to load (1 to 10).
            scaler (sklearn.preprocessing.StandardScaler): Scaler object.
            fit_scaler (bool): If True, fit the scaler to the loaded data.
        """
        self.X, self.y = self._load_data(batch_numbers)
        
        # Convert labels from 1-6 to 0-5 for PyTorch CrossEntropyLoss
        self.y = self.y - 1
        
        self.scaler = scaler
        if fit_scaler:
            if self.scaler is None:
                self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif self.scaler is not None:
            self.X = self.scaler.transform(self.X)

        # Convert to PyTorch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def _load_data(self, batch_numbers):
        all_X = []
        all_y = []
        
        for batch_num in batch_numbers:
            file_path = DRIFT_GAS_DATA_RAW_DIR / f'batch{batch_num}.dat'
            if not file_path.exists():
                print(f"Warning: {file_path} not found. Skipping.")
                continue
                
            X, y = load_svmlight_file(str(file_path))
            all_X.append(X.toarray())
            all_y.append(y)
            
        if not all_X:
            raise ValueError("No data loaded. Check the batch numbers and file paths.")
            
        X_concat = np.vstack(all_X)
        y_concat = np.concatenate(all_y)
        
        return X_concat, y_concat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(train_batches=[1], test_batches=[2], batch_size=32):
    """
    Creates and returns training and testing DataLoaders.
    Fits the scaler on the training data and applies it to the testing data.
    """
    scaler = StandardScaler()
    
    print(f"Loading training data from batches: {train_batches}...")
    train_dataset = DriftGasDataset(batch_numbers=train_batches, scaler=scaler, fit_scaler=True)
    
    print(f"Loading testing data from batches: {test_batches}...")
    test_dataset = DriftGasDataset(batch_numbers=test_batches, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Don't drop last for testing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler
