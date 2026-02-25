import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


class GasRegressionDataset(Dataset):
    """
    UCI Gas Sensor Array under Dynamic Gas Mixtures (Regression).
    Loads preprocessed sliding windows saved as .pt:
        {
            "X": Tensor (N, T, 16),
            "y": Tensor (N, 2),  # [gas1, ethylene]
            "sampling_rate": 100,
            "window_size": 500,
            # (optional) "stride": int,
            # (optional) "file_name": str,
            # (optional) "target_names": List[str]
        }

    Supports:
      - loading single file (ethylene_CO.pt or ethylene_methane.pt)
      - loading both files and concatenating
      - optional time-based split by indices if the .pt includes split indices
    """

    def __init__(
        self,
        data_dir: str = "data/gas_sensor",
        files: Optional[List[str]] = None,
        split: Optional[str] = None,  # "train" | "val" | "test" if indices exist
        device: Optional[str] = None,
    ):
        """
        Args:
            data_dir: directory containing .pt files
            files: list of pt filenames to load. default loads both:
                   ["ethylene_CO.pt", "ethylene_methane.pt"]
            split: optional. If the saved dict contains split indices:
                   - "train" / "val" / "test" to subset
            device: optional. If provided, move tensors to device in memory.
        """
        self.data_dir = Path(data_dir)
        self.files = files or ["ethylene_CO.pt", "ethylene_methane.pt"]
        self.split = split

        X_list = []
        y_list = []
        meta: Dict[str, Any] = {}

        for fname in self.files:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found at: {path}")

            # torch.load compatibility (weights_only introduced in newer versions)
            try:
                d = torch.load(path, weights_only=True)
            except TypeError:
                d = torch.load(path)

            if "X" not in d or "y" not in d:
                raise KeyError(
                    f"{path.name} must contain keys 'X' and 'y'. Found: {list(d.keys())}"
                )

            X = d["X"]
            y = d["y"]

            # Optional split support
            if split is not None:
                if "split_indices" not in d:
                    raise KeyError(
                        f"{path.name} has no 'split_indices'. Save split indices during preprocessing "
                        f"or set split=None."
                    )
                if split not in d["split_indices"]:
                    raise KeyError(
                        f"Invalid split='{split}'. Available: {list(d['split_indices'].keys())}"
                    )
                idx = d["split_indices"][split]
                X = X[idx]
                y = y[idx]

            X_list.append(X)
            y_list.append(y)

            # Keep/merge meta (last file wins on conflict, which is fine here)
            for k in ["sampling_rate", "window_size", "stride", "target_names", "sensor_order", "file_name"]:
                if k in d:
                    meta[k] = d[k]

        self.X = torch.cat(X_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        self.meta = meta

        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

        # Sanity checks
        if self.X.ndim != 3 or self.X.shape[-1] != 16:
            raise ValueError(f"X should have shape (N, T, 16). Got {tuple(self.X.shape)}")
        if self.y.ndim != 2 or self.y.shape[-1] != 2:
            raise ValueError(f"y should have shape (N, 2). Got {tuple(self.y.shape)}")

        print(
            f"Loaded regression dataset: N={len(self.X)}, "
            f"T={self.X.shape[1]}, C={self.X.shape[2]} from {self.files}"
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # return (X_window, y_target)
        return self.X[idx], self.y[idx]