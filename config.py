from pathlib import Path

# Base Paths (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
ANALYZE_DIR = PROJECT_ROOT / "analyze"

# Drift Gas Sensor Paths
DRIFT_GAS_DATA_DIR = DATA_DIR / "drift_gas_sensor"
DRIFT_GAS_DATA_RAW_DIR = DATA_RAW_DIR / "drift_gas_sensor"
DRIFT_GAS_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "drift_gas_sensor"
DRIFT_GAS_ANALYZE_DIR = ANALYZE_DIR / "drift_gas_sensor"

# CIFAR-10 Paths
CIFAR10_DATA_DIR = DATA_DIR / "cifar10"
CIFAR10_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "cifar10"
CIFAR10_ANALYZE_DIR = ANALYZE_DIR / "cifar10"