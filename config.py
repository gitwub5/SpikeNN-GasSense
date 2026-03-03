from pathlib import Path

# Base Paths (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"

# DM Gas Sensor Paths
DM_GAS_DATA_DIR = DATA_DIR / "dm_gas_sensor"
DM_GAS_DATA_RAW_DIR = DATA_RAW_DIR / "dm_gas_sensor"
DM_GAS_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "dm_gas_sensor"

# QCM Paths
QCM_DATA_DIR = DATA_DIR / "qcm"
QCM_CSV_DIR = QCM_DATA_DIR / "csv"
QCM_PLOTS_DIR = QCM_DATA_DIR / "plots"
QCM_DATA_RAW_DIR = DATA_RAW_DIR / "qcm"
