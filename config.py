from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
CAPTURE_DIR = ROOT / "captures"
DB_PATH = ROOT / "db" / "inventory.db"
DEFAULT_DATA_DIR = ROOT / "data" / "d1-clean"

MODELS = {
    "EfficientNet-B0 (fine-tuned)": {
        "arch": "efficientnet_b0",
        "weights": "best_effnet_b0.pth",
        "img_size": 224,
    },
    "Simple CNN (from scratch)": {
        "arch": "simplecnn",
        "weights": "final_best_cnn_v3.pth",
        "img_size": 224,
    },
}

ACTIVE_MODEL = MODELS["EfficientNet-B0 (fine-tuned)"]

CONF_THRESHOLD = 0.5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CARTON_KEYWORDS = ("juice", "milk", "yoghurt", "oatghurt", "soyghurt", "sour-cream")