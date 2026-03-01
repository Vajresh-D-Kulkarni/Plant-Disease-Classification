from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b3_stage2.keras"
IMAGE_SIZE = (224, 224)
