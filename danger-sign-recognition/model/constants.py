from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
DEFAULT_APP_MODEL_DIR = PROJECT_ROOT / "app" / "model"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_ARCHITECTURES = ("efficientnet_b0", "resnet18", "mobilenet_v3_small", "strong_cnn")

ARTIFACT_FILENAMES = {
    "best_checkpoint": "best_danger_sign_model.pt",
    "final_checkpoint": "danger_sign_model_final.pt",
    "history_csv": "history.csv",
    "history_json": "history.json",
    "labels": "labels.json",
    "confusion_matrix": "confusion_matrix.json",
    "training_config": "training_config.json",
    "experiment_report": "experiment_report.json",
    "model_card": "model_card.md",
    "dataset_audit": "dataset_audit.json",
}
