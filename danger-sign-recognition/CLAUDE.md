# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Danger warning sign image recognition system for 5 hazard categories (explosion, falling objects, falling rocks, flammable, water hazard). Uses EfficientNet-B0 with ImageNet pretrained weights, fine-tuned via transfer learning. A Python web frontend (built on `http.server`) serves a single-page app for image upload, camera capture, and real-time recognition. All UI text and class labels are in Chinese. Academic project for Tianjin University (TJU).

## Commands

All commands run from this directory (`danger-sign-recognition/`).

```bash
# Install
pip install -r requirements.txt

# Start web frontend (opens at http://127.0.0.1:7860/)
python python_frontend/server.py --host 127.0.0.1 --port 7860

# Train (default recommended config)
python model/train_model.py --dataset dataset_viewpoint --out model/artifacts_viewpoint \
  --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30 --batch-size 16 --repeats 1

# Evaluate on augmented test set
python model/evaluate_model.py \
  --checkpoint model/artifacts_viewpoint/best_danger_sign_model.pt \
  --dataset dataset_test_viewpoint --out model/artifacts_viewpoint/test_classification_report.json --repeats 1

# Evaluate frontend scene photos (with signal-crop preprocessing)
python model/evaluate_frontend_photos.py \
  --checkpoint model/artifacts_viewpoint/best_danger_sign_model.pt \
  --dataset test_inputs/frontend_photos --crop square --padding 0.18

# Generate augmented training set (if dataset_viewpoint/ is missing)
python model/build_viewpoint_dataset.py --source dataset --out dataset_viewpoint --per-class 220 --size 320

# Generate independent test set
python model/build_viewpoint_dataset.py --source dataset --out dataset_test_viewpoint --per-class 80 --size 320 --seed 9090 --no-source-copy

# Audit a dataset
python model/data_audit.py --dataset dataset_viewpoint --out model/artifacts_viewpoint/dataset_audit.json
```

No test suite or linter is configured. All scripts support `--help`.

## Architecture

### PROJECT_ROOT import pattern

Every script resolves `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (points to `danger-sign-recognition/`). Scripts outside the `model/` package insert `PROJECT_ROOT` into `sys.path` before importing `from model.*`. New scripts must follow this pattern.

### Training pipeline

`train_model.py` → `TrainingConfig` (frozen dataclass) → seed → `data_audit.py` audits dataset → `AugmentedSignDataset` scans class subdirectories, repeats samples with heavy runtime augmentation → `build_model()` factory creates architecture and replaces classifier head with two-layer MLP → `run_epoch()` train/val loop with optional AMP → `EarlyStopping` + `CheckpointWriter` → `reporting.py` exports artifacts.

### Model checkpoint format

`.pt` files are dicts: `{"model": state_dict, "metadata": {...}, "history": [...]}`. Metadata includes `arch`, `classes`, `image_size`. At inference time, `arch` and `classes` from metadata drive `build_model()` to reconstruct the architecture before loading weights.

### Signal detection preprocessing (shared between server and evaluation)

`browser_signal_bounds` and `square_bounds` are defined in `evaluate_frontend_photos.py` and imported by `server.py`. The pipeline: downscale → find yellow-core + dark stroke pixels → connected-component analysis → expand to square with padding → resize → ImageNet normalize. This is the critical path for frontend accuracy — both the server and `evaluate_frontend_photos.py` must use the same crop logic.

### Frontend (python_frontend/server.py)

Single-file 637-line web app. `DangerSignPredictor` wraps inference with signal-crop preprocessing. `PythonFrontendHandler` on `ThreadingHTTPServer` serves inline HTML/JS/CSS (no static files except sign icons in `assets/`). API: `GET /` (page), `GET /api/status` (health), `POST /api/predict` (base64 data URL → predictions with confidence, risk level, action). Client-side vanilla JS handles file upload, drag-and-drop, camera via `getUserMedia`, live recognition at 1.2s intervals.

### Key architectural choices

- `build_model()` in `architectures.py` is the single factory for all architectures — `efficientnet_b0`, `resnet18`, `mobilenet_v3_small`, `strong_cnn`
- ResNet18 freeze differs from EfficientNet/MobileNet: freezes all except `layer4` and `fc` (not just `features`)
- `StrongCNN` is the custom CNN (depthwise-separable + squeeze-excitation) — use when pretrained weights are unavailable
- `SIGN_INFO` in `server.py` maps class IDs to Chinese names, risk levels, and safety recommendations
- Dataset format: one subdirectory per class under a root, each containing image files (extensions in `constants.py`)

## Key Constraints

- Default model path: `model/artifacts_viewpoint/best_danger_sign_model.pt`
- `dataset_viewpoint/` is gitignored — regenerate with `build_viewpoint_dataset.py` if missing
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.ppm`
- The outer repo (parent directory) has its own `.gitignore` excluding `dataset_scene_crops/`, `artifacts_scene*/`, and evaluation JSON outputs
