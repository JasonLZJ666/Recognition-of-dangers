# Danger Sign Recognition Model Card

## Task
Classify five categories of warning and danger signs for a browser-based safety inspection demo.

## Model
- Architecture: `efficientnet_b0`
- Image size: `224`
- Pretrained: `True`
- Freeze backbone: `True`
- Total parameters: `4336769`
- Trainable parameters: `329221`
- Estimated model size MB: `16.704`

## Training
- Best metric: `val_macro_f1` = `1.000000`
- Best epoch: `1`
- Device: `cpu`
- PyTorch: `2.12.1+cpu`

## Classes
- `explosion`
- `falling_objects`
- `falling_rocks`
- `flammable`
- `water_hazard`

## Validation Summary
- Loss: `1.0127351908456712`
- Top-1: `1.0`
- Top-3: `1.0`
- Macro F1: `1.0`
- Weighted F1: `1.0`
