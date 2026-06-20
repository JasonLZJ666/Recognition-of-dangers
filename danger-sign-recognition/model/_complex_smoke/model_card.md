# Danger Sign Recognition Model Card

## Task
Classify five categories of warning and danger signs for a browser-based safety inspection demo.

## Model
- Architecture: `strong_cnn`
- Image size: `96`
- Pretrained: `False`
- Freeze backbone: `False`
- Total parameters: `258777`
- Trainable parameters: `258777`
- Estimated model size MB: `1.002`

## Training
- Best metric: `val_macro_f1` = `0.066667`
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
- Loss: `1.617717981338501`
- Top-1: `0.2`
- Top-3: `0.6`
- Macro F1: `0.06666666666666668`
- Weighted F1: `0.06666666666666668`
