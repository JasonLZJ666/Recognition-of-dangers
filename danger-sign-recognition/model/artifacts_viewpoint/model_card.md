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
- Best metric: `val_macro_f1` = `0.994562`
- Best epoch: `15`
- Device: `cpu`
- PyTorch: `2.12.1+cpu`

## Classes
- `explosion`
- `falling_objects`
- `falling_rocks`
- `flammable`
- `water_hazard`

## Validation Summary
- Loss: `0.33707927141793714`
- Top-1: `0.9945701357466064`
- Top-3: `1.0`
- Macro F1: `0.9945615449320672`
- Weighted F1: `0.9945615449320672`
