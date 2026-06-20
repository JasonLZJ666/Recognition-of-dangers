# 危险标志识别项目

本项目完成了危险标志素材整理、天津大学风格识别系统界面、离线检测原型、增强版深度学习训练脚本和设计报告。

## 快速运行

如果只看离线演示，可以直接打开：

```text
app/index.html
```

如果要加载训练出来的 ONNX 模型，请用本地服务器打开：

```powershell
cd C:\Users\23950\Desktop\proj\danger-sign-recognition
powershell -ExecutionPolicy Bypass -File app\start_server.ps1
```

然后访问：

```text
http://localhost:8000
```

默认登录：

```text
账号：admin
密码：123456
```

系统支持图片检测、视频检测和摄像头检测。摄像头模式需要浏览器允许摄像头权限。训练模型通过 ONNX Runtime Web 加载，浏览器通常不允许 `file://` 页面读取 ONNX 文件，因此正式演示请使用上面的本地服务器方式。

## 项目结构

```text
danger-sign-recognition/
  app/                  天津大学风格前端识别系统
  dataset/              从 PDF 素材提取的 5 类标准样本
  docs/                 设计报告与报告图片
  model/                标签文件、分模块训练代码与 ONNX 导出脚本
```

## 已识别类别

| 类别 ID | 中文名称 | 风险等级 |
|---|---|---|
| flammable | 易燃危险 | 高风险 |
| falling_rocks | 落石危险 | 高风险 |
| water_hazard | 水域危险 | 中高风险 |
| falling_objects | 高处坠物 | 高风险 |
| explosion | 爆炸危险 | 极高风险 |

## 模型说明

前端系统已接入 ONNX Runtime Web，会优先加载 `app/model/danger_sign_model.onnx` 进行训练模型推理。如果 ONNX 模型或浏览器运行库加载失败，系统会自动回退到无依赖离线识别引擎：它会把输入图像归一化为危险标志特征图，再与 5 类标准样本进行相似度匹配，适合课程演示、系统验收和离线运行。

训练脚本升级为更完整的深度学习训练管线，支持：

- `efficientnet_b0`
- `resnet18`
- `mobilenet_v3_small`
- `strong_cnn`

训练过程包含随机仿射、透视扰动、颜色扰动、RandAugment、AdamW、CosineAnnealingLR、Label Smoothing、AMP 混合精度、早停、Top-1/Top-3、宏平均 F1、加权 F1、逐类别 Precision/Recall/F1、训练历史、混淆矩阵、数据集审计、实验报告、模型卡片和最佳模型保存。

深度学习训练脚本位于：

```text
model/train_model.py
```

训练代码已拆分为多个文件：

```text
model/constants.py          路径、类别和归一化常量
model/config.py             训练参数数据类和配置序列化
model/datasets.py           数据集读取和数据增强
model/architectures.py      EfficientNet/ResNet/MobileNet/带注意力 StrongCNN 构建
model/metrics.py            Top-K、F1、Precision/Recall 和混淆矩阵
model/callbacks.py          EarlyStopping 和 CheckpointWriter
model/model_utils.py        参数量、模型体积、环境快照等工具
model/reporting.py          CSV、JSON、实验报告和模型卡片导出
model/data_audit.py         数据集尺寸、数量、重复哈希审计
model/training_engine.py    单轮训练/验证循环和指标汇总
model/evaluate_model.py     独立 checkpoint 评估和分类报告导出
model/train_model.py        命令行入口和训练编排
model/export_onnx.py        PyTorch 权重导出 ONNX
```

在安装 PyTorch、torchvision、Pillow 后运行：

```text
python model/train_model.py --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30
```

如果网络无法下载 ImageNet 预训练权重，可以先使用无预训练版本：

```text
python model/train_model.py --arch efficientnet_b0 --epochs 30
```

脚本会基于 `dataset/<class_name>/` 的图片进行数据增强训练，并输出：

```text
model/artifacts/best_danger_sign_model.pt
model/artifacts/danger_sign_model_final.pt
model/artifacts/history.csv
model/artifacts/history.json
model/artifacts/confusion_matrix.json
model/artifacts/training_config.json
model/artifacts/dataset_audit.json
model/artifacts/experiment_report.json
model/artifacts/model_card.md
model/artifacts/classification_report.json
```

训练完成后也可以单独评估 checkpoint：

```text
python model/evaluate_model.py --checkpoint model/artifacts/best_danger_sign_model.pt
```

## 导入模型到网页前端

训练完成后，把 PyTorch 模型导出为 ONNX：

```powershell
python model\export_onnx.py
```

导出结果：

```text
app/model/danger_sign_model.onnx
app/model/danger_sign_model.onnx.data
app/model/model_metadata.json
```

网页端会自动读取这些文件，不需要手动在页面里选择模型。

## 交付清单

- 自主整理数据：`dataset/`
- 模型选型与训练脚本：`model/train_model.py`
- 模型导出脚本：`model/export_onnx.py`
- 系统登录界面：`app/index.html`
- 图片/视频/摄像头检测界面：`app/index.html`
- 设计报告：`docs/危险标志识别系统设计报告.docx`
