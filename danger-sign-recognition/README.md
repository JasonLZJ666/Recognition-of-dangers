# 危险标志识别项目

本项目完成了危险标志素材整理、苹果官网式极简识别系统界面、离线检测原型、增强版深度学习训练脚本和设计报告。

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
  app/                  苹果官网式极简前端识别系统
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

## 可扩展数据集

当前项目自带的是从 PDF 提取的 5 类标准样本。若要提高模型真实场景准确率，可以继续补充公开数据集和自采数据。推荐优先从以下来源扩展：

| 数据集 | 适合用途 | 说明 |
|---|---|---|
| TT100K / Tsinghua-Tencent 100K | 中文道路交通标志、真实街景检测 | 清华-腾讯公开交通标志数据，来自 100000 张腾讯街景全景图，包含约 30000 个交通标志实例，提供类别、框和像素级标注。适合补充落石、水域、施工警告等真实道路标志。 |
| GTSRB | 交通标志分类预训练/对比实验 | 德国交通标志识别基准，40 多类、5 万多张图片，适合做分类模型基线和迁移学习对照。 |
| Mapillary Traffic Sign Dataset | 全球多场景交通标志检测/分类 | 包含全球街景、天气和光照变化，论文介绍为 100K 街景图、300 多个交通标志类别，适合增强泛化能力。 |
| GLARE Dataset | 强光、眩光鲁棒性测试 | 面向强太阳眩光场景的交通标志检测数据，可用于测试模型在逆光、过曝情况下是否稳定。 |
| GHS hazard pictograms | 易燃、爆炸等化学危险图标补充 | GHS 是化学品危险图形符号体系，可用于补充易燃、爆炸、腐蚀、毒性等标准图标样本；但它更偏标准图标，不等同于真实摄像头场景。 |

建议扩展时仍按当前目录结构整理：

```text
dataset/
  flammable/
  falling_rocks/
  water_hazard/
  falling_objects/
  explosion/
```

如果公开数据集的类别名称和本项目不完全一致，先筛选相近类别，再人工检查标签。例如 `falling_rocks` 可以优先找落石/山体滑坡/边坡警告标志，`flammable` 和 `explosion` 可以补充 GHS 或危化品运输标志。

### 已下载的数据集

本项目当前已经下载并整理了两类外部数据：

```text
external_datasets/raw/GTSRB_Final_Training_Images.zip
external_datasets/raw/GTSRB_Final_Training_Images/
external_datasets/prepared/gtsrb_warning/
external_datasets/raw/ghs_pictograms/
external_datasets/prepared/ghs_pictograms/
dataset_extended/
dataset_all/
dataset_viewpoint/
dataset_test_viewpoint/
```

其中：

- `external_datasets/prepared/gtsrb_warning/` 是从 GTSRB 中筛选出来的 10 类警告标志辅助数据集，共 5550 张图，可用于训练一个“交通警告标志”分类模型。
- `external_datasets/prepared/ghs_pictograms/` 包含 GHS 易燃和爆炸标准危险图标 PNG。
- `dataset_extended/` 是当前 5 类危险标志训练集的增强版本，保留原有 5 类目录，并把 GHS 易燃/爆炸图标补充进 `flammable` 和 `explosion` 类，适合直接训练网页前端对应的 5 类模型。
- `dataset_all/` 是方案 B 使用的 15 类混合数据集，由 `dataset_extended/` 的 5 类危险标志和 `gtsrb_warning/` 的 10 类交通警告标志合并而成，共 5557 张图，适合展示更大规模的训练实验。
- `dataset_viewpoint/` 是最终推荐使用的 5 类视角/光照增强数据集，只包含附件里的 5 种危险标志，不增加新类别；每类 221 张，共 1105 张，包含旋转、缩放、仿射倾斜、亮度变化、对比度变化、轻微模糊、阴影和背景扰动。
- `dataset_test_viewpoint/` 是独立测试集，只包含附件里的 5 种危险标志，不增加新类别；每类 80 张，共 400 张，使用不同随机种子生成，不复制干净原图，适合在训练完成后做测试集评估。

最终网页模型建议使用 `dataset_viewpoint/` 训练，因为最终识别目标仍然是附件里的 5 类危险标志，只是拍摄时会出现不同角度和光线：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_viewpoint --per-class 220 --size 320
python model\train_model.py --dataset dataset_viewpoint --out model\artifacts_viewpoint --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30 --batch-size 16 --repeats 1
python model\export_onnx.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt
```

生成独立测试集并评估训练结果：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_test_viewpoint --per-class 80 --size 320 --seed 9090 --no-source-copy
python model\evaluate_model.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset dataset_test_viewpoint --out model\artifacts_viewpoint\test_classification_report.json --repeats 1
```

前端真实上传照片的准确率主要受裁剪定位影响。当前版本网页端会优先定位黄色警示主体，再扩展为带留白的方形区域送入 ONNX 模型，避免把地面、墙面暗线等背景一起喂给模型。可用下面命令复现实验：

```powershell
python model\build_frontend_test_photos.py --per-class 6
python model\evaluate_frontend_photos.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset test_inputs\frontend_photos --crop square --padding 0.18
```

当前 `artifacts_viewpoint` 模型在 `test_inputs/frontend_photos/` 上评估为 `30/30`，准确率 `1.0000`；在 `dataset_test_viewpoint/` 上 Top-1 为 `0.9975`、Top-3 为 `1.0000`、Macro-F1 为 `0.9975`。

如果要继续做“真实拍照场景”微调，可以生成场景裁剪训练集，再从已有 checkpoint 继续训练：

```powershell
python model\build_scene_crop_dataset.py --source dataset --base dataset_viewpoint --out dataset_scene_finetune --per-class 100 --base-limit-per-class 50 --size 320 --padding 0.18
python model\train_model.py --dataset dataset_scene_finetune --out model\artifacts_scene_finetune --arch efficientnet_b0 --pretrained --freeze-backbone --init-checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --epochs 8 --batch-size 64 --image-size 192 --repeats 1 --lr 0.00016 --no-auto-augment
```

使用增强后的 5 类数据训练当前网页模型：

```powershell
python model\train_model.py --dataset dataset_extended --out model\artifacts_extended --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30
python model\export_onnx.py --checkpoint model\artifacts_extended\best_danger_sign_model.pt
```

使用 GTSRB 警告标志子集训练辅助模型：

```powershell
python model\train_model.py --dataset external_datasets\prepared\gtsrb_warning --out model\artifacts_gtsrb_warning --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 20
```

使用方案 B 的 15 类混合数据集训练模型：

```powershell
python model\train_model.py --dataset dataset_all --out model\artifacts_all --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 20 --batch-size 16 --repeats 1
```

注意：GTSRB 的类别体系和当前网页的 5 类危险标志不完全一致，所以不要直接把 `gtsrb_warning` 或 `dataset_all` 导出替换网页模型；它们更适合作为额外实验、预训练对照或报告里的大数据集扩展证明。若要让网页展示 15 类结果，需要同步扩展 `app/app.js` 里的 `SIGN_LIBRARY`。

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
model/build_viewpoint_dataset.py  从 5 张附件图生成角度/光照增强数据
model/build_scene_crop_dataset.py 从 5 类附件图生成前端场景裁剪微调数据
model/training_engine.py    单轮训练/验证循环和指标汇总
model/evaluate_model.py     独立 checkpoint 评估和分类报告导出
model/evaluate_frontend_photos.py 前端照片裁剪一致性评估
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
