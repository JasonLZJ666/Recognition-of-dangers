# 危险标志识别系统

本项目实现了一个面向危险警示标志的图像识别系统，支持图片上传、拖拽图片、摄像头单帧识别和摄像头实时识别。

当前系统识别 5 类危险标志：

| 类别 ID | 中文名称 | 风险等级 |
|---|---|---|
| `explosion` | 爆炸风险 | 极高风险 |
| `falling_objects` | 高处坠物 | 高风险 |
| `falling_rocks` | 落石危险 | 高风险 |
| `flammable` | 易燃危险 | 高风险 |
| `water_hazard` | 水域危险 | 中高风险 |

当前推荐使用 **Python 前端推理版**。网页负责采集图片或摄像头画面，Python 后端在启动时自动加载已经训练好的 PyTorch 模型并完成推理。这样不依赖浏览器 ONNX Runtime，现场演示更稳定。

## 1. 快速启动

进入项目目录：

```powershell
cd C:\Users\23950\Desktop\proj\danger-sign-recognition
```

安装依赖：

```powershell
pip install -r requirements.txt
```

启动系统：

```powershell
powershell -ExecutionPolicy Bypass -File python_frontend\start_python_frontend.ps1
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

启动成功后，系统会自动加载：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

PowerShell 窗口不要关闭，关闭后网页服务也会停止。

## 2. 使用方式

网页打开后可以进行以下操作：

- 上传图片识别
- 拖拽图片识别
- 开启摄像头并识别当前画面
- 开启实时识别，连续识别摄像头画面
- 查看预测类别、置信度、风险等级、处置建议和推理耗时

如果浏览器请求摄像头权限，需要点击允许。

## 3. 项目结构

```text
danger-sign-recognition/
  dataset/                  原始 5 类危险标志基础图片
  dataset_viewpoint/        视角和光照增强训练集，当前为 1105 张
  docs/                     技术报告、部署说明、答辩 PPT、演讲稿
  model/                    训练、评估、模型结构和数据增强代码
  python_frontend/          Python 前端和后端推理服务
  test_inputs/              前端演示测试图片
  requirements.txt          Python 依赖
  README.md                 项目说明
```

## 4. 模型说明

当前默认模型：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

模型结构：

```text
EfficientNet-B0
```

训练方式：

- 使用 ImageNet 预训练权重进行迁移学习
- 冻结 EfficientNet-B0 主干特征提取部分
- 替换并训练 5 分类输出头
- 使用 AdamW、学习率调度、Label Smoothing、Dropout 和早停机制
- 输出训练历史、混淆矩阵、模型配置和最佳权重

当前本地验证结果：

```text
dataset_test_viewpoint: Top-1 = 0.9975, Macro-F1 = 0.9975
test_inputs/frontend_photos: 30/30, Accuracy = 1.0000
```

这些结果表示模型在当前课程数据、增强测试集和前端测试照片范围内表现良好。如果要用于真实现场，还需要继续采集真实场景图片进行扩充和再训练。

## 5. 数据集说明

基础数据集：

```text
dataset/
```

增强训练集：

```text
dataset_viewpoint/
```

`dataset_viewpoint/` 由基础图片生成，包含旋转、缩放、透视、亮度、对比度、模糊、阴影和背景扰动，用来模拟不同拍摄角度和光照条件。

当前数量：

```text
5 类 x 221 张 = 1105 张
```

如果该文件夹被删除，可以重新生成：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_viewpoint --per-class 220 --size 320
```

## 6. 重新训练模型

推荐训练命令：

```powershell
python model\train_model.py --dataset dataset_viewpoint --out model\artifacts_viewpoint --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30 --batch-size 16 --repeats 1
```

训练完成后，新的最佳模型会保存到：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

重新启动 Python 前端后，系统会自动加载新的模型。

## 7. 评估模型

生成独立增强测试集：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_test_viewpoint --per-class 80 --size 320 --seed 9090 --no-source-copy
```

评估测试集：

```powershell
python model\evaluate_model.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset dataset_test_viewpoint --out model\artifacts_viewpoint\test_classification_report.json --repeats 1
```

评估前端测试照片：

```powershell
python model\evaluate_frontend_photos.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset test_inputs\frontend_photos --crop square --padding 0.18
```

## 8. Python 前端接口

Python 前端默认运行在：

```text
http://127.0.0.1:7860/
```

接口如下：

| 接口 | 说明 |
|---|---|
| `GET /` | 网页界面 |
| `GET /api/status` | 查看模型是否加载成功 |
| `POST /api/predict` | 提交 base64 图片并返回识别结果 |

可以通过下面地址检查模型状态：

```text
http://127.0.0.1:7860/api/status
```

如果返回中包含 `"loaded": true`，说明模型已经成功加载。

## 9. 主要代码文件

```text
model/constants.py                类别、路径和归一化参数
model/config.py                   训练配置
model/datasets.py                 数据集读取与增强
model/architectures.py            EfficientNet、ResNet、MobileNet、StrongCNN
model/training_engine.py          单轮训练和验证流程
model/metrics.py                  Accuracy、F1、混淆矩阵等指标
model/reporting.py                训练报告和模型卡片导出
model/data_audit.py               数据集审计
model/build_viewpoint_dataset.py  生成视角和光照增强数据集
model/evaluate_model.py           测试集评估
model/evaluate_frontend_photos.py 前端测试照片评估
model/train_model.py              训练入口
python_frontend/server.py         Python Web 服务和模型推理接口
```

## 10. 答辩材料

```text
docs/危险标志识别系统技术报告.md
docs/部署到其他电脑运行说明.md
docs/危险标志识别系统答辩PPT_精致版.pptx
docs/危险标志识别系统答辩演讲稿.md
```

如果要把项目复制到其他电脑运行，优先阅读：

```text
docs/部署到其他电脑运行说明.md
```

## 11. 常见问题

### PowerShell 提示无法运行脚本

使用下面命令启动：

```powershell
powershell -ExecutionPolicy Bypass -File python_frontend\start_python_frontend.ps1
```

### 提示没有 torch

重新安装依赖：

```powershell
pip install -r requirements.txt
```

### 提示找不到模型

检查文件是否存在：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

### 端口 7860 被占用

换一个端口启动：

```powershell
python python_frontend\server.py --port 7861
```

然后访问：

```text
http://127.0.0.1:7861/
```

## 12. 交付清单

- 训练代码：`model/`
- 当前最佳模型：`model/artifacts_viewpoint/best_danger_sign_model.pt`
- 原始数据集：`dataset/`
- 增强训练集：`dataset_viewpoint/`
- Python 前端：`python_frontend/`
- 测试图片：`test_inputs/`
- 技术报告和答辩材料：`docs/`
