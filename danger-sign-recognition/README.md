# 危险标志识别系统

本项目用于识别 5 类危险警示标志：爆炸风险、高处坠物、落石危险、易燃危险和水域危险。

当前保留的是 **Python 前端推理版**：网页负责上传图片和采集摄像头画面，Python 后端在启动时自动加载已经训练好的 PyTorch 模型并完成识别。这个版本不依赖浏览器 ONNX Runtime，演示和部署更稳定。

## 快速启动

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

启动时默认加载：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

PowerShell 窗口不要关闭，关闭后网页服务也会停止。

## 核心功能

- 图片上传识别
- 拖拽图片识别
- 摄像头单帧识别
- 摄像头实时识别
- 启动时自动导入训练好的 `.pt` 模型
- 返回 Top-5 置信度、风险等级、处置建议和推理耗时

## 模型信息

当前推荐模型：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

模型结构：

```text
EfficientNet-B0
```

类别顺序：

| 类别 ID | 中文名称 | 风险等级 |
|---|---|---|
| `explosion` | 爆炸风险 | 极高风险 |
| `falling_objects` | 高处坠物 | 高风险 |
| `falling_rocks` | 落石危险 | 高风险 |
| `flammable` | 易燃危险 | 高风险 |
| `water_hazard` | 水域危险 | 中高风险 |

当前训练方案只围绕附件中的 5 类危险标志展开，通过旋转、缩放、透视、亮度、对比度、模糊、阴影和背景扰动生成不同角度与光线下的样本。

本地验证结果：

```text
dataset_test_viewpoint: Top-1 = 0.9975, Macro-F1 = 0.9975
test_inputs/frontend_photos: 30/30, Accuracy = 1.0000
```

## 重新训练

如果清理后的项目中没有 `dataset_viewpoint/`，先重新生成增强训练集：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_viewpoint --per-class 220 --size 320
```

训练模型：

```powershell
python model\train_model.py --dataset dataset_viewpoint --out model\artifacts_viewpoint --arch efficientnet_b0 --pretrained --freeze-backbone --epochs 30 --batch-size 16 --repeats 1
```

训练完成后，Python 前端会默认读取：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

## 评估模型

生成独立测试集并评估：

```powershell
python model\build_viewpoint_dataset.py --source dataset --out dataset_test_viewpoint --per-class 80 --size 320 --seed 9090 --no-source-copy
python model\evaluate_model.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset dataset_test_viewpoint --out model\artifacts_viewpoint\test_classification_report.json --repeats 1
```

评估前端测试照片：

```powershell
python model\evaluate_frontend_photos.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt --dataset test_inputs\frontend_photos --crop square --padding 0.18
```

## 项目结构

```text
danger-sign-recognition/
  dataset/                 附件整理出的 5 类基础样本
  docs/                    技术报告、部署说明、答辩 PPT、演讲稿
  model/                   训练、评估、模型结构和数据增强代码
  python_frontend/         当前推荐使用的网页前端和 Python 推理后端
  test_inputs/             前端演示测试照片
  requirements.txt         Python 依赖
  README.md                项目入口说明
```

## 训练代码结构

```text
model/constants.py               路径、类别、归一化参数
model/config.py                  训练配置
model/datasets.py                数据集读取与增强
model/architectures.py           EfficientNet/ResNet/MobileNet/StrongCNN
model/training_engine.py         单轮训练与验证循环
model/metrics.py                 准确率、F1、混淆矩阵
model/reporting.py               训练报告和模型卡片导出
model/data_audit.py              数据集审计
model/build_viewpoint_dataset.py 生成视角和光照增强数据
model/evaluate_model.py          独立测试集评估
model/evaluate_frontend_photos.py 前端照片评估
model/train_model.py             训练入口
```

## Python 前端接口

```text
GET  /              网页界面
GET  /api/status    查看当前加载的模型
POST /api/predict   提交 base64 图片并返回识别结果
```

## 交付材料

- 当前最佳模型：`model/artifacts_viewpoint/best_danger_sign_model.pt`
- Python 前端：`python_frontend/server.py`
- 一键启动脚本：`python_frontend/start_python_frontend.ps1`
- 技术报告：`docs/危险标志识别系统技术报告.md`
- 部署说明：`docs/部署到其他电脑运行说明.md`
- 答辩 PPT：`docs/危险标志识别系统答辩PPT_精致版.pptx`
- 答辩演讲稿：`docs/危险标志识别系统答辩演讲稿.md`
