# Python 前端启动说明

这个前端不使用浏览器 ONNX Runtime。启动 `server.py` 时会自动加载训练好的 PyTorch checkpoint，网页上传图片或摄像头截图后，由 Python 后端完成识别。

## 启动

```powershell
cd C:\Users\23950\Desktop\proj\danger-sign-recognition
powershell -ExecutionPolicy Bypass -File python_frontend\start_python_frontend.ps1
```

打开：

```text
http://127.0.0.1:7860/
```

默认加载：

```text
model\artifacts_viewpoint\best_danger_sign_model.pt
```

## 接口

- `GET /`：网页前端
- `GET /api/status`：查看当前加载的模型
- `POST /api/predict`：提交 base64 图片并返回识别结果

## 指定模型

```powershell
python python_frontend\server.py --checkpoint model\artifacts_viewpoint\best_danger_sign_model.pt
```
