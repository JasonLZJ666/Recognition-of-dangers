# 前端测试照片

这个目录用于存放网页前端手动上传测试用的照片，不作为训练集直接使用。

当前已生成：

```text
frontend_photos/
  explosion/          6 张
  falling_objects/    6 张
  falling_rocks/      6 张
  flammable/          6 张
  water_hazard/       6 张
  _contact_sheet.jpg  汇总预览图
```

这些照片由 `dataset/` 中的 5 类原始危险标志合成，保留原始类别，只模拟真实拍摄时常见的变化：

- 不同背景：墙面、室外、实验室桌面；
- 不同角度：轻微旋转、偏移；
- 不同光线：亮度、对比度变化；
- 轻微模糊和阴影。

## 使用方法

先启动网页：

```powershell
cd C:\Users\23950\Desktop\proj\danger-sign-recognition
powershell -ExecutionPolicy Bypass -File app\start_server.ps1
```

然后打开：

```text
http://localhost:8000
```

在网页的图片识别区域上传 `test_inputs/frontend_photos/` 下任意 `.jpg` 图片即可测试模型识别效果。

## 重新生成

如需重新生成一批不同随机效果的照片：

```powershell
python model\build_frontend_test_photos.py --source dataset --out test_inputs\frontend_photos --per-class 6 --seed 8102026
```

修改 `--per-class` 可以控制每个类别生成多少张，修改 `--seed` 可以得到不同角度和光线组合。
