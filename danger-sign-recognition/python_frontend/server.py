# -*- coding: utf-8 -*-
"""Python web frontend for the danger sign recognition model.

This server loads the trained PyTorch checkpoint on startup and performs
inference on the Python side. The browser only sends images or camera frames,
so this frontend does not depend on ONNX Runtime Web.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures import build_model  # noqa: E402
from model.constants import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model.evaluate_frontend_photos import browser_signal_bounds, square_bounds  # noqa: E402


SIGN_INFO = {
    "explosion": {
        "name": "爆炸风险",
        "risk": "极高风险",
        "action": "停止明火作业，迅速撤离危险区域。",
        "asset": "/assets/signs/explosion.png",
    },
    "falling_objects": {
        "name": "高处坠物",
        "risk": "高风险",
        "action": "佩戴安全帽，避开吊装和高处作业区域。",
        "asset": "/assets/signs/falling_objects.png",
    },
    "falling_rocks": {
        "name": "落石危险",
        "risk": "高风险",
        "action": "远离边坡和山体，快速通过危险路段。",
        "asset": "/assets/signs/falling_rocks.png",
    },
    "flammable": {
        "name": "易燃危险",
        "risk": "高风险",
        "action": "隔离火源，保持通风，禁止吸烟和明火。",
        "asset": "/assets/signs/flammable.png",
    },
    "water_hazard": {
        "name": "水域危险",
        "risk": "中高风险",
        "action": "远离水边，设置围挡和警戒线。",
        "asset": "/assets/signs/water_hazard.png",
    },
}

DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "model" / "artifacts_viewpoint" / "best_danger_sign_model.pt",
)


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        checkpoint = path if path.is_absolute() else PROJECT_ROOT / path
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    for checkpoint in DEFAULT_CHECKPOINTS:
        if checkpoint.exists():
            return checkpoint
    candidates = "\n".join(f"  - {item}" for item in DEFAULT_CHECKPOINTS)
    raise FileNotFoundError(f"no checkpoint found. Tried:\n{candidates}")


def decode_data_url(value: str) -> bytes:
    if "," in value and value.strip().lower().startswith("data:"):
        _, encoded = value.split(",", 1)
    else:
        encoded = value
    return base64.b64decode(encoded)


def softmax(values: torch.Tensor) -> list[float]:
    return torch.softmax(values, dim=1)[0].detach().cpu().tolist()


class DangerSignPredictor:
    def __init__(self, checkpoint_path: Path, device_name: str = "auto", crop_padding: float = 0.18) -> None:
        self.checkpoint_path = checkpoint_path
        self.crop_padding = crop_padding
        self.device = self._resolve_device(device_name)
        self.model, self.metadata = self._load_model()
        self.classes = list(self.metadata["classes"])
        self.image_size = int(self.metadata.get("image_size") or 224)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _load_model(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        started = time.perf_counter()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        metadata = dict(checkpoint["metadata"])
        train_config = metadata.get("training_config") or metadata.get("args", {})
        classes = list(metadata["classes"])
        model = build_model(
            arch=str(metadata["arch"]),
            num_classes=len(classes),
            pretrained=False,
            dropout=float(train_config.get("dropout", 0.35)),
            freeze_backbone=False,
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model.to(self.device)
        metadata["load_seconds"] = round(time.perf_counter() - started, 3)
        return model, metadata

    def summary(self) -> dict[str, Any]:
        train_config = self.metadata.get("training_config") or self.metadata.get("args", {})
        return {
            "loaded": True,
            "checkpoint": str(self.checkpoint_path.relative_to(PROJECT_ROOT)),
            "arch": self.metadata.get("arch"),
            "device": str(self.device),
            "classes": self.classes,
            "imageSize": self.image_size,
            "dataset": train_config.get("dataset"),
            "bestEpoch": self.metadata.get("best_epoch"),
            "bestMetricName": self.metadata.get("best_metric_name"),
            "bestMetricValue": self.metadata.get("best_metric_value"),
            "loadSeconds": self.metadata.get("load_seconds"),
        }

    def _preprocess(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        bounds = browser_signal_bounds(image)
        bounds = square_bounds(bounds, image.size, self.crop_padding)
        cropped = image.crop(bounds)
        tensor = self.transform(cropped).unsqueeze(0).to(self.device)
        return tensor, bounds

    def predict(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        tensor, bounds = self._preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = softmax(logits)

        candidates = []
        for index, class_id in enumerate(self.classes):
            info = SIGN_INFO.get(class_id, {})
            candidates.append(
                {
                    "id": class_id,
                    "name": info.get("name", class_id),
                    "risk": info.get("risk", "未知"),
                    "action": info.get("action", "请人工复核。"),
                    "asset": info.get("asset", ""),
                    "confidence": float(probabilities[index]),
                }
            )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "ok": True,
            "backend": "PyTorch Python",
            "latencyMs": round((time.perf_counter() - started) * 1000),
            "crop": {"x1": bounds[0], "y1": bounds[1], "x2": bounds[2], "y2": bounds[3]},
            "best": candidates[0],
            "candidates": candidates,
            "model": self.summary(),
        }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>危险标志识别 Python 前端</title>
  <style>
    :root {
      --tju-blue: #004b8d;
      --ink: #111827;
      --muted: #6b7280;
      --line: #d8dee8;
      --soft: #f5f7fb;
      --panel: #ffffff;
      --accent: #d8aa36;
      --danger: #b91c1c;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--soft);
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: linear-gradient(180deg, #fff 0, #f3f6fb 46%, #eef3f9 100%); }
    button, input { font: inherit; }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 280px minmax(0, 1fr); }
    .side { background: var(--tju-blue); color: #fff; padding: 28px 22px; display: flex; flex-direction: column; gap: 22px; }
    .brand { display: flex; gap: 14px; align-items: center; }
    .mark { width: 54px; height: 54px; border: 2px solid rgba(255,255,255,.85); display: grid; place-items: center; font-weight: 800; }
    .brand h1 { font-size: 18px; margin: 0; line-height: 1.3; }
    .brand p, .status-card p { margin: 4px 0 0; color: rgba(255,255,255,.72); font-size: 12px; }
    .status-card { border-top: 1px solid rgba(255,255,255,.25); padding-top: 18px; }
    .status-card strong { display: block; font-size: 14px; margin-top: 8px; word-break: break-all; }
    .samples { display: grid; gap: 10px; }
    .sample { display: flex; align-items: center; gap: 10px; padding: 10px; border: 1px solid rgba(255,255,255,.2); background: rgba(255,255,255,.08); }
    .sample img { width: 40px; height: 40px; object-fit: contain; background: #fff; }
    .sample span { font-size: 13px; }
    .main { padding: 28px; display: grid; gap: 18px; }
    .topbar { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; }
    .eyebrow { margin: 0 0 6px; color: var(--tju-blue); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    h2 { margin: 0; font-size: 28px; letter-spacing: 0; }
    .pill { border: 1px solid var(--line); background: #fff; padding: 10px 12px; font-size: 13px; color: var(--muted); min-width: 160px; text-align: center; }
    .grid { display: grid; grid-template-columns: minmax(320px, 1.1fr) minmax(320px, .9fr); gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); padding: 18px; box-shadow: 0 18px 48px rgba(18, 38, 63, .08); }
    .panel h3 { margin: 0 0 14px; font-size: 18px; }
    .drop { border: 1.5px dashed #aab6c8; min-height: 132px; display: grid; place-items: center; text-align: center; padding: 18px; background: #fbfcff; cursor: pointer; }
    .drop:hover { border-color: var(--tju-blue); background: #f4f8fd; }
    .drop input { display: none; }
    .drop strong { display: block; margin-bottom: 6px; }
    .drop small { color: var(--muted); }
    .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
    .btn { border: 0; background: var(--tju-blue); color: #fff; padding: 10px 14px; cursor: pointer; }
    .btn.secondary { background: #e7edf5; color: var(--ink); }
    .btn.warn { background: var(--danger); }
    .btn:disabled { opacity: .5; cursor: not-allowed; }
    .preview { width: 100%; aspect-ratio: 16 / 10; background: #111; display: grid; place-items: center; overflow: hidden; position: relative; }
    .preview img, .preview video { width: 100%; height: 100%; object-fit: contain; background: #111; }
    .hint { color: var(--muted); font-size: 13px; margin: 12px 0 0; line-height: 1.7; }
    .result-head { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
    .result-head img { width: 72px; height: 72px; object-fit: contain; border: 1px solid var(--line); background: #fff; }
    .result-name { margin: 0; font-size: 24px; }
    .risk { display: inline-block; margin-top: 6px; color: #fff; background: var(--danger); padding: 4px 8px; font-size: 12px; }
    .score-list { display: grid; gap: 10px; }
    .score-row { display: grid; gap: 6px; }
    .score-label { display: flex; justify-content: space-between; font-size: 13px; }
    .track { height: 8px; background: #e5eaf2; overflow: hidden; }
    .fill { height: 100%; background: var(--tju-blue); width: 0%; transition: width .2s ease; }
    .meta { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
    .meta div { background: #f7f9fc; border: 1px solid var(--line); padding: 10px; }
    .meta span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .meta strong { font-size: 13px; word-break: break-all; }
    @media (max-width: 900px) {
      .shell { grid-template-columns: 1fr; }
      .side { min-height: auto; }
      .grid { grid-template-columns: 1fr; }
      .topbar { display: grid; }
      .meta { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="side">
      <div class="brand">
        <div class="mark">TJU</div>
        <div>
          <h1>天津大学计算机科学与技术学院<br />危险标志识别</h1>
          <p>模型在服务启动时自动导入</p>
        </div>
      </div>
      <section class="status-card">
        <p>当前模型</p>
        <strong id="checkpoint">加载中...</strong>
        <p id="modelLine">PyTorch 后端</p>
      </section>
      <section class="samples" id="sampleList"></section>
    </aside>
    <main class="main">
      <header class="topbar">
        <div>
          <p class="eyebrow">Peiyang Safety AI</p>
          <h2>图片与摄像头识别</h2>
        </div>
        <div id="statusPill" class="pill">连接模型中</div>
      </header>

      <section class="grid">
        <div class="panel">
          <h3>检测输入</h3>
          <label class="drop" id="dropZone">
            <input id="fileInput" type="file" accept="image/*" />
            <span>
              <strong>选择或拖入一张图片</strong>
              <small>JPG / PNG / WEBP，后端自动裁剪并推理</small>
            </span>
          </label>
          <div class="actions">
            <button class="btn" id="cameraStart" type="button">开启摄像头</button>
            <button class="btn secondary" id="captureFrame" type="button" disabled>识别当前画面</button>
            <button class="btn secondary" id="liveToggle" type="button" disabled>实时识别</button>
            <button class="btn warn" id="cameraStop" type="button" disabled>关闭摄像头</button>
          </div>
          <p class="hint" id="message">等待输入。这个页面不使用 ONNX Web，推理全部在 Python 后端完成。</p>
          <div class="preview">
            <img id="previewImage" alt="预览图" />
            <video id="cameraVideo" autoplay muted playsinline hidden></video>
          </div>
          <canvas id="captureCanvas" hidden></canvas>
        </div>

        <div class="panel">
          <h3>识别结果</h3>
          <div class="result-head">
            <img id="resultIcon" alt="" />
            <div>
              <p class="result-name" id="resultName">等待检测</p>
              <span class="risk" id="riskBadge">--</span>
            </div>
          </div>
          <div class="score-list" id="scoreList"></div>
          <p class="hint" id="actionText">请上传图片或开启摄像头。</p>
          <div class="meta">
            <div><span>后端</span><strong id="backendValue">Python</strong></div>
            <div><span>耗时</span><strong id="latencyValue">-- ms</strong></div>
            <div><span>裁剪框</span><strong id="cropValue">--</strong></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const signs = [
      ["explosion", "爆炸风险", "/assets/signs/explosion.png"],
      ["falling_objects", "高处坠物", "/assets/signs/falling_objects.png"],
      ["falling_rocks", "落石危险", "/assets/signs/falling_rocks.png"],
      ["flammable", "易燃危险", "/assets/signs/flammable.png"],
      ["water_hazard", "水域危险", "/assets/signs/water_hazard.png"]
    ];
    const els = {
      checkpoint: document.querySelector("#checkpoint"),
      modelLine: document.querySelector("#modelLine"),
      statusPill: document.querySelector("#statusPill"),
      fileInput: document.querySelector("#fileInput"),
      dropZone: document.querySelector("#dropZone"),
      message: document.querySelector("#message"),
      previewImage: document.querySelector("#previewImage"),
      cameraVideo: document.querySelector("#cameraVideo"),
      captureCanvas: document.querySelector("#captureCanvas"),
      cameraStart: document.querySelector("#cameraStart"),
      captureFrame: document.querySelector("#captureFrame"),
      liveToggle: document.querySelector("#liveToggle"),
      cameraStop: document.querySelector("#cameraStop"),
      resultIcon: document.querySelector("#resultIcon"),
      resultName: document.querySelector("#resultName"),
      riskBadge: document.querySelector("#riskBadge"),
      scoreList: document.querySelector("#scoreList"),
      actionText: document.querySelector("#actionText"),
      backendValue: document.querySelector("#backendValue"),
      latencyValue: document.querySelector("#latencyValue"),
      cropValue: document.querySelector("#cropValue"),
      sampleList: document.querySelector("#sampleList")
    };
    let stream = null;
    let liveTimer = null;
    let busy = false;

    function percent(value) {
      return `${Math.round(value * 100)}%`;
    }

    function setMessage(text, isError = false) {
      els.message.textContent = text;
      els.message.style.color = isError ? "#b91c1c" : "#6b7280";
    }

    function renderSamples() {
      els.sampleList.innerHTML = signs.map(([id, name, asset]) => `
        <div class="sample">
          <img src="${asset}" alt="${name}">
          <span>${name}</span>
        </div>
      `).join("");
    }

    async function loadStatus() {
      const response = await fetch("/api/status");
      const status = await response.json();
      els.checkpoint.textContent = status.checkpoint;
      els.modelLine.textContent = `${status.arch} · ${status.device} · ${status.classes.length} 类`;
      els.statusPill.textContent = "模型已加载";
    }

    function updateResult(data) {
      const best = data.best;
      els.resultIcon.src = best.asset || "";
      els.resultName.textContent = best.name;
      els.riskBadge.textContent = `${best.risk} · ${percent(best.confidence)}`;
      els.actionText.textContent = best.action;
      els.backendValue.textContent = data.backend;
      els.latencyValue.textContent = `${data.latencyMs} ms`;
      els.cropValue.textContent = `${data.crop.x1},${data.crop.y1} - ${data.crop.x2},${data.crop.y2}`;
      els.scoreList.innerHTML = data.candidates.map((item) => `
        <div class="score-row">
          <div class="score-label"><span>${item.name}</span><strong>${percent(item.confidence)}</strong></div>
          <div class="track"><div class="fill" style="width:${Math.round(item.confidence * 100)}%"></div></div>
        </div>
      `).join("");
    }

    async function predictDataUrl(dataUrl, sourceLabel) {
      if (busy) return;
      busy = true;
      setMessage(`正在识别：${sourceLabel}`);
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "识别失败");
        updateResult(data);
        setMessage(`识别完成：${sourceLabel}`);
      } catch (error) {
        setMessage(error.message || String(error), true);
      } finally {
        busy = false;
      }
    }

    function readFile(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result;
        els.previewImage.src = dataUrl;
        els.previewImage.hidden = false;
        els.cameraVideo.hidden = true;
        predictDataUrl(dataUrl, file.name);
      };
      reader.readAsDataURL(file);
    }

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setMessage("当前浏览器不支持摄像头调用", true);
        return;
      }
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      els.cameraVideo.srcObject = stream;
      els.cameraVideo.hidden = false;
      els.previewImage.hidden = true;
      els.captureFrame.disabled = false;
      els.liveToggle.disabled = false;
      els.cameraStop.disabled = false;
      setMessage("摄像头已开启");
    }

    function captureCameraFrame() {
      if (!stream || els.cameraVideo.readyState < 2) return;
      const canvas = els.captureCanvas;
      const video = els.cameraVideo;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      predictDataUrl(canvas.toDataURL("image/jpeg", 0.9), "摄像头画面");
    }

    function stopCamera() {
      if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
        els.liveToggle.textContent = "实时识别";
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      els.cameraVideo.srcObject = null;
      els.captureFrame.disabled = true;
      els.liveToggle.disabled = true;
      els.cameraStop.disabled = true;
      setMessage("摄像头已关闭");
    }

    els.fileInput.addEventListener("change", (event) => readFile(event.target.files[0]));
    els.dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      els.dropZone.style.borderColor = "#004b8d";
    });
    els.dropZone.addEventListener("dragleave", () => {
      els.dropZone.style.borderColor = "#aab6c8";
    });
    els.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      els.dropZone.style.borderColor = "#aab6c8";
      readFile(event.dataTransfer.files[0]);
    });
    els.cameraStart.addEventListener("click", startCamera);
    els.captureFrame.addEventListener("click", captureCameraFrame);
    els.cameraStop.addEventListener("click", stopCamera);
    els.liveToggle.addEventListener("click", () => {
      if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = null;
        els.liveToggle.textContent = "实时识别";
      } else {
        captureCameraFrame();
        liveTimer = setInterval(captureCameraFrame, 1200);
        els.liveToggle.textContent = "停止实时";
      }
    });

    renderSamples();
    loadStatus().catch((error) => setMessage(error.message || String(error), true));
  </script>
</body>
</html>
"""


class PythonFrontendHandler(BaseHTTPRequestHandler):
    predictor: DangerSignPredictor

    server_version = "DangerSignPythonFrontend/1.0"

    def _send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _not_found(self) -> None:
        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(self.predictor.summary())
            return
        if path.startswith("/assets/"):
            asset_path = (PROJECT_ROOT / "python_frontend" / path.lstrip("/")).resolve()
            assets_root = (PROJECT_ROOT / "python_frontend" / "assets").resolve()
            if assets_root not in asset_path.parents or not asset_path.exists() or not asset_path.is_file():
                self._not_found()
                return
            content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
            self._send_bytes(asset_path.read_bytes(), content_type)
            return
        self._not_found()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._not_found()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                raise ValueError("image request is empty or too large")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            image_value = payload.get("image")
            if not isinstance(image_value, str):
                raise ValueError("missing image data")
            image_bytes = decode_data_url(image_value)
            image = Image.open(io.BytesIO(image_bytes))
            result = self.predictor.predict(image)
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Python frontend for danger sign recognition")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a trained .pt checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--crop-padding", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"loading model: {checkpoint}", flush=True)
    predictor = DangerSignPredictor(checkpoint, device_name=args.device, crop_padding=args.crop_padding)
    PythonFrontendHandler.predictor = predictor
    print(
        "model loaded: "
        f"arch={predictor.metadata.get('arch')} "
        f"device={predictor.device} "
        f"classes={len(predictor.classes)} "
        f"time={predictor.metadata.get('load_seconds')}s",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), PythonFrontendHandler)
    print(f"open http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserver stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
