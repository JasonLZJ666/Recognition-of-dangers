const SIGN_LIBRARY = [
  {
    id: "flammable",
    name: "易燃危险",
    risk: "高风险",
    action: "隔离火源，保持通风",
    asset: "assets/signs/flammable.png"
  },
  {
    id: "falling_rocks",
    name: "落石危险",
    risk: "高风险",
    action: "避开边坡，快速通过",
    asset: "assets/signs/falling_rocks.png"
  },
  {
    id: "water_hazard",
    name: "水域危险",
    risk: "中高风险",
    action: "远离水边，设置围挡",
    asset: "assets/signs/water_hazard.png"
  },
  {
    id: "falling_objects",
    name: "高处坠物",
    risk: "高风险",
    action: "佩戴安全帽，避开作业区",
    asset: "assets/signs/falling_objects.png"
  },
  {
    id: "explosion",
    name: "爆炸危险",
    risk: "极高风险",
    action: "停止明火，撤离危险区",
    asset: "assets/signs/explosion.png"
  }
];

const SIGN_BY_ID = new Map(SIGN_LIBRARY.map((sign) => [sign.id, sign]));
const MODEL_METADATA_URL = "model/model_metadata.json";
const MODEL_BASE_PATH = "model/";
const WASM_BASE_PATH = new URL("vendor/onnxruntime-web/", window.location.href).href;
const SAMPLE_SIZE = 72;
const REALTIME_INFERENCE_INTERVAL_MS = 520;
const REALTIME_EMA_ALPHA = 0.42;
const REALTIME_HISTORY_SIZE = 6;
const REALTIME_STABLE_VOTES = 3;
const PREVIEW_PARAMS = new URLSearchParams(window.location.search);
const PREVIEW_MODE = PREVIEW_PARAMS.get("preview") === "1";
const SKIP_ONNX = PREVIEW_PARAMS.get("noOnnx") === "1";
const state = {
  mode: "image",
  references: [],
  onnxSession: null,
  onnxMetadata: null,
  inferenceBackend: "signature",
  inferenceBusy: false,
  activeTimer: null,
  cameraStream: null,
  videoPlaying: false,
  realtimeSmoothing: null
};

const els = {
  loginScreen: document.querySelector("#loginScreen"),
  appScreen: document.querySelector("#appScreen"),
  loginForm: document.querySelector("#loginForm"),
  username: document.querySelector("#username"),
  password: document.querySelector("#password"),
  loginError: document.querySelector("#loginError"),
  logoutButton: document.querySelector("#logoutButton"),
  systemStatus: document.querySelector("#systemStatus"),
  modeLabel: document.querySelector("#modeLabel"),
  modeButtons: document.querySelectorAll(".mode-button"),
  sampleList: document.querySelector("#sampleList"),
  imageControls: document.querySelector("#imageControls"),
  videoControls: document.querySelector("#videoControls"),
  cameraControls: document.querySelector("#cameraControls"),
  imageInput: document.querySelector("#imageInput"),
  videoInput: document.querySelector("#videoInput"),
  pauseVideoButton: document.querySelector("#pauseVideoButton"),
  startCameraButton: document.querySelector("#startCameraButton"),
  stopCameraButton: document.querySelector("#stopCameraButton"),
  inputMessage: document.querySelector("#inputMessage"),
  previewCanvas: document.querySelector("#previewCanvas"),
  videoSource: document.querySelector("#videoSource"),
  cameraSource: document.querySelector("#cameraSource"),
  confidenceValue: document.querySelector("#confidenceValue"),
  resultThumb: document.querySelector("#resultThumb"),
  resultName: document.querySelector("#resultName"),
  riskBadge: document.querySelector("#riskBadge"),
  scoreList: document.querySelector("#scoreList"),
  detailMode: document.querySelector("#detailMode"),
  detailAction: document.querySelector("#detailAction"),
  backendValue: document.querySelector("#backendValue"),
  modelArchValue: document.querySelector("#modelArchValue"),
  latencyValue: document.querySelector("#latencyValue"),
  runtimeStatus: document.querySelector("#runtimeStatus"),
  inputSourceValue: document.querySelector("#inputSourceValue"),
  queueValue: document.querySelector("#queueValue")
};

const ctx = els.previewCanvas.getContext("2d", { willReadFrequently: true });

function setStatus(text) {
  els.systemStatus.textContent = text;
}

function setRuntime(text) {
  if (els.runtimeStatus) els.runtimeStatus.textContent = text;
}

function setBackend(text) {
  if (els.backendValue) els.backendValue.textContent = text;
}

function setInputSource(text) {
  if (els.inputSourceValue) els.inputSourceValue.textContent = text;
}

function setQueue(text) {
  if (els.queueValue) els.queueValue.textContent = text;
}

function setMessage(text, isError = false) {
  els.inputMessage.textContent = text;
  els.inputMessage.style.color = isError ? "var(--hazard-red)" : "var(--muted)";
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function sourceSize(source) {
  return {
    width: source.videoWidth || source.naturalWidth || source.width,
    height: source.videoHeight || source.naturalHeight || source.height
  };
}

function isLikelyYellowPixel(r, g, b, a) {
  if (a <= 20) return false;
  const maxChannel = Math.max(r, g, b);
  const minChannel = Math.min(r, g, b);
  const saturation = maxChannel === 0 ? 0 : (maxChannel - minChannel) / maxChannel;
  const yellowBalance = (r + g) / 2 - b;
  const relativeYellow = r > b + 24 && g > b + 18 && yellowBalance > 28;
  const brightYellow = r > 130 && g > 105 && b < 150 && r + g - b > 210;
  const shadowYellow = r > 70 && g > 58 && b < 130 && relativeYellow && saturation > 0.18;
  const highlightYellow = r > 190 && g > 170 && b < 175 && relativeYellow && saturation > 0.2;
  return brightYellow || shadowYellow || highlightYellow;
}

function countMaskInBox(mask, width, height, minX, minY, maxX, maxY, padding) {
  const left = Math.max(0, minX - padding);
  const top = Math.max(0, minY - padding);
  const right = Math.min(width - 1, maxX + padding);
  const bottom = Math.min(height - 1, maxY + padding);
  let count = 0;
  for (let y = top; y <= bottom; y += 1) {
    for (let x = left; x <= right; x += 1) {
      count += mask[y * width + x];
    }
  }
  return count;
}

function largestComponentBounds(mask, width, height, minPixels, supportMask = null) {
  const visited = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  let best = null;

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;

    let top = 0;
    let count = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    stack[top] = start;
    top += 1;
    visited[start] = 1;

    while (top > 0) {
      top -= 1;
      const index = stack[top];
      const x = index % width;
      const y = (index - x) / width;
      count += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [
        x > 0 ? index - 1 : -1,
        x < width - 1 ? index + 1 : -1,
        y > 0 ? index - width : -1,
        y < height - 1 ? index + width : -1
      ];

      for (const next of neighbors) {
        if (next >= 0 && mask[next] && !visited[next]) {
          visited[next] = 1;
          stack[top] = next;
          top += 1;
        }
      }
    }

    if (count >= minPixels) {
      const area = (maxX - minX + 1) * (maxY - minY + 1);
      const support = supportMask ? countMaskInBox(supportMask, width, height, minX, minY, maxX, maxY, 10) : 0;
      const score = supportMask
        ? (support > 0 ? support * 50 + count * 0.2 : count * 0.03)
        : count + Math.sqrt(area) * 0.2;
      if (!best || score > best.score) {
        best = { minX, minY, maxX, maxY, count, support, score };
      }
    }
  }

  return best;
}

function findSignalBounds(source) {
  const { width, height } = sourceSize(source);
  const maxScan = 300;
  const scale = Math.min(1, maxScan / Math.max(width, height));
  const scanWidth = Math.max(1, Math.round(width * scale));
  const scanHeight = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = scanWidth;
  canvas.height = scanHeight;
  const scanCtx = canvas.getContext("2d", { willReadFrequently: true });
  scanCtx.fillStyle = "#fff";
  scanCtx.fillRect(0, 0, scanWidth, scanHeight);
  scanCtx.drawImage(source, 0, 0, scanWidth, scanHeight);
  const data = scanCtx.getImageData(0, 0, scanWidth, scanHeight).data;
  const yellowMask = new Uint8Array(scanWidth * scanHeight);
  const signMask = new Uint8Array(scanWidth * scanHeight);
  const darkMask = new Uint8Array(scanWidth * scanHeight);

  for (let y = 0; y < scanHeight; y += 1) {
    for (let x = 0; x < scanWidth; x += 1) {
      const pixel = y * scanWidth + x;
      const i = pixel * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      const yellowPixel = isLikelyYellowPixel(r, g, b, a);
      const nonWhite = a > 20 && (r < 242 || g < 242 || b < 232);
      const darkStroke = r < 125 && g < 120 && b < 115;
      const warmStroke = r > 155 && g > 125 && b < 115;
      const signPixel = nonWhite && (yellowPixel || darkStroke || warmStroke);
      if (yellowPixel) yellowMask[pixel] = 1;
      if (darkStroke) darkMask[pixel] = 1;
      if (signPixel) signMask[pixel] = 1;
    }
  }

  const yellowBounds = largestComponentBounds(yellowMask, scanWidth, scanHeight, 35, darkMask);
  const signBounds = largestComponentBounds(signMask, scanWidth, scanHeight, 80);
  const selected = yellowBounds || signBounds;
  if (!selected) {
    return { x: 0, y: 0, width, height };
  }

  const padX = Math.max(4, Math.round((selected.maxX - selected.minX) * 0.08));
  const padY = Math.max(4, Math.round((selected.maxY - selected.minY) * 0.08));
  const minX = Math.max(0, selected.minX - padX);
  const minY = Math.max(0, selected.minY - padY);
  const maxX = Math.min(scanWidth - 1, selected.maxX + padX);
  const maxY = Math.min(scanHeight - 1, selected.maxY + padY);

  return {
    x: minX / scale,
    y: minY / scale,
    width: (maxX - minX + 1) / scale,
    height: (maxY - minY + 1) / scale
  };
}

function squareSignalBounds(bounds, source, padding = 0.28) {
  const { width: sourceWidth, height: sourceHeight } = sourceSize(source);
  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;
  const rawSide = Math.max(bounds.width, bounds.height) * (1 + padding * 2);
  const side = Math.max(1, Math.min(rawSide, sourceWidth, sourceHeight));
  const maxX = Math.max(0, sourceWidth - side);
  const maxY = Math.max(0, sourceHeight - side);
  const x = Math.min(Math.max(0, centerX - side / 2), maxX);
  const y = Math.min(Math.max(0, centerY - side / 2), maxY);
  return { x, y, width: side, height: side };
}

function pixelFeature(r, g, b) {
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  const darkness = Math.max(0, 1 - luminance);
  const yellow = r > 150 && g > 125 && b < 115 ? Math.min(1, (r + g - 2 * b) / 420) : 0;
  const black = r < 95 && g < 85 && b < 80 ? 1 : 0;
  return Math.max(black, darkness * 0.72, yellow * 0.48);
}

function buildSignature(source) {
  const bounds = squareSignalBounds(findSignalBounds(source), source);
  const canvas = document.createElement("canvas");
  canvas.width = SAMPLE_SIZE;
  canvas.height = SAMPLE_SIZE;
  const localCtx = canvas.getContext("2d", { willReadFrequently: true });
  localCtx.fillStyle = "#fff";
  localCtx.fillRect(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  localCtx.drawImage(
    source,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    SAMPLE_SIZE,
    SAMPLE_SIZE
  );
  const imageData = localCtx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE).data;
  const signature = new Float32Array(SAMPLE_SIZE * SAMPLE_SIZE);
  for (let i = 0, p = 0; i < imageData.length; i += 4, p += 1) {
    signature[p] = pixelFeature(imageData[i], imageData[i + 1], imageData[i + 2]);
  }
  return { signature, bounds };
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return sum / a.length;
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function preprocessForOnnx(source, metadata) {
  const bounds = squareSignalBounds(findSignalBounds(source), source);
  const size = metadata.imageSize || 224;
  const mean = metadata.mean || [0.485, 0.456, 0.406];
  const std = metadata.std || [0.229, 0.224, 0.225];
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const localCtx = canvas.getContext("2d", { willReadFrequently: true });
  localCtx.fillStyle = "#fff";
  localCtx.fillRect(0, 0, size, size);
  localCtx.drawImage(
    source,
    bounds.x,
    bounds.y,
    bounds.width,
    bounds.height,
    0,
    0,
    size,
    size
  );

  const imageData = localCtx.getImageData(0, 0, size, size).data;
  const tensorData = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x;
      const offset = pixel * 4;
      tensorData[pixel] = imageData[offset] / 255 - mean[0];
      tensorData[pixel] /= std[0];
      tensorData[plane + pixel] = imageData[offset + 1] / 255 - mean[1];
      tensorData[plane + pixel] /= std[1];
      tensorData[plane * 2 + pixel] = imageData[offset + 2] / 255 - mean[2];
      tensorData[plane * 2 + pixel] /= std[2];
    }
  }

  return { tensorData, bounds, size };
}

async function classifyWithOnnx(source) {
  const metadata = state.onnxMetadata;
  const session = state.onnxSession;
  const { tensorData, bounds, size } = preprocessForOnnx(source, metadata);
  const inputName = metadata.inputName || session.inputNames[0];
  const outputName = metadata.outputName || session.outputNames[0];
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, size, size]);
  const outputs = await session.run({ [inputName]: inputTensor });
  const outputTensor = outputs[outputName] || outputs[Object.keys(outputs)[0]];
  const probabilities = softmax(Array.from(outputTensor.data));

  const candidates = metadata.classes
    .map((id, index) => {
      const sign = SIGN_BY_ID.get(id) || {
        id,
        name: id,
        risk: "未知",
        action: "请人工复核",
        asset: ""
      };
      const confidence = probabilities[index] || 0;
      return { ...sign, confidence, distance: 1 - confidence };
    })
    .sort((a, b) => b.confidence - a.confidence);

  return {
    best: candidates[0],
    candidates,
    bounds,
    backend: "onnx"
  };
}

function classifyBySignature(source) {
  const { signature, bounds } = buildSignature(source);
  const candidates = state.references
    .map((reference) => {
      const d = distance(signature, reference.signature);
      const confidence = Math.max(0.02, Math.min(0.99, 1 - d * 4.8));
      return { ...reference, distance: d, confidence };
    })
    .sort((a, b) => a.distance - b.distance);

  return {
    best: candidates[0],
    candidates,
    bounds,
    backend: "signature"
  };
}

async function classify(source) {
  if (state.onnxSession && state.onnxMetadata && window.ort) {
    return classifyWithOnnx(source);
  }
  return classifyBySignature(source);
}

function resetRealtimeSmoothing() {
  state.realtimeSmoothing = {
    backend: null,
    scores: new Map(),
    history: [],
    lastStableId: null
  };
}

function moveCandidateToFront(candidates, candidateId) {
  const index = candidates.findIndex((candidate) => candidate.id === candidateId);
  if (index <= 0) return candidates;
  const reordered = [...candidates];
  const [candidate] = reordered.splice(index, 1);
  reordered.unshift(candidate);
  return reordered;
}

function stabilizeRealtimeResult(result) {
  if (!state.realtimeSmoothing || state.realtimeSmoothing.backend !== result.backend) {
    resetRealtimeSmoothing();
    state.realtimeSmoothing.backend = result.backend;
  }

  const smoothing = state.realtimeSmoothing;
  const nextScores = new Map();
  result.candidates.forEach((candidate) => {
    const previous = smoothing.scores.has(candidate.id) ? smoothing.scores.get(candidate.id) : candidate.confidence;
    nextScores.set(candidate.id, previous * (1 - REALTIME_EMA_ALPHA) + candidate.confidence * REALTIME_EMA_ALPHA);
  });
  smoothing.scores = nextScores;

  smoothing.history.push(result.best.id);
  if (smoothing.history.length > REALTIME_HISTORY_SIZE) {
    smoothing.history.shift();
  }

  const smoothedCandidates = result.candidates
    .map((candidate) => ({
      ...candidate,
      confidence: Math.max(0.01, Math.min(0.99, nextScores.get(candidate.id) ?? candidate.confidence))
    }))
    .sort((a, b) => b.confidence - a.confidence);

  const leadingId = smoothedCandidates[0].id;
  const leadingVotes = smoothing.history.filter((id) => id === leadingId).length;
  const rawGap = result.candidates.length > 1 ? result.candidates[0].confidence - result.candidates[1].confidence : 1;

  if (leadingVotes >= REALTIME_STABLE_VOTES || smoothedCandidates[0].confidence >= 0.78) {
    smoothing.lastStableId = leadingId;
  }

  let candidates = smoothedCandidates;
  if (smoothing.lastStableId && smoothing.lastStableId !== leadingId) {
    const stableCandidate = smoothedCandidates.find((candidate) => candidate.id === smoothing.lastStableId);
    if (stableCandidate) {
      const leadOverStable = smoothedCandidates[0].confidence - stableCandidate.confidence;
      if (leadingVotes < REALTIME_STABLE_VOTES || leadOverStable < 0.16 || rawGap < 0.12) {
        stableCandidate.confidence = Math.max(stableCandidate.confidence, smoothedCandidates[0].confidence - 0.02);
        candidates = moveCandidateToFront(smoothedCandidates, smoothing.lastStableId);
      }
    }
  }

  return {
    ...result,
    best: candidates[0],
    candidates,
    stabilized: true
  };
}

function drawPreview(source, result) {
  const canvas = els.previewCanvas;
  const { width, height } = sourceSize(source);
  const ratio = Math.min(canvas.width / width, canvas.height / height);
  const drawWidth = width * ratio;
  const drawHeight = height * ratio;
  const offsetX = (canvas.width - drawWidth) / 2;
  const offsetY = (canvas.height - drawHeight) / 2;

  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, offsetX, offsetY, drawWidth, drawHeight);

  if (result?.bounds) {
    ctx.save();
    ctx.strokeStyle = "#0071e3";
    ctx.lineWidth = 4;
    ctx.setLineDash([16, 8]);
    ctx.strokeRect(
      offsetX + result.bounds.x * ratio,
      offsetY + result.bounds.y * ratio,
      result.bounds.width * ratio,
      result.bounds.height * ratio
    );
    ctx.restore();
  }
}

function drawIdleCanvas() {
  const canvas = els.previewCanvas;
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  ctx.strokeRect(80, 70, canvas.width - 160, canvas.height - 140);

  ctx.fillStyle = "rgba(255,255,255,0.96)";
  ctx.font = "700 34px Microsoft YaHei, Segoe UI, Arial";
  ctx.textAlign = "center";
  ctx.fillText("天津大学危险标志识别", canvas.width / 2, canvas.height / 2 - 18);

  ctx.fillStyle = "rgba(255,255,255,0.68)";
  ctx.font = "700 18px Microsoft YaHei, Segoe UI, Arial";
  ctx.fillText("ONNX Runtime Web · EfficientNet-B0", canvas.width / 2, canvas.height / 2 + 22);
  ctx.restore();
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function updateResult(result) {
  const best = result.best;
  els.confidenceValue.textContent = formatPercent(best.confidence);
  els.resultThumb.src = best.asset;
  els.resultThumb.alt = best.name;
  els.resultName.textContent = best.name;
  els.riskBadge.textContent = best.risk;
  els.detailAction.textContent = best.action;
  setBackend(result.backend === "onnx" ? "ONNX 训练模型" : "样本相似度");

  els.scoreList.innerHTML = "";
  result.candidates.forEach((candidate, index) => {
    const row = document.createElement("div");
    row.className = `score-row${index === 0 ? " is-best" : ""}`;
    row.innerHTML = `
      <div class="score-head">
        <span>${candidate.name}</span>
        <span>${formatPercent(candidate.confidence)}</span>
      </div>
      <div class="score-track">
        <div class="score-fill" style="width:${Math.round(candidate.confidence * 100)}%"></div>
      </div>
    `;
    els.scoreList.appendChild(row);
  });
}

async function analyzeSource(source, message, options = {}) {
  if (!state.references.length && !state.onnxSession) return;
  const start = performance.now();
  try {
    const rawResult = await classify(source);
    const result = options.stabilize ? stabilizeRealtimeResult(rawResult) : rawResult;
    const elapsed = Math.max(1, Math.round(performance.now() - start));
    if (els.latencyValue) els.latencyValue.textContent = `${elapsed} ms`;
    drawPreview(source, result);
    updateResult(result);
    if (message) setMessage(message);
  } catch (error) {
    console.warn("ONNX inference failed, falling back to signature model.", error);
    if (!state.references.length) {
      setMessage("模型推理失败，请检查 ONNX 文件", true);
      return;
    }
    const rawFallback = classifyBySignature(source);
    const fallback = options.stabilize ? stabilizeRealtimeResult(rawFallback) : rawFallback;
    const elapsed = Math.max(1, Math.round(performance.now() - start));
    if (els.latencyValue) els.latencyValue.textContent = `${elapsed} ms`;
    drawPreview(source, fallback);
    updateResult(fallback);
    setStatus("标准样本模型就绪");
    state.inferenceBackend = "signature";
    if (message) setMessage(`${message}（已回退到离线相似度识别）`);
  }
}

function clearTimer() {
  if (state.activeTimer) {
    clearInterval(state.activeTimer);
    state.activeTimer = null;
  }
}

function stopCamera() {
  clearTimer();
  resetRealtimeSmoothing();
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  els.cameraSource.srcObject = null;
}

function stopVideoLoop() {
  clearTimer();
  resetRealtimeSmoothing();
  els.videoSource.pause();
  state.videoPlaying = false;
  els.pauseVideoButton.innerHTML = '<span class="icon" aria-hidden="true">▷</span>继续';
}

function startVideoLoop(video, label) {
  clearTimer();
  resetRealtimeSmoothing();
  state.videoPlaying = true;
  video.play();
  state.activeTimer = setInterval(() => {
    if (video.readyState >= 2 && !video.paused && !video.ended) {
      if (state.inferenceBusy) return;
      state.inferenceBusy = true;
      setQueue("推理中");
      analyzeSource(video, label, { stabilize: true }).finally(() => {
        state.inferenceBusy = false;
        setQueue("空闲");
      });
    }
  }, REALTIME_INFERENCE_INTERVAL_MS);
}

function setMode(mode) {
  state.mode = mode;
  clearTimer();
  resetRealtimeSmoothing();
  stopCamera();
  els.videoSource.pause();

  const modeNames = {
    image: "图片检测",
    video: "视频检测",
    camera: "摄像头"
  };
  els.modeLabel.textContent = modeNames[mode];
  els.detailMode.textContent = modeNames[mode];

  els.imageControls.classList.toggle("hidden", mode !== "image");
  els.videoControls.classList.toggle("hidden", mode !== "video");
  els.cameraControls.classList.toggle("hidden", mode !== "camera");
  els.modeButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.mode === mode));
  setInputSource(modeNames[mode]);
  setMessage(mode === "image" ? "已加载标准样本" : "等待输入源");
}

async function handleImageFile(file) {
  if (!file) return;
  resetRealtimeSmoothing();
  setInputSource(file.name);
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    await analyzeSource(img, `已检测：${file.name}`);
  } catch {
    setMessage("图片读取失败", true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function handleVideoFile(file) {
  if (!file) return;
  resetRealtimeSmoothing();
  setInputSource(file.name);
  const url = URL.createObjectURL(file);
  els.videoSource.src = url;
  els.videoSource.onloadeddata = () => {
    startVideoLoop(els.videoSource, `视频检测：${file.name}`);
  };
  els.videoSource.onended = () => {
    stopVideoLoop();
    setMessage("视频检测完成");
    URL.revokeObjectURL(url);
  };
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("当前浏览器不支持摄像头调用", true);
    return;
  }
  try {
    stopCamera();
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 24, max: 30 }
      },
      audio: false
    });
    const [videoTrack] = state.cameraStream.getVideoTracks();
    if (videoTrack?.applyConstraints) {
      await videoTrack
        .applyConstraints({
          advanced: [
            { focusMode: "continuous" },
            { exposureMode: "continuous" },
            { whiteBalanceMode: "continuous" }
          ]
        })
        .catch(() => {});
    }
    els.cameraSource.srcObject = state.cameraStream;
    await els.cameraSource.play();
    setInputSource("摄像头实时流");
    startVideoLoop(els.cameraSource, "摄像头检测中");
  } catch {
    setMessage("摄像头未开启或权限被浏览器拦截", true);
  }
}

function renderSamples() {
  els.sampleList.innerHTML = "";
  SIGN_LIBRARY.forEach((sign) => {
    const card = document.createElement("article");
    card.className = "sample-card";
    card.tabIndex = 0;
    card.dataset.signId = sign.id;
    card.innerHTML = `
      <img src="${sign.asset}" alt="${sign.name}" />
      <div>
        <p>${sign.name}</p>
        <span>${sign.risk}</span>
      </div>
    `;
    const analyzeSample = async () => {
      const reference = state.references.find((item) => item.id === sign.id);
      if (!reference) return;
      setInputSource(sign.name);
      await analyzeSource(reference.image, `样本复测：${sign.name}`);
    };
    card.addEventListener("click", analyzeSample);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        analyzeSample();
      }
    });
    els.sampleList.appendChild(card);
  });
}

async function initializeOnnxModel() {
  if (!window.ort) {
    throw new Error("onnxruntime-web is not loaded");
  }

  ort.env.wasm.wasmPaths = WASM_BASE_PATH;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.simd = false;

  const metadataResponse = await fetch(MODEL_METADATA_URL);
  if (!metadataResponse.ok) {
    throw new Error(`metadata fetch failed: ${metadataResponse.status}`);
  }
  const metadata = await metadataResponse.json();
  const externalDataName = metadata.externalData || `${metadata.model}.data`;
  const externalDataPath = metadata.externalDataPath || externalDataName;
  const modelUrl = new URL(`${MODEL_BASE_PATH}${metadata.model}`, window.location.href).href;
  const externalDataUrl = new URL(`${MODEL_BASE_PATH}${externalDataName}`, window.location.href).href;
  const sessionOptions = {
    executionProviders: ["wasm"],
    externalData: [
      {
        path: externalDataPath,
        data: externalDataUrl
      }
    ]
  };

  let session;
  try {
    session = await ort.InferenceSession.create(modelUrl, sessionOptions);
  } catch (urlError) {
    console.warn("ONNX URL session creation failed, retrying with binary buffers.", urlError);
    const [modelResponse, externalResponse] = await Promise.all([fetch(modelUrl), fetch(externalDataUrl)]);
    if (!modelResponse.ok) throw new Error(`model fetch failed: ${modelResponse.status}`);
    if (!externalResponse.ok) throw new Error(`external data fetch failed: ${externalResponse.status}`);
    const [modelBuffer, externalBuffer] = await Promise.all([
      modelResponse.arrayBuffer(),
      externalResponse.arrayBuffer()
    ]);
    session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: ["wasm"],
      externalData: [
        {
          path: externalDataPath,
          data: new Uint8Array(externalBuffer)
        }
      ]
    });
  }

  state.onnxMetadata = metadata;
  state.onnxSession = session;
  state.inferenceBackend = "onnx";
  if (els.modelArchValue) els.modelArchValue.textContent = metadata.arch || "ONNX";
  setBackend("ONNX 训练模型");
  setRuntime("ONNX Runtime Web");
  setStatus("ONNX模型就绪");
}

async function loadReferences() {
  setStatus("模型加载中");
  const loaded = [];
  for (const sign of SIGN_LIBRARY) {
    const img = await loadImage(sign.asset);
    const { signature } = buildSignature(img);
    loaded.push({ ...sign, image: img, signature });
  }
  state.references = loaded;
  setStatus("标准样本模型就绪");

  if (SKIP_ONNX) {
    setRuntime("Preview");
    await analyzeSource(loaded[0].image, "界面预览模式，当前跳过 ONNX 加载");
    return;
  }

  try {
    await initializeOnnxModel();
    await analyzeSource(loaded[0].image, "ONNX训练模型已导入网页端");
  } catch (error) {
    console.warn("ONNX model initialization failed.", error);
    setStatus("标准样本模型就绪");
    setRuntime("ONNX 未加载");
    setBackend("样本相似度");
    const detail = error?.message ? `：${error.message}` : "";
    await analyzeSource(
      loaded[0].image,
      `ONNX模型未加载${detail}；当前只使用离线相似度识别，准确率会明显低于训练模型`
    );
  }
}

function bindEvents() {
  els.loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const ok = els.username.value.trim() === "admin" && els.password.value === "123456";
    els.loginError.hidden = ok;
    if (ok) {
      els.loginScreen.classList.add("hidden");
      els.appScreen.classList.remove("hidden");
    }
  });

  els.logoutButton.addEventListener("click", () => {
    stopCamera();
    clearTimer();
    els.appScreen.classList.add("hidden");
    els.loginScreen.classList.remove("hidden");
  });

  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.imageInput.addEventListener("change", (event) => handleImageFile(event.target.files[0]));
  els.videoInput.addEventListener("change", (event) => handleVideoFile(event.target.files[0]));

  document.querySelectorAll(".file-drop").forEach((drop) => {
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.classList.add("is-dragging");
    });
    drop.addEventListener("dragleave", () => {
      drop.classList.remove("is-dragging");
    });
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("is-dragging");
      const file = event.dataTransfer.files[0];
      if (!file) return;
      if (file.type.startsWith("video/")) {
        setMode("video");
        handleVideoFile(file);
      } else {
        setMode("image");
        handleImageFile(file);
      }
    });
  });

  els.startCameraButton.addEventListener("click", startCamera);
  els.stopCameraButton.addEventListener("click", () => {
    stopCamera();
    setMessage("摄像头已停止");
  });
  els.pauseVideoButton.addEventListener("click", () => {
    if (!els.videoSource.src) return;
    if (state.videoPlaying) {
      stopVideoLoop();
    } else {
      startVideoLoop(els.videoSource, "视频检测中");
      els.pauseVideoButton.innerHTML = '<span class="icon" aria-hidden="true">Ⅱ</span>暂停';
    }
  });
}

renderSamples();
bindEvents();
drawIdleCanvas();
if (PREVIEW_MODE) {
  els.loginScreen.classList.add("hidden");
  els.appScreen.classList.remove("hidden");
}
loadReferences().catch(() => {
  setStatus("加载失败");
  setMessage("样本加载失败，请检查 assets/signs 目录", true);
});
