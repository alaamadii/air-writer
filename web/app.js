import "./styles.css";
import {
  GestureMode,
  GestureStateMachine,
  PointOneEuroFilter,
  StrokeController,
  TrackerFrameGate,
  classifyGesture,
  fingerStates,
} from "./core.js";

const INFERENCE_WIDTH = 960;
const INFERENCE_HEIGHT = 540;
const TRACKER_TIMEOUT_MS = 45_000;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const MODE_LABELS = {
  [GestureMode.DRAW]: "رسم",
  [GestureMode.MOVE]: "تحريك",
  [GestureMode.ERASE]: "ممحاة",
  [GestureMode.CLEAR]: "مسح الكل",
  [GestureMode.IDLE]: "انتظار",
};

const ui = {
  video: document.querySelector("#cameraVideo"),
  drawingCanvas: document.querySelector("#drawingCanvas"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  cameraStage: document.querySelector("#cameraStage"),
  emptyState: document.querySelector("#emptyState"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  saveButton: document.querySelector("#saveButton"),
  landmarksButton: document.querySelector("#landmarksButton"),
  diagnosticsButton: document.querySelector("#diagnosticsButton"),
  fullscreenButton: document.querySelector("#fullscreenButton"),
  controlDock: document.querySelector("#controlDock"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  modeChip: document.querySelector("#modeChip"),
  modeText: document.querySelector("#modeText"),
  engineBadge: document.querySelector("#engineBadge"),
  diagnostics: document.querySelector("#diagnostics"),
  toast: document.querySelector("#toast"),
  supportNote: document.querySelector("#supportNote"),
  modelProgress: document.querySelector("#modelProgress"),
  modelProgressBar: document.querySelector("#modelProgressBar"),
  modelProgressText: document.querySelector("#modelProgressText"),
};

const drawingContext = ui.drawingCanvas.getContext("2d");
const overlayContext = ui.overlayCanvas.getContext("2d");
const gestures = new GestureStateMachine();
const smoother = new PointOneEuroFilter();
const strokes = new StrokeController(160);
const frameGate = new TrackerFrameGate(30);
const frameTimes = [];
const captureCanvas = document.createElement("canvas");
captureCanvas.width = INFERENCE_WIDTH;
captureCanvas.height = INFERENCE_HEIGHT;
const captureContext = captureCanvas.getContext("2d", { alpha: false, desynchronized: true });

let trackerWorker = null;
let trackerReadyPromise = null;
let trackerReady = false;
let stream = null;
let animationFrame = null;
let lastVideoTime = -1;
let selectedColor = "#ff4d5f";
let brushSize = 8;
let showLandmarks = true;
let showDiagnostics = false;
let running = false;
let delegate = "--";
let trackerInitializationMs = null;
let latestInferenceMs = null;
let trackerProgress = 0;
let trackerProgressLabel = "بدء التحميل…";
let trackingSessionId = 0;
let cameraStartPending = false;
let dwellTarget = null;
let dwellStartedAt = 0;
let dwellLatched = false;
let toastTimer = null;

function setStatus(text, state = "") {
  ui.statusText.textContent = text;
  ui.statusDot.className = `status-dot${state ? ` is-${state}` : ""}`;
}

function setMode(mode) {
  ui.modeChip.dataset.mode = mode;
  ui.modeText.textContent = MODE_LABELS[mode];
}

function showToast(message, isError = false) {
  window.clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.className = `toast is-visible${isError ? " is-error" : ""}`;
  toastTimer = window.setTimeout(() => { ui.toast.className = "toast"; }, 3200);
}

function setControlsEnabled(enabled) {
  for (const button of [ui.stopButton, ui.saveButton, ui.landmarksButton, ui.diagnosticsButton]) {
    button.disabled = !enabled;
  }
}

function setModelProgress(progress, label, visible = cameraStartPending) {
  const safeProgress = Math.max(0, Math.min(100, progress));
  trackerProgress = safeProgress;
  trackerProgressLabel = label;
  ui.modelProgress.hidden = !visible;
  ui.modelProgressBar.style.width = `${safeProgress}%`;
  ui.modelProgress.setAttribute("aria-valuenow", String(safeProgress));
  ui.modelProgressText.textContent = label;
}

function trackerStatusLabel(stage) {
  if (stage === "runtime") return "تحميل محرّك MediaPipe…";
  if (stage === "model") return "تحميل نموذج اليد…";
  if (stage === "compile") return "تجهيز التتبع على جهازك…";
  if (stage === "fallback") return "تشغيل وضع التوافق…";
  return "تجهيز تتبع اليد…";
}

function trackerErrorMessage(message) {
  if (message.includes("Deployment Protection") || message.includes("HTTP 401") || message.includes("HTTP 302")) {
    return "تعذّر تحميل نموذج اليد بسبب حماية Vercel. استخدم رابط Production العام أو عطّل الحماية عنه.";
  }
  if (message.includes("timed out")) {
    return "استغرق تحميل نموذج اليد وقتًا طويلًا. تحقق من الاتصال ثم أعد المحاولة.";
  }
  return "تعذّر تجهيز تتبع اليد. أعد تحميل الصفحة أو جرّب متصفحًا حديثًا.";
}

function destroyTrackerWorker() {
  trackerWorker?.terminate();
  trackerWorker = null;
  trackerReadyPromise = null;
  trackerReady = false;
  trackerProgress = 0;
  trackerProgressLabel = "بدء التحميل…";
  frameGate.reset();
}

function ensureTrackerReady() {
  if (trackerReady) return Promise.resolve();
  if (trackerReadyPromise) return trackerReadyPromise;
  if (typeof Worker === "undefined" || typeof createImageBitmap === "undefined") {
    return Promise.reject(new Error("This browser does not support the required worker APIs."));
  }

  ui.engineBadge.textContent = "MediaPipe · تجهيز";
  trackerWorker = new Worker(new URL("./hand-tracking-worker.js", import.meta.url), { type: "module" });
  trackerReadyPromise = new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      destroyTrackerWorker();
      reject(new Error("MediaPipe initialization timed out."));
    }, TRACKER_TIMEOUT_MS);

    trackerWorker.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "status") {
        ui.engineBadge.textContent = `MediaPipe · ${message.progress}%`;
        setModelProgress(message.progress, trackerStatusLabel(message.stage));
        return;
      }
      if (message.type === "ready") {
        window.clearTimeout(timeout);
        trackerReady = true;
        delegate = message.delegate;
        trackerInitializationMs = message.initializationMs;
        ui.engineBadge.textContent = `MediaPipe · ${delegate}`;
        setModelProgress(100, "تتبع اليد جاهز", cameraStartPending);
        resolve();
        return;
      }
      if (message.type === "fatal") {
        window.clearTimeout(timeout);
        const error = new Error(message.message);
        destroyTrackerWorker();
        reject(error);
        return;
      }
      if (message.type === "result") {
        frameGate.complete();
        if (!running || message.sessionId !== trackingSessionId) return;
        latestInferenceMs = message.inferenceMs;
        processResults(message, message.timestampMs);
        return;
      }
      if (message.type === "frame-error") {
        frameGate.complete();
        if (!running || message.sessionId !== trackingSessionId) return;
        console.error("Hand tracking frame failed", message.message);
        setStatus("خطأ في التتبع", "error");
      }
    });

    trackerWorker.addEventListener("error", (event) => {
      window.clearTimeout(timeout);
      const error = new Error(event.message || "Hand tracking worker failed.");
      destroyTrackerWorker();
      reject(error);
    }, { once: true });
  });
  return trackerReadyPromise;
}

function cameraErrorMessage(error) {
  if (!window.isSecureContext) return "الكاميرا تحتاج HTTPS. افتح الموقع من رابط Vercel الآمن.";
  if (error?.name === "NotAllowedError") return "تم رفض إذن الكاميرا. اسمح للكاميرا من إعدادات المتصفح ثم حاول مجددًا.";
  if (error?.name === "NotFoundError") return "لم يتم العثور على كاميرا متاحة على هذا الجهاز.";
  if (error?.name === "NotReadableError") return "الكاميرا مستخدمة من برنامج آخر. أغلقه ثم حاول مجددًا.";
  return "تعذر تشغيل الكاميرا. تحقق من الإذن واتصال الكاميرا ثم حاول مجددًا.";
}

function waitForVideoMetadata() {
  if (ui.video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
  return new Promise((resolve, reject) => {
    ui.video.addEventListener("loadedmetadata", resolve, { once: true });
    ui.video.addEventListener("error", reject, { once: true });
  });
}

function resizeCanvases() {
  const width = ui.video.videoWidth || 1280;
  const height = ui.video.videoHeight || 720;
  for (const canvas of [ui.drawingCanvas, ui.overlayCanvas]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }
  drawingContext.lineCap = "round";
  drawingContext.lineJoin = "round";
}

async function startCamera() {
  if (running) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("هذا المتصفح لا يدعم الوصول إلى الكاميرا.", true);
    return;
  }
  cameraStartPending = true;
  ui.startButton.disabled = true;
  ui.startButton.lastChild.textContent = " جارٍ التجهيز…";
  setStatus(trackerReady ? "فتح الكاميرا…" : "تجهيز تتبع اليد…", "loading");
  setModelProgress(
    trackerReady ? 100 : Math.max(2, trackerProgress),
    trackerReady ? "تتبع اليد جاهز" : trackerProgressLabel,
    !trackerReady,
  );

  try {
    // Attach the rejection handler immediately while camera permission is pending.
    const trackerOutcome = ensureTrackerReady().then(
      () => ({ error: null }),
      (error) => ({ error }),
    );
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    ui.video.srcObject = stream;
    await waitForVideoMetadata();
    await ui.video.play();
    const { error: trackerError } = await trackerOutcome;
    if (trackerError) throw trackerError;
    resizeCanvases();
    resetTracking();
    running = true;
    trackingSessionId += 1;
    frameGate.reset();
    lastVideoTime = -1;
    ui.emptyState.classList.add("is-hidden");
    ui.controlDock.classList.add("is-visible");
    setControlsEnabled(true);
    setStatus("الكاميرا تعمل", "live");
    animationFrame = requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error(error);
    stopCamera(false);
    const isCameraError = ["NotAllowedError", "NotFoundError", "NotReadableError", "OverconstrainedError"].includes(error?.name);
    const message = isCameraError ? cameraErrorMessage(error) : trackerErrorMessage(error?.message ?? String(error));
    setStatus("تعذر التشغيل", "error");
    ui.supportNote.textContent = message;
    showToast(message, true);
  } finally {
    cameraStartPending = false;
    window.setTimeout(() => { ui.modelProgress.hidden = true; }, 350);
    ui.startButton.disabled = false;
    ui.startButton.lastChild.textContent = " تشغيل الكاميرا";
  }
}

function stopCamera(updateStatus = true) {
  running = false;
  trackingSessionId += 1;
  frameGate.reset();
  if (animationFrame !== null) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  ui.video.srcObject = null;
  overlayContext.clearRect(0, 0, ui.overlayCanvas.width, ui.overlayCanvas.height);
  ui.emptyState.classList.remove("is-hidden");
  ui.controlDock.classList.remove("is-visible");
  setControlsEnabled(false);
  resetTracking();
  if (updateStatus) setStatus("الكاميرا متوقفة");
}

function resetTracking() {
  resetHandState();
  frameTimes.length = 0;
  latestInferenceMs = null;
}

function resetHandState() {
  smoother.reset();
  gestures.reset();
  strokes.reset();
  resetDwell();
  setMode(GestureMode.IDLE);
}

function canvasContentRect() {
  const rect = ui.overlayCanvas.getBoundingClientRect();
  const scale = Math.min(rect.width / ui.overlayCanvas.width, rect.height / ui.overlayCanvas.height);
  const width = ui.overlayCanvas.width * scale;
  const height = ui.overlayCanvas.height * scale;
  return { left: rect.left + (rect.width - width) / 2, top: rect.top + (rect.height - height) / 2, width, height };
}

function canvasPointToClient([x, y]) {
  const content = canvasContentRect();
  return [
    content.left + (x / ui.overlayCanvas.width) * content.width,
    content.top + (y / ui.overlayCanvas.height) * content.height,
  ];
}

function toolbarTargetAt(point) {
  const [clientX, clientY] = canvasPointToClient(point);
  for (const element of document.elementsFromPoint(clientX, clientY)) {
    const target = element.closest?.("[data-air-action]");
    if (target && ui.controlDock.contains(target)) return target;
  }
  return null;
}

function resetDwell() {
  if (dwellTarget) dwellTarget.style.removeProperty("--dwell-progress");
  dwellTarget = null;
  dwellStartedAt = 0;
  dwellLatched = false;
}

function updateDwell(point, selecting, timestampMs) {
  const target = selecting ? toolbarTargetAt(point) : null;
  if (target !== dwellTarget) {
    resetDwell();
    dwellTarget = target;
    dwellStartedAt = timestampMs;
  }
  if (!target || dwellLatched) return Boolean(target);
  const progress = Math.min(1, (timestampMs - dwellStartedAt) / 450);
  target.style.setProperty("--dwell-progress", progress);
  if (progress >= 1) {
    activateTool(target);
    dwellLatched = true;
    target.style.removeProperty("--dwell-progress");
  }
  return true;
}

function activateTool(target) {
  const action = target.dataset.airAction;
  if (action === "color") {
    selectedColor = target.dataset.value;
    document.querySelectorAll('[data-air-action="color"]').forEach((button) => button.classList.toggle("is-selected", button === target));
  } else if (action === "size") {
    brushSize = Number(target.dataset.value);
    document.querySelectorAll('[data-air-action="size"]').forEach((button) => button.classList.toggle("is-selected", button === target));
  } else if (action === "clear") {
    clearDrawing();
    showToast("تم مسح اللوحة");
  }
  strokes.reset();
}

function clearDrawing() {
  drawingContext.clearRect(0, 0, ui.drawingCanvas.width, ui.drawingCanvas.height);
  strokes.reset();
}

function drawSegment(mode, segment) {
  if (!segment) return;
  const [[startX, startY], [endX, endY]] = segment;
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.moveTo(startX, startY);
  drawingContext.lineTo(endX, endY);
  if (mode === GestureMode.ERASE) {
    drawingContext.globalCompositeOperation = "destination-out";
    drawingContext.lineWidth = 40;
  } else {
    drawingContext.globalCompositeOperation = "source-over";
    drawingContext.strokeStyle = selectedColor;
    drawingContext.lineWidth = brushSize;
  }
  drawingContext.stroke();
  drawingContext.restore();
}

function mirroredPoint(landmark) {
  return [(1 - landmark.x) * ui.overlayCanvas.width, landmark.y * ui.overlayCanvas.height];
}

function drawHand(landmarks) {
  if (!showLandmarks) return;
  overlayContext.save();
  overlayContext.lineCap = "round";
  overlayContext.lineJoin = "round";
  overlayContext.strokeStyle = "rgba(55, 217, 155, .92)";
  overlayContext.lineWidth = Math.max(2.5, ui.overlayCanvas.width / 430);
  for (const [start, end] of HAND_CONNECTIONS) {
    const [startX, startY] = mirroredPoint(landmarks[start]);
    const [endX, endY] = mirroredPoint(landmarks[end]);
    overlayContext.beginPath();
    overlayContext.moveTo(startX, startY);
    overlayContext.lineTo(endX, endY);
    overlayContext.stroke();
  }
  const radius = Math.max(4, ui.overlayCanvas.width / 230);
  for (const landmark of landmarks) {
    const [x, y] = mirroredPoint(landmark);
    overlayContext.beginPath();
    overlayContext.arc(x, y, radius, 0, Math.PI * 2);
    overlayContext.fillStyle = "#4d8dff";
    overlayContext.fill();
    overlayContext.strokeStyle = "#ffffff";
    overlayContext.lineWidth = 2;
    overlayContext.stroke();
  }
  overlayContext.restore();
}

function drawCursor(point, mode) {
  const [x, y] = point;
  overlayContext.save();
  overlayContext.beginPath();
  if (mode === GestureMode.ERASE) {
    overlayContext.arc(x, y, 20, 0, Math.PI * 2);
    overlayContext.strokeStyle = "#ffffff";
    overlayContext.lineWidth = 2;
    overlayContext.stroke();
  } else {
    const radius = mode === GestureMode.DRAW ? Math.max(6, brushSize / 2 + 3) : 9;
    overlayContext.arc(x, y, radius, 0, Math.PI * 2);
    overlayContext.fillStyle = mode === GestureMode.DRAW ? selectedColor : "#4d8dff";
    overlayContext.fill();
    overlayContext.strokeStyle = "#ffffff";
    overlayContext.lineWidth = 2;
    overlayContext.stroke();
  }
  overlayContext.restore();
}

function updateFrameRate(timestampMs) {
  frameTimes.push(timestampMs);
  while (frameTimes.length && frameTimes[0] < timestampMs - 1000) frameTimes.shift();
  return frameTimes.length > 1 ? ((frameTimes.length - 1) * 1000) / (frameTimes.at(-1) - frameTimes[0]) : 0;
}

function renderDiagnostics(fps, rawMode, stableMode) {
  if (!showDiagnostics) return;
  const inference = latestInferenceMs === null ? "--" : `${latestInferenceMs.toFixed(1)} ms`;
  const initialization = trackerInitializationMs === null ? "--" : `${trackerInitializationMs} ms`;
  ui.diagnostics.textContent = `Inference: ${fps.toFixed(1)} FPS (${inference})\nWorker: active · ${delegate}\nGesture: ${rawMode} → ${stableMode}\nInference size: ${INFERENCE_WIDTH}×${INFERENCE_HEIGHT}\nCamera: ${ui.video.videoWidth}×${ui.video.videoHeight}\nCold start: ${initialization}`;
}

function processResults(result, timestampMs) {
  overlayContext.clearRect(0, 0, ui.overlayCanvas.width, ui.overlayCanvas.height);
  const fps = updateFrameRate(timestampMs);
  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    resetHandState();
    renderDiagnostics(fps, GestureMode.IDLE, GestureMode.IDLE);
    return;
  }

  drawHand(landmarks);
  const states = fingerStates(landmarks, result.worldLandmarks?.[0]);
  const rawMode = classifyGesture(states);
  const gestureUpdate = gestures.update(rawMode, timestampMs);
  const stableMode = gestureUpdate.mode;
  setMode(stableMode);

  const indexTip = landmarks[8];
  const normalizedPoint = smoother.filter([1 - indexTip.x, indexTip.y], timestampMs);
  const canvasPoint = [normalizedPoint[0] * ui.drawingCanvas.width, normalizedPoint[1] * ui.drawingCanvas.height];
  const overToolbar = updateDwell(canvasPoint, stableMode === GestureMode.MOVE, timestampMs);

  if (gestureUpdate.clearTriggered) {
    clearDrawing();
    showToast("تم مسح اللوحة بإشارة الكف");
  } else if (rawMode === GestureMode.CLEAR || overToolbar) {
    strokes.reset();
  } else {
    drawSegment(stableMode, strokes.update(stableMode, canvasPoint));
  }
  drawCursor(canvasPoint, stableMode);

  renderDiagnostics(fps, rawMode, stableMode);
}

async function submitInferenceFrame(timestampMs, sessionId) {
  try {
    captureContext.drawImage(ui.video, 0, 0, INFERENCE_WIDTH, INFERENCE_HEIGHT);
    const frame = await createImageBitmap(captureCanvas);
    if (!running || sessionId !== trackingSessionId || !trackerWorker) {
      frame.close();
      frameGate.complete();
      return;
    }
    trackerWorker.postMessage({ type: "frame", frame, timestampMs, sessionId }, [frame]);
  } catch (error) {
    frameGate.complete();
    console.error("Hand tracking frame failed", error);
    setStatus("خطأ في التتبع", "error");
  }
}

function renderLoop(timestampMs) {
  if (!running) return;
  if (
    trackerReady &&
    ui.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    ui.video.currentTime !== lastVideoTime &&
    frameGate.tryStart(timestampMs)
  ) {
    lastVideoTime = ui.video.currentTime;
    void submitInferenceFrame(timestampMs, trackingSessionId);
  }
  animationFrame = requestAnimationFrame(renderLoop);
}

function saveSnapshot() {
  if (!running) return;
  const output = document.createElement("canvas");
  output.width = ui.video.videoWidth;
  output.height = ui.video.videoHeight;
  const context = output.getContext("2d");
  context.translate(output.width, 0);
  context.scale(-1, 1);
  context.drawImage(ui.video, 0, 0, output.width, output.height);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.drawImage(ui.drawingCanvas, 0, 0);
  const link = document.createElement("a");
  link.download = `air-writer-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.png`;
  link.href = output.toDataURL("image/png");
  link.click();
  showToast("تم حفظ الصورة على جهازك");
}

ui.startButton.addEventListener("click", startCamera);
ui.stopButton.addEventListener("click", () => stopCamera());
ui.saveButton.addEventListener("click", saveSnapshot);
ui.landmarksButton.addEventListener("click", () => {
  showLandmarks = !showLandmarks;
  ui.landmarksButton.setAttribute("aria-pressed", showLandmarks);
});
ui.diagnosticsButton.addEventListener("click", () => {
  showDiagnostics = !showDiagnostics;
  ui.diagnostics.hidden = !showDiagnostics;
  ui.diagnosticsButton.setAttribute("aria-pressed", showDiagnostics);
});
ui.fullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await ui.cameraStage.requestFullscreen();
  } catch { showToast("ملء الشاشة غير متاح في هذا المتصفح", true); }
});
ui.controlDock.addEventListener("click", (event) => {
  const target = event.target.closest("[data-air-action]");
  if (target) activateTool(target);
});
window.addEventListener("beforeunload", () => {
  stopCamera(false);
  trackerWorker?.postMessage({ type: "close" });
  trackerWorker?.terminate();
});

if (!window.isSecureContext && location.hostname !== "localhost") {
  ui.supportNote.textContent = "الكاميرا تحتاج اتصال HTTPS آمن";
}

const preloadTracker = () => {
  ensureTrackerReady().catch((error) => {
    console.warn("MediaPipe preload failed; it will be retried when the camera starts.", error);
    ui.engineBadge.textContent = "MediaPipe · إعادة المحاولة";
  });
};

if ("requestIdleCallback" in window) window.requestIdleCallback(preloadTracker, { timeout: 750 });
else window.setTimeout(preloadTracker, 100);
