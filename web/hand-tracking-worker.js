import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_ROOT = `${self.location.origin}/mediapipe-wasm`;
const MODEL_URL = new URL("../hand_landmarker.task", import.meta.url).href;
const INFERENCE_WIDTH = 960;
const INFERENCE_HEIGHT = 540;

let landmarker = null;
let delegate = "CPU";
let ready = false;

function publishStatus(stage, progress, detail = "") {
  self.postMessage({ type: "status", stage, progress, detail });
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function fetchModel() {
  publishStatus("model", 25);
  const response = await fetch(MODEL_URL, { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`Hand model request failed with HTTP ${response.status}.`);
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    throw new Error("Hand model returned HTML instead of model data. Check Vercel Deployment Protection.");
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    publishStatus("model", 65);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const ratio = total ? received / total : 0;
    publishStatus("model", Math.round(25 + Math.min(1, ratio) * 40));
  }

  const model = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    model.set(chunk, offset);
    offset += chunk.length;
  }
  publishStatus("model", 65);
  return model;
}

async function initialize() {
  const startedAt = performance.now();
  try {
    publishStatus("runtime", 5);
    const [vision, modelAssetBuffer] = await Promise.all([
      // Module workers cannot expose globals from MediaPipe's classic loader.
      // The module loader explicitly publishes ModuleFactory on globalThis.
      FilesetResolver.forVisionTasks(WASM_ROOT, true),
      fetchModel(),
    ]);
    publishStatus("compile", 72);

    const commonOptions = {
      baseOptions: { modelAssetBuffer },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };

    if (typeof OffscreenCanvas !== "undefined") {
      try {
        landmarker = await HandLandmarker.createFromOptions(vision, {
          ...commonOptions,
          baseOptions: { ...commonOptions.baseOptions, delegate: "GPU" },
          canvas: new OffscreenCanvas(INFERENCE_WIDTH, INFERENCE_HEIGHT),
        });
        delegate = "GPU";
      } catch (gpuError) {
        publishStatus("fallback", 82, errorMessage(gpuError));
      }
    }

    if (!landmarker) {
      landmarker = await HandLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { ...commonOptions.baseOptions, delegate: "CPU" },
      });
      delegate = "CPU";
    }

    ready = true;
    self.postMessage({
      type: "ready",
      delegate,
      initializationMs: Math.round(performance.now() - startedAt),
      inferenceWidth: INFERENCE_WIDTH,
      inferenceHeight: INFERENCE_HEIGHT,
    });
  } catch (error) {
    self.postMessage({ type: "fatal", message: errorMessage(error) });
  }
}

function processFrame(message) {
  const { frame, timestampMs, sessionId } = message;
  if (!ready || !landmarker) {
    frame.close();
    self.postMessage({ type: "frame-error", sessionId, message: "Tracker is not ready." });
    return;
  }

  const startedAt = performance.now();
  try {
    const result = landmarker.detectForVideo(frame, timestampMs);
    self.postMessage({
      type: "result",
      sessionId,
      timestampMs,
      inferenceMs: performance.now() - startedAt,
      landmarks: result.landmarks,
      worldLandmarks: result.worldLandmarks,
    });
  } catch (error) {
    self.postMessage({ type: "frame-error", sessionId, message: errorMessage(error) });
  } finally {
    frame.close();
  }
}

self.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "frame") processFrame(message);
  if (message.type === "close") {
    landmarker?.close();
    self.close();
  }
});

initialize();
