export const GestureMode = Object.freeze({
  DRAW: "draw",
  MOVE: "move",
  ERASE: "erase",
  CLEAR: "clear",
  IDLE: "idle",
});

const FINGER_CHAINS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

function point(value) {
  if (!value) return null;
  const coordinates = [Number(value.x ?? value[0]), Number(value.y ?? value[1]), Number(value.z ?? value[2] ?? 0)];
  return coordinates.every(Number.isFinite) ? coordinates : null;
}

function vector(start, end) {
  return [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
}

export function distance(first, second) {
  const a = point(first);
  const b = point(second);
  if (!a || !b) return null;
  const delta = vector(a, b);
  return Math.hypot(...delta);
}

export function jointAngle(first, vertex, third) {
  const a = point(first);
  const b = point(vertex);
  const c = point(third);
  if (!a || !b || !c) return null;
  const firstVector = vector(b, a);
  const secondVector = vector(b, c);
  const firstLength = Math.hypot(...firstVector);
  const secondLength = Math.hypot(...secondVector);
  if (firstLength <= 1e-9 || secondLength <= 1e-9) return null;
  const cosine = firstVector.reduce((sum, value, index) => sum + value * secondVector[index], 0) / (firstLength * secondLength);
  return (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
}

function validLandmarks(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;
  const points = landmarks.slice(0, 21).map(point);
  return points.every(Boolean) ? points : null;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function palmSize(points) {
  const spans = [5, 9, 13, 17].map((index) => distance(points[0], points[index]));
  spans.push(distance(points[5], points[17]), distance(points[0], points[9]));
  const valid = spans.filter((span) => span !== null && span > 1e-9);
  return valid.length ? median(valid) : null;
}

export function fingerStates(normalizedLandmarks, worldLandmarks = null) {
  const normalized = validLandmarks(normalizedLandmarks);
  if (!normalized) return [];
  const world = validLandmarks(worldLandmarks);
  const worldPalmSize = world ? palmSize(world) : null;
  const points = world && worldPalmSize ? world : normalized;
  const handScale = world && worldPalmSize ? worldPalmSize : palmSize(normalized);
  if (!handScale || handScale <= 1e-9) return [];

  const thumbMcpAngle = jointAngle(points[1], points[2], points[3]);
  const thumbIpAngle = jointAngle(points[2], points[3], points[4]);
  const thumbLength = distance(points[1], points[4]);
  const thumbSpread = distance(points[4], points[5]);
  const thumbExtended =
    thumbMcpAngle !== null && thumbMcpAngle >= 135 &&
    thumbIpAngle !== null && thumbIpAngle >= 145 &&
    thumbLength !== null && thumbLength / handScale >= 0.55 &&
    thumbSpread !== null && thumbSpread / handScale >= 0.45;

  const states = [Number(thumbExtended)];
  for (const [mcp, pip, dip, tip] of FINGER_CHAINS) {
    const pipAngle = jointAngle(points[mcp], points[pip], points[dip]);
    const dipAngle = jointAngle(points[pip], points[dip], points[tip]);
    const reach = distance(points[mcp], points[tip]);
    states.push(Number(
      pipAngle !== null && pipAngle >= 135 &&
      dipAngle !== null && dipAngle >= 135 &&
      reach !== null && reach / handScale >= 0.62,
    ));
  }
  return states;
}

export function classifyGesture(fingers) {
  if (fingers.length !== 5) return GestureMode.IDLE;
  const [, index, middle, ring, pinky] = fingers;
  if (index && middle && ring && pinky) return GestureMode.CLEAR;
  if (index && middle && ring && !pinky) return GestureMode.ERASE;
  if (index && middle && !ring) return GestureMode.MOVE;
  if (index && !middle) return GestureMode.DRAW;
  return GestureMode.IDLE;
}

export class GestureStateMachine {
  constructor({ stabilizationMs = 50, releaseGraceMs = 100, clearHoldMs = 600, maxGapMs = 250 } = {}) {
    this.settings = { stabilizationMs, releaseGraceMs, clearHoldMs, maxGapMs };
    if (Object.values(this.settings).some((value) => value < 0)) throw new Error("Gesture timings must be non-negative");
    this.reset();
  }

  update(rawMode, timestampMs) {
    if (this.lastTimestampMs !== null) {
      if (timestampMs <= this.lastTimestampMs) return { mode: this.stableMode, clearTriggered: false };
      if (timestampMs - this.lastTimestampMs > this.settings.maxGapMs) this.reset();
    }
    this.lastTimestampMs = timestampMs;
    if (rawMode !== GestureMode.CLEAR) this.clearLatched = false;
    if (rawMode === GestureMode.CLEAR) return this.updateClear(timestampMs);

    if (this.stableMode === GestureMode.CLEAR) {
      this.stableMode = GestureMode.IDLE;
      this.candidateMode = null;
    }
    if (rawMode === this.stableMode) {
      this.candidateMode = null;
      return { mode: this.stableMode, clearTriggered: false };
    }
    if (rawMode !== this.candidateMode) {
      this.candidateMode = rawMode;
      this.candidateSinceMs = timestampMs;
      return { mode: this.stableMode, clearTriggered: false };
    }
    const delay = rawMode === GestureMode.IDLE ? this.settings.releaseGraceMs : this.settings.stabilizationMs;
    if (timestampMs - this.candidateSinceMs >= delay) {
      this.stableMode = rawMode;
      this.candidateMode = null;
    }
    return { mode: this.stableMode, clearTriggered: false };
  }

  updateClear(timestampMs) {
    if (this.clearLatched) {
      this.stableMode = GestureMode.CLEAR;
      return { mode: this.stableMode, clearTriggered: false };
    }
    if (this.candidateMode !== GestureMode.CLEAR) {
      this.candidateMode = GestureMode.CLEAR;
      this.candidateSinceMs = timestampMs;
      return { mode: this.stableMode, clearTriggered: false };
    }
    if (timestampMs - this.candidateSinceMs >= this.settings.clearHoldMs) {
      this.stableMode = GestureMode.CLEAR;
      this.candidateMode = null;
      this.clearLatched = true;
      return { mode: this.stableMode, clearTriggered: true };
    }
    return { mode: this.stableMode, clearTriggered: false };
  }

  reset() {
    this.stableMode = GestureMode.IDLE;
    this.candidateMode = null;
    this.candidateSinceMs = 0;
    this.clearLatched = false;
    this.lastTimestampMs = null;
  }
}

class LowPassFilter {
  constructor() { this.value = null; }
  apply(value, alpha) {
    this.value = this.value === null ? value : alpha * value + (1 - alpha) * this.value;
    return this.value;
  }
  reset() { this.value = null; }
}

function smoothingFactor(deltaSeconds, cutoffHz) {
  const timeConstant = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + timeConstant / deltaSeconds);
}

class OneEuroFilter {
  constructor({ minCutoff = 2, beta = 5, derivativeCutoff = 1, maxGapMs = 250 } = {}) {
    this.settings = { minCutoff, beta, derivativeCutoff, maxGapMs };
    this.valueFilter = new LowPassFilter();
    this.derivativeFilter = new LowPassFilter();
    this.reset();
  }
  filter(value, timestampMs) {
    if (this.lastTimestampMs === null || timestampMs - this.lastTimestampMs > this.settings.maxGapMs) {
      this.reset();
      this.lastRaw = value;
      this.lastTimestampMs = timestampMs;
      this.derivativeFilter.apply(0, 1);
      return this.valueFilter.apply(value, 1);
    }
    const elapsedMs = timestampMs - this.lastTimestampMs;
    if (elapsedMs <= 0) return this.valueFilter.value ?? value;
    const deltaSeconds = elapsedMs / 1000;
    const derivative = (value - this.lastRaw) / deltaSeconds;
    const derivativeAlpha = smoothingFactor(deltaSeconds, this.settings.derivativeCutoff);
    const speed = this.derivativeFilter.apply(derivative, derivativeAlpha);
    const cutoff = this.settings.minCutoff + this.settings.beta * Math.abs(speed);
    const filtered = this.valueFilter.apply(value, smoothingFactor(deltaSeconds, cutoff));
    this.lastRaw = value;
    this.lastTimestampMs = timestampMs;
    return filtered;
  }
  reset() {
    this.valueFilter.reset();
    this.derivativeFilter.reset();
    this.lastRaw = null;
    this.lastTimestampMs = null;
  }
}

export class PointOneEuroFilter {
  constructor(settings = {}) {
    this.x = new OneEuroFilter(settings);
    this.y = new OneEuroFilter(settings);
  }
  filter([x, y], timestampMs) { return [this.x.filter(x, timestampMs), this.y.filter(y, timestampMs)]; }
  reset() { this.x.reset(); this.y.reset(); }
}

export class StrokeController {
  constructor(maxJumpPixels = 160) {
    this.maxJumpPixels = maxJumpPixels;
    this.reset();
  }
  update(mode, currentPoint) {
    if (![GestureMode.DRAW, GestureMode.ERASE].includes(mode) || !currentPoint) {
      this.reset();
      return null;
    }
    if (this.mode !== mode || !this.previous) {
      this.mode = mode;
      this.previous = currentPoint;
      return [currentPoint, currentPoint];
    }
    const previous = this.previous;
    this.previous = currentPoint;
    return Math.hypot(previous[0] - currentPoint[0], previous[1] - currentPoint[1]) <= this.maxJumpPixels
      ? [previous, currentPoint]
      : null;
  }
  reset() { this.mode = null; this.previous = null; }
}

export class TrackerFrameGate {
  constructor(maxFps = 30) {
    if (!Number.isFinite(maxFps) || maxFps <= 0) throw new Error("maxFps must be positive");
    this.minimumIntervalMs = 1000 / maxFps;
    this.reset();
  }

  tryStart(timestampMs) {
    if (this.busy || timestampMs - this.lastStartedAt < this.minimumIntervalMs) return false;
    this.busy = true;
    this.lastStartedAt = timestampMs;
    return true;
  }

  complete() {
    this.busy = false;
  }

  reset() {
    this.busy = false;
    this.lastStartedAt = Number.NEGATIVE_INFINITY;
  }
}
