import assert from "node:assert/strict";
import test from "node:test";
import {
  GestureMode,
  GestureStateMachine,
  PointOneEuroFilter,
  StrokeController,
  classifyGesture,
  fingerStates,
  jointAngle,
} from "./core.js";

function makeHand(extended) {
  const points = Array.from({ length: 21 }, () => [0, 0, 0]);
  points[1] = [-0.2, -0.05, 0];
  points[2] = [-0.4, -0.1, 0];
  points[3] = [-0.62, -0.13, 0];
  points[4] = [-0.85, -0.15, 0];
  if (!extended.has("thumb")) {
    points.splice(1, 4, [-0.2, -0.05, 0], [-0.32, -0.15, 0], [-0.15, -0.22, 0], [0.02, -0.2, 0]);
  }
  for (const [name, start, x] of [["index", 5, -0.3], ["middle", 9, -0.1], ["ring", 13, 0.1], ["pinky", 17, 0.28]]) {
    points[start] = [x, -0.38, 0];
    if (extended.has(name)) {
      points[start + 1] = [x, -0.68, 0];
      points[start + 2] = [x, -0.94, 0];
      points[start + 3] = [x, -1.2, 0];
    } else {
      points[start + 1] = [x, -0.64, 0];
      points[start + 2] = [x + 0.18, -0.48, 0];
      points[start + 3] = [x + 0.2, -0.28, 0];
    }
  }
  return points;
}

test("finger geometry is rotation-independent and maps gestures", () => {
  const source = makeHand(new Set(["index", "middle", "ring"]));
  assert.deepEqual(fingerStates(source), [0, 1, 1, 1, 0]);
  assert.equal(classifyGesture(fingerStates(source)), GestureMode.ERASE);
  const angle = Math.PI / 2;
  const rotated = source.map(([x, y, z]) => [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle), z]);
  assert.deepEqual(fingerStates(rotated), [0, 1, 1, 1, 0]);
  assert.equal(Math.round(jointAngle([1, 0, 0], [0, 0, 0], [-1, 0, 0])), 180);
});

test("all primary finger combinations map to the expected mode", () => {
  assert.equal(classifyGesture([0, 1, 0, 0, 0]), GestureMode.DRAW);
  assert.equal(classifyGesture([0, 1, 1, 0, 0]), GestureMode.MOVE);
  assert.equal(classifyGesture([0, 1, 1, 1, 0]), GestureMode.ERASE);
  assert.equal(classifyGesture([0, 1, 1, 1, 1]), GestureMode.CLEAR);
  assert.equal(classifyGesture([]), GestureMode.IDLE);
});

test("gesture state stabilizes actions and fires clear once per hold", () => {
  const machine = new GestureStateMachine({ stabilizationMs: 80, clearHoldMs: 600 });
  assert.equal(machine.update(GestureMode.DRAW, 0).mode, GestureMode.IDLE);
  assert.equal(machine.update(GestureMode.DRAW, 80).mode, GestureMode.DRAW);
  machine.reset();
  machine.update(GestureMode.CLEAR, 100);
  machine.update(GestureMode.CLEAR, 300);
  machine.update(GestureMode.CLEAR, 500);
  assert.equal(machine.update(GestureMode.CLEAR, 699).clearTriggered, false);
  assert.equal(machine.update(GestureMode.CLEAR, 700).clearTriggered, true);
  assert.equal(machine.update(GestureMode.CLEAR, 800).clearTriggered, false);
});

test("stroke continuity resets on mode transitions and large jumps", () => {
  const strokes = new StrokeController(20);
  assert.deepEqual(strokes.update(GestureMode.DRAW, [0, 0]), [[0, 0], [0, 0]]);
  assert.deepEqual(strokes.update(GestureMode.DRAW, [10, 10]), [[0, 0], [10, 10]]);
  assert.equal(strokes.update(GestureMode.MOVE, [12, 12]), null);
  assert.deepEqual(strokes.update(GestureMode.DRAW, [30, 30]), [[30, 30], [30, 30]]);
  assert.equal(strokes.update(GestureMode.DRAW, [100, 100]), null);
});

test("One Euro point smoothing follows movement and resets after a stale gap", () => {
  const filter = new PointOneEuroFilter({ minCutoff: 1, beta: 0, maxGapMs: 100 });
  assert.deepEqual(filter.filter([0, 0], 0), [0, 0]);
  const smoothed = filter.filter([1, 1], 16);
  assert.ok(smoothed[0] > 0 && smoothed[0] < 1);
  assert.deepEqual(filter.filter([0.25, 0.5], 200), [0.25, 0.5]);
});
