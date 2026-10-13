# Air Writer - Real-Time Hand Gesture Drawing

[![CI](https://github.com/alaamadii/air-writer/actions/workflows/ci.yml/badge.svg)](https://github.com/alaamadii/air-writer/actions/workflows/ci.yml)

Air Writer turns hand gestures into drawing actions on a virtual OpenCV canvas. It uses
MediaPipe Hand Landmarker, geometry-based finger states, adaptive fingertip smoothing, and
timestamp-based gesture stabilization. Camera frames stay local and are not saved by default.

## Gestures

| Finger state | Mode | Behavior |
| --- | --- | --- |
| Index | `DRAW` | Draw with the index fingertip |
| Index + middle | `MOVE` | Move without changing the canvas |
| Index + middle + ring | `ERASE` | Erase under the index fingertip |
| Open palm held for about 600 ms | `CLEAR` | Clear once, then require release before another clear |
| Anything else or no fresh hand | `IDLE` | No canvas change |

Keyboard shortcuts: `C` clears the canvas, `L` toggles hand landmarks, `D` toggles
diagnostics, and `Q` quits.

## Drawing Toolbar

The translucent toolbar at the top of the camera view provides:

- Six drawing colors: red, blue, green, yellow, purple, and white.
- Small, medium, and large brush sizes.
- A `CLEAR` button.
- A live mode indicator showing `DRAW`, `MOVE`, `ERASE`, `CLEAR`, or `IDLE`.

Select a control with a normal mouse click, or hold the `MOVE` gesture (index and middle fingers)
over it for about 450 ms. A yellow progress line shows the dwell selection. Drawing is blocked
inside the toolbar and stroke continuity resets at its boundary, so using the controls cannot
leave accidental marks. The selected color and brush size are outlined in white.

## Tracking Architecture

The tracker runs MediaPipe in `LIVE_STREAM` mode. The camera/render loop mirrors each frame
exactly once, makes a smaller RGB inference copy, and calls `detect_async()` with a strictly
increasing timestamp from `time.monotonic_ns()`. MediaPipe's video-mode graph manages its own
hand ROI and may drop input frames when inference is busy; this is expected and lets capture and
display continue without blocking.

The callback performs no OpenCV rendering. It publishes only the newest completed result and its
timestamp through a lock-protected, one-item mailbox, so there is no application frame backlog.
The main thread reads a safe snapshot. It updates smoothing and gesture state only once for each
new result timestamp. Results older than `MAX_RESULT_AGE_MS` (200 ms by default) are unavailable,
which resets fingertip, gesture, and stroke continuity state.

Capture and display default to 1280x720. Inference defaults to a same-aspect-ratio 960x540 copy.
Normalized MediaPipe landmarks are mapped directly back to the full display dimensions. Change
`INFERENCE_WIDTH` and `INFERENCE_HEIGHT` together; 960x540 is a useful manual comparison when a
hand is often far from the camera.

## Stability and Gesture Geometry

Finger extension uses MCP/PIP/DIP/TIP joint angles and palm-normalized distances rather than
screen-axis comparisons. MediaPipe 3D world landmarks are preferred; normalized landmarks are a
validated fallback. This makes the rules insensitive to handedness and substantially less
sensitive to in-plane hand rotation.

The index tip passes through a timestamp-aware One Euro filter. Its defaults favor steady slow
movement while raising the cutoff during fast movement to avoid obvious lag:

- `SMOOTHING_MIN_CUTOFF = 2.0`: raise for less smoothing and lower for steadier slow motion.
- `SMOOTHING_BETA = 5.0`: raise for quicker response to fast movement. This value is scaled for
  normalized image coordinates rather than pixel coordinates.
- `SMOOTHING_DERIVATIVE_CUTOFF = 1.0`: smooths the speed estimate.
- `SMOOTHING_MAX_GAP_MS = 250`: reinitialize after a long gap instead of bridging it.

The filter also resets after hand loss, stale tracking, or reacquisition. `DRAW`, `MOVE`, and
`ERASE` must remain consistent for `GESTURE_STABILIZATION_MS` (50 ms). A short
`GESTURE_RELEASE_GRACE_MS` (100 ms) absorbs an isolated missing detection. `CLEAR` requires a
continuous `CLEAR_HOLD_MS` (600 ms) open-palm hold, fires only once per hold, and must be released
or changed before it can fire again. Timers use only new MediaPipe result timestamps; display
frames and stale results cannot advance them.

Stroke continuity is explicit. Entering `DRAW` or `ERASE` starts at the current point, while
`MOVE`, `IDLE`, `CLEAR`, hand loss, stale tracking, and transitions between draw and erase reset
the previous point. A jump larger than `MAX_STROKE_JUMP_PIXELS` restarts the stroke instead of
drawing a long accidental line.

## Confidence Settings

MediaPipe's confidence controls represent different decisions and are configured separately:

- `MIN_HAND_DETECTION_CONFIDENCE = 0.5`: confidence required to start a hand detection.
- `MIN_HAND_PRESENCE_CONFIDENCE = 0.5`: confidence that the tracked region still contains a hand.
- `MIN_TRACKING_CONFIDENCE = 0.5`: confidence in landmark tracking between detections.

Lower values can retain marginal hands but admit more false positives. Higher values reject more
uncertain results and may cause more reacquisition. Tune them independently for the target camera,
lighting, distance, and background.

## Diagnostics and Rendering

Set `SHOW_DIAGNOSTICS = True` to show rolling display FPS, completed-result FPS, latest result
age, callback latency, raw gesture, and stabilized gesture. These measurements are runtime
observations, not hardware-independent performance claims. Hand landmarks are visible by default
and render above the canvas. Press `L` to hide or show them while the application is running.

## Installation and Run

Requirements are Python 3.10+, a webcam, and a desktop environment with OpenCV window support.

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Automated Verification

The deterministic suite does not open a webcam:

```bash
ruff check .
ruff format --check .
python -m compileall -q -x "(\\.venv|venv)" .
python -m unittest discover -s tests -v
```

Tests cover gesture mapping, timestamp stabilization and stale gaps, CLEAR hold/latching,
One Euro filtering and reset behavior, landmark geometry and rotated synthetic poses, invalid
inputs, and stroke transitions/jump rejection.

## Manual Webcam Verification

1. Start with the default 960x540 inference size and press `D` to show diagnostics.
2. Run `python main.py`; confirm the preview is mirrored naturally and the cursor stays directly
   under the displayed index fingertip (no second mirror or coordinate offset).
3. Draw slowly and quickly. Check that slow jitter is reduced without an obvious fast-motion lag.
4. Switch repeatedly among `DRAW`, `MOVE`, and `ERASE`. Confirm no connecting line appears at a
   transition, after removing/reintroducing the hand, or after moving the hand a large distance.
5. Briefly hide a finger or hand and confirm an isolated miss does not cause rapid mode flicker;
   keep the hand absent beyond 200 ms and confirm drawing continuity resets.
6. Rotate and tilt both left and right hands while testing every gesture.
7. Hold an open palm: the canvas should clear only after roughly 600 ms and only once until the
   palm changes or closes.
8. Observe display FPS, result FPS, result age, and callback latency under the target lighting and
   workload. Compare 960x540 with 640x360 only if the result FPS is too low.
9. Press `D` or `L` to turn diagnostics or landmark drawing off when desired.

Webcam accuracy, throughput, latency, and the best confidence/smoothing values remain dependent on
the actual camera, CPU/GPU delegate, lighting, background, hand distance, and user motion. Those
items require the manual hardware procedure above and are not asserted by the unit tests.

## Project Structure

```text
air-writer/
|-- main.py                 # Capture, timestamp consumption, canvas, and display
|-- hand_tracker.py         # LIVE_STREAM detector and latest-result mailbox
|-- hand_geometry.py        # Rotation-robust pure landmark geometry
|-- gesture_logic.py        # Gesture mapping and temporal state machine
|-- smoothing.py            # Pure timestamp-aware One Euro filter
|-- stroke.py               # Pure drawing continuity state
|-- toolbar.py              # Color/size controls, mouse and gesture selection
|-- config.py               # Runtime and tuning settings
|-- hand_landmarker.task    # MediaPipe model asset
`-- tests/                  # Deterministic unit tests
```
