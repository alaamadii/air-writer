from __future__ import annotations

from collections import deque

import cv2
import numpy as np

import config
from gesture_logic import GestureMode, GestureStateMachine, classify_gesture
from hand_tracker import HandTracker, TrackerDiagnostics, monotonic_ms
from smoothing import PointOneEuroFilter
from stroke import StrokeController


class RollingFrameRate:
    def __init__(self, window_ms: int) -> None:
        self.window_ms = window_ms
        self._timestamps: deque[int] = deque()

    def record(self, timestamp_ms: int) -> float:
        self._timestamps.append(timestamp_ms)
        cutoff = timestamp_ms - self.window_ms
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) * 1000.0 / elapsed if elapsed > 0 else 0.0


def _pixel_point(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = min(width - 1, max(0, round(point[0] * width)))
    y = min(height - 1, max(0, round(point[1] * height)))
    return x, y


def _draw_cursor(frame: np.ndarray, point: tuple[int, int], mode: GestureMode) -> None:
    if mode == GestureMode.DRAW:
        cv2.circle(frame, point, 10, config.DEFAULT_BRUSH_COLOR, cv2.FILLED)
    elif mode == GestureMode.MOVE:
        cv2.circle(frame, point, 10, (255, 0, 0), cv2.FILLED)
    elif mode == GestureMode.ERASE:
        cv2.circle(frame, point, 20, (255, 255, 255), 2)


def _apply_canvas_segment(
    canvas: np.ndarray,
    mode: GestureMode,
    segment: tuple[tuple[int, int], tuple[int, int]] | None,
) -> None:
    """Apply one draw/erase segment to the persistent canvas."""
    if segment is None:
        return
    start, end = segment
    if mode == GestureMode.DRAW:
        cv2.line(
            canvas,
            start,
            end,
            config.DEFAULT_BRUSH_COLOR,
            config.DEFAULT_BRUSH_THICKNESS,
        )
    elif mode == GestureMode.ERASE:
        cv2.line(canvas, start, end, (0, 0, 0), config.ERASER_THICKNESS)


def _draw_diagnostics(
    frame: np.ndarray,
    display_fps: float,
    diagnostics: TrackerDiagnostics,
    raw_mode: GestureMode,
    stable_mode: GestureMode,
) -> None:
    age = "--" if diagnostics.result_age_ms is None else str(diagnostics.result_age_ms)
    latency = (
        "--" if diagnostics.callback_latency_ms is None else str(diagnostics.callback_latency_ms)
    )
    lines = (
        f"Display FPS: {display_fps:.1f}",
        f"Result FPS: {diagnostics.result_fps:.1f}",
        f"Result age: {age} ms",
        f"Callback latency: {latency} ms",
        f"Gesture: {raw_mode.value} -> {stable_mode.value}",
    )
    for index, line in enumerate(lines):
        y = 25 + index * 23
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Check camera permissions or update CAMERA_INDEX in config.py."
        )

    tracker: HandTracker | None = None
    canvas: np.ndarray | None = None
    smoother = PointOneEuroFilter(
        min_cutoff=config.SMOOTHING_MIN_CUTOFF,
        beta=config.SMOOTHING_BETA,
        derivative_cutoff=config.SMOOTHING_DERIVATIVE_CUTOFF,
        max_gap_ms=config.SMOOTHING_MAX_GAP_MS,
    )
    gestures = GestureStateMachine(
        stabilization_ms=config.GESTURE_STABILIZATION_MS,
        release_grace_ms=config.GESTURE_RELEASE_GRACE_MS,
        clear_hold_ms=config.CLEAR_HOLD_MS,
        max_gap_ms=config.GESTURE_MAX_GAP_MS,
    )
    strokes = StrokeController(config.MAX_STROKE_JUMP_PIXELS)
    display_rate = RollingFrameRate(config.METRICS_WINDOW_MS)
    last_result_timestamp_ms: int | None = None
    hand_visible = False
    result_was_fresh = False
    current_point: tuple[int, int] | None = None
    raw_mode = GestureMode.IDLE
    stable_mode = GestureMode.IDLE
    show_landmarks = config.DRAW_HAND_LANDMARKS
    show_diagnostics = config.SHOW_DIAGNOSTICS

    print("Air Writing started")
    print("- Index finger: draw")
    print("- Index + middle: move")
    print("- Index + middle + ring: erase")
    print("- Hold open palm: clear")
    print("- L: show/hide hand landmarks")
    print("- D: show/hide diagnostics")
    print("- C: clear canvas")
    print("- Q: quit")

    try:
        tracker = HandTracker()
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera. Exiting...")
                break

            # Flip once before both inference and display so normalized landmarks
            # map directly onto this frame without another mirror transform.
            frame = cv2.flip(frame, 1)
            height, width, channels = frame.shape
            if canvas is None or canvas.shape != frame.shape:
                canvas = np.zeros((height, width, channels), dtype=np.uint8)
                strokes.reset()

            tracker.process_frame(frame)
            now_ms = monotonic_ms()
            snapshot = tracker.get_latest_result(now_ms=now_ms)

            if snapshot is None:
                if result_was_fresh:
                    smoother.reset()
                    gestures.reset()
                    strokes.reset()
                    hand_visible = False
                    current_point = None
                    raw_mode = GestureMode.IDLE
                    stable_mode = GestureMode.IDLE
                result_was_fresh = False
            else:
                result_was_fresh = True
                if snapshot.timestamp_ms != last_result_timestamp_ms:
                    last_result_timestamp_ms = snapshot.timestamp_ms
                    normalized_tip = tracker.get_index_finger_tip(snapshot)
                    if normalized_tip is None:
                        smoother.reset()
                        strokes.reset()
                        hand_visible = False
                        current_point = None
                        raw_mode = GestureMode.IDLE
                        stable_mode = gestures.update(raw_mode, snapshot.timestamp_ms).mode
                    else:
                        if not hand_visible:
                            smoother.reset()
                        hand_visible = True
                        smoothed_tip = smoother.filter(normalized_tip, snapshot.timestamp_ms)
                        current_point = _pixel_point(smoothed_tip, width, height)
                        raw_mode = classify_gesture(tracker.get_fingers_up(snapshot))
                        gesture_update = gestures.update(raw_mode, snapshot.timestamp_ms)
                        stable_mode = gesture_update.mode

                        if gesture_update.clear_triggered:
                            canvas.fill(0)
                            strokes.reset()
                        elif raw_mode == GestureMode.CLEAR:
                            # Never continue an old stroke while an open-palm hold
                            # is waiting to reach the clear threshold.
                            strokes.reset()
                        else:
                            segment = strokes.update(stable_mode, current_point)
                            _apply_canvas_segment(canvas, stable_mode, segment)

            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
            final_frame = cv2.add(frame_bg, canvas)

            # Draw interaction overlays last so the canvas cannot cover them.
            tracker.draw_landmarks(final_frame, snapshot, enabled=show_landmarks)
            if current_point is not None and result_was_fresh:
                _draw_cursor(final_frame, current_point, stable_mode)

            display_fps = display_rate.record(now_ms)
            if show_diagnostics:
                _draw_diagnostics(
                    final_frame,
                    display_fps,
                    tracker.get_diagnostics(now_ms),
                    raw_mode,
                    stable_mode,
                )

            cv2.imshow("Air Writing", final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                canvas.fill(0)
                strokes.reset()
            if key == ord("l"):
                show_landmarks = not show_landmarks
            if key == ord("d"):
                show_diagnostics = not show_diagnostics
    finally:
        try:
            if tracker is not None:
                tracker.close()
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
