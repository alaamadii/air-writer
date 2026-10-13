"""Asynchronous MediaPipe hand tracking with a latest-result mailbox."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from hand_geometry import finger_states


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class ResultSnapshot:
    """Immutable metadata plus one completed MediaPipe result reference."""

    result: Any
    timestamp_ms: int
    completed_at_ms: int
    callback_latency_ms: int


@dataclass(frozen=True)
class TrackerDiagnostics:
    result_fps: float
    result_age_ms: int | None
    callback_latency_ms: int | None


class HandTracker:
    """Submit frames without blocking rendering and expose only the latest result."""

    INDEX_FINGER_TIP = 8

    def __init__(self) -> None:
        model_path = Path(__file__).resolve().with_name("hand_landmarker.task")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Make sure hand_landmarker.task is included with the project."
            )

        self._state_lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._closed = False
        self._latest: ResultSnapshot | None = None
        self._last_submitted_timestamp_ms: int | None = None
        self._completion_times_ms: deque[int] = deque()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            result_callback=self._result_callback,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def process_frame(self, frame: Any) -> int | None:
        """Resize and submit one already-mirrored BGR frame for async inference.

        ``detect_async`` may intentionally drop a frame when its internal task is
        busy. No application-side frame queue is retained.
        """
        inference_frame = self._resize_for_inference(frame)
        image_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        with self._detector_lock:
            with self._state_lock:
                if self._closed:
                    return None
                timestamp_ms = monotonic_ms()
                if self._last_submitted_timestamp_ms is not None:
                    timestamp_ms = max(timestamp_ms, self._last_submitted_timestamp_ms + 1)
                self._last_submitted_timestamp_ms = timestamp_ms
            self._detector.detect_async(mp_image, timestamp_ms)
        return timestamp_ms

    @staticmethod
    def _resize_for_inference(frame: Any) -> Any:
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("frame must have positive dimensions")
        scale = min(config.INFERENCE_WIDTH / width, config.INFERENCE_HEIGHT / height)
        if scale >= 1.0:
            return frame
        target = max(1, round(width * scale)), max(1, round(height * scale))
        return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)

    def _result_callback(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        completed_at_ms = monotonic_ms()
        snapshot = ResultSnapshot(
            result=result,
            timestamp_ms=timestamp_ms,
            completed_at_ms=completed_at_ms,
            callback_latency_ms=max(0, completed_at_ms - timestamp_ms),
        )
        with self._state_lock:
            if self._closed:
                return
            if self._latest is None or timestamp_ms > self._latest.timestamp_ms:
                self._latest = snapshot
                self._completion_times_ms.append(completed_at_ms)
                self._prune_completion_times(completed_at_ms)

    def get_latest_result(
        self,
        max_age_ms: int = config.MAX_RESULT_AGE_MS,
        now_ms: int | None = None,
    ) -> ResultSnapshot | None:
        """Return an atomic latest-result snapshot if it is still fresh."""
        current_ms = monotonic_ms() if now_ms is None else now_ms
        with self._state_lock:
            snapshot = self._latest
            if snapshot is None or current_ms - snapshot.timestamp_ms > max_age_ms:
                return None
            return snapshot

    def get_diagnostics(self, now_ms: int | None = None) -> TrackerDiagnostics:
        current_ms = monotonic_ms() if now_ms is None else now_ms
        with self._state_lock:
            self._prune_completion_times(current_ms)
            snapshot = self._latest
            result_fps = self._rate(self._completion_times_ms)
            return TrackerDiagnostics(
                result_fps=result_fps,
                result_age_ms=(
                    max(0, current_ms - snapshot.timestamp_ms) if snapshot is not None else None
                ),
                callback_latency_ms=(
                    snapshot.callback_latency_ms if snapshot is not None else None
                ),
            )

    @staticmethod
    def get_index_finger_tip(snapshot: ResultSnapshot) -> tuple[float, float] | None:
        """Return normalized index-tip coordinates from a result snapshot."""
        if not snapshot.result.hand_landmarks:
            return None
        landmark = snapshot.result.hand_landmarks[0][HandTracker.INDEX_FINGER_TIP]
        return float(landmark.x), float(landmark.y)

    @staticmethod
    def get_fingers_up(snapshot: ResultSnapshot) -> list[int]:
        """Return geometry-derived thumb-to-pinky extension states."""
        if not snapshot.result.hand_landmarks:
            return []
        world = snapshot.result.hand_world_landmarks
        world_hand = world[0] if world else None
        return finger_states(snapshot.result.hand_landmarks[0], world_hand)

    def draw_landmarks(
        self,
        frame: Any,
        snapshot: ResultSnapshot | None,
        enabled: bool | None = None,
    ) -> None:
        """Render a fresh result on the display frame (main thread only)."""
        should_draw = config.DRAW_HAND_LANDMARKS if enabled is None else enabled
        if not should_draw or snapshot is None:
            return
        height, width = frame.shape[:2]
        for hand_landmarks in snapshot.result.hand_landmarks:
            for connection in self._connections:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]
                start_pos = (round(start.x * width), round(start.y * height))
                end_pos = (round(end.x * width), round(end.y * height))
                cv2.line(frame, start_pos, end_pos, (0, 220, 0), 3, cv2.LINE_AA)
            for landmark in hand_landmarks:
                position = (round(landmark.x * width), round(landmark.y * height))
                cv2.circle(frame, position, 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, position, 5, (255, 80, 0), -1, cv2.LINE_AA)

    def _prune_completion_times(self, now_ms: int) -> None:
        cutoff = now_ms - config.METRICS_WINDOW_MS
        while self._completion_times_ms and self._completion_times_ms[0] < cutoff:
            self._completion_times_ms.popleft()

    @staticmethod
    def _rate(timestamps_ms: deque[int]) -> float:
        if len(timestamps_ms) < 2:
            return 0.0
        elapsed_ms = timestamps_ms[-1] - timestamps_ms[0]
        return (len(timestamps_ms) - 1) * 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

    def close(self) -> None:
        """Stop accepting callbacks and safely release MediaPipe resources."""
        with self._detector_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                self._latest = None
            self._detector.close()
