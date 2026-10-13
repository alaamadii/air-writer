"""Pure gesture classification and timestamp-based stabilization."""

from dataclasses import dataclass
from enum import Enum


class GestureMode(str, Enum):
    DRAW = "draw"
    MOVE = "move"
    ERASE = "erase"
    CLEAR = "clear"
    IDLE = "idle"


def classify_gesture(fingers: list[int]) -> GestureMode:
    """Map a five-finger state to the application mode.

    Finger order is: thumb, index, middle, ring, pinky.
    """
    if len(fingers) != 5:
        return GestureMode.IDLE

    _, index, middle, ring, pinky = fingers

    # Do not make CLEAR depend on the thumb classifier: thumb pose varies
    # significantly with camera angle while four extended fingers are enough
    # to distinguish an open palm from the three-finger eraser gesture.
    if index == 1 and middle == 1 and ring == 1 and pinky == 1:
        return GestureMode.CLEAR
    if index == 1 and middle == 1 and ring == 1 and pinky == 0:
        return GestureMode.ERASE
    if index == 1 and middle == 1 and ring == 0:
        return GestureMode.MOVE
    if index == 1 and middle == 0:
        return GestureMode.DRAW

    return GestureMode.IDLE


@dataclass(frozen=True)
class GestureUpdate:
    mode: GestureMode
    clear_triggered: bool = False


class GestureStateMachine:
    """Stabilize raw gestures using MediaPipe result timestamps, not frames."""

    def __init__(
        self,
        stabilization_ms: int = 80,
        release_grace_ms: int = 100,
        clear_hold_ms: int = 600,
        max_gap_ms: int = 250,
    ) -> None:
        if min(stabilization_ms, release_grace_ms, clear_hold_ms, max_gap_ms) < 0:
            raise ValueError("gesture timings must be non-negative")
        self.stabilization_ms = stabilization_ms
        self.release_grace_ms = release_grace_ms
        self.clear_hold_ms = clear_hold_ms
        self.max_gap_ms = max_gap_ms
        self.reset()

    @property
    def mode(self) -> GestureMode:
        return self._stable_mode

    def update(self, raw_mode: GestureMode, timestamp_ms: int) -> GestureUpdate:
        """Consume one new result; duplicate/older timestamps are ignored."""
        if self._last_timestamp_ms is not None:
            if timestamp_ms <= self._last_timestamp_ms:
                return GestureUpdate(self._stable_mode)
            if timestamp_ms - self._last_timestamp_ms > self.max_gap_ms:
                self.reset()
        self._last_timestamp_ms = timestamp_ms

        if raw_mode != GestureMode.CLEAR:
            self._clear_latched = False

        if raw_mode == GestureMode.CLEAR:
            return self._update_clear(timestamp_ms)

        if self._stable_mode == GestureMode.CLEAR:
            self._stable_mode = GestureMode.IDLE
            self._candidate_mode = None

        if raw_mode == self._stable_mode:
            self._candidate_mode = None
            return GestureUpdate(self._stable_mode)

        if raw_mode != self._candidate_mode:
            self._candidate_mode = raw_mode
            self._candidate_since_ms = timestamp_ms
            return GestureUpdate(self._stable_mode)

        delay = self.release_grace_ms if raw_mode == GestureMode.IDLE else self.stabilization_ms
        if timestamp_ms - self._candidate_since_ms >= delay:
            self._stable_mode = raw_mode
            self._candidate_mode = None
        return GestureUpdate(self._stable_mode)

    def _update_clear(self, timestamp_ms: int) -> GestureUpdate:
        if self._clear_latched:
            self._stable_mode = GestureMode.CLEAR
            return GestureUpdate(self._stable_mode)
        if self._candidate_mode != GestureMode.CLEAR:
            self._candidate_mode = GestureMode.CLEAR
            self._candidate_since_ms = timestamp_ms
            return GestureUpdate(self._stable_mode)
        if timestamp_ms - self._candidate_since_ms >= self.clear_hold_ms:
            self._stable_mode = GestureMode.CLEAR
            self._candidate_mode = None
            self._clear_latched = True
            return GestureUpdate(self._stable_mode, clear_triggered=True)
        return GestureUpdate(self._stable_mode)

    def reset(self) -> None:
        """Return to idle after stale tracking or explicit hand loss."""
        self._stable_mode = GestureMode.IDLE
        self._candidate_mode: GestureMode | None = None
        self._candidate_since_ms = 0
        self._clear_latched = False
        self._last_timestamp_ms: int | None = None
