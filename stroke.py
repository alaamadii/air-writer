"""Pure stroke continuity state for drawing and erasing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gesture_logic import GestureMode

Point = tuple[int, int]
Segment = tuple[Point, Point]


@dataclass
class StrokeController:
    """Connect only consecutive points from the same drawing mode."""

    max_jump_pixels: float
    _mode: GestureMode | None = None
    _previous: Point | None = None

    def update(self, mode: GestureMode, point: Point | None) -> Segment | None:
        if mode not in (GestureMode.DRAW, GestureMode.ERASE) or point is None:
            self.reset()
            return None

        if self._mode != mode or self._previous is None:
            self._mode = mode
            self._previous = point
            return point, point

        previous = self._previous
        self._previous = point
        if math.dist(previous, point) > self.max_jump_pixels:
            return None
        return previous, point

    def reset(self) -> None:
        self._mode = None
        self._previous = None
