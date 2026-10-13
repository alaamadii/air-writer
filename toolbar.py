"""OpenCV toolbar for selecting drawing colors and brush sizes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2

Color = tuple[int, int, int]
Point = tuple[int, int]


class ToolbarAction(str, Enum):
    STYLE_CHANGED = "style_changed"
    CLEAR = "clear"


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def contains(self, point: Point) -> bool:
        px, py = point
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


@dataclass(frozen=True)
class ToolbarButton:
    key: str
    kind: str
    value: Color | int | None
    rect: Rect
    label: str = ""


class DrawingToolbar:
    """State, hit testing, dwell selection, and rendering for the toolbar."""

    COLORS: tuple[tuple[str, Color], ...] = (
        ("Red", (0, 0, 255)),
        ("Blue", (255, 90, 20)),
        ("Green", (60, 210, 60)),
        ("Yellow", (0, 220, 255)),
        ("Purple", (210, 70, 190)),
        ("White", (255, 255, 255)),
    )
    BRUSH_SIZES: tuple[tuple[str, int], ...] = (
        ("S", 4),
        ("M", 8),
        ("L", 14),
    )

    def __init__(
        self,
        default_color: Color,
        default_brush_size: int,
        height: int = 112,
        dwell_ms: int = 450,
    ) -> None:
        if height <= 0 or dwell_ms < 0:
            raise ValueError("toolbar height must be positive and dwell time non-negative")
        self.height = height
        self.dwell_ms = dwell_ms
        self.selected_color = default_color
        self.brush_thickness = default_brush_size
        self._hover_key: str | None = None
        self._hover_since_ms = 0
        self._hover_latched = False

    def buttons(self, frame_width: int) -> tuple[ToolbarButton, ...]:
        """Return the controls for the current frame width."""
        del frame_width  # Reserved for a future compact layout.
        y = 45
        height = 50
        buttons: list[ToolbarButton] = [
            ToolbarButton("clear", "clear", None, Rect(132, y, 82, height), "CLEAR")
        ]

        x = 290
        for index, (_name, color) in enumerate(self.COLORS):
            buttons.append(ToolbarButton(f"color_{index}", "color", color, Rect(x, y, 46, height)))
            x += 54

        x = 690
        for index, (label, size) in enumerate(self.BRUSH_SIZES):
            buttons.append(
                ToolbarButton(f"size_{index}", "size", size, Rect(x, y, 62, height), label)
            )
            x += 70
        return tuple(buttons)

    def contains(self, point: Point | None) -> bool:
        return point is not None and 0 <= point[1] < self.height

    def hit_test(self, point: Point | None, frame_width: int) -> ToolbarButton | None:
        if point is None or not self.contains(point):
            return None
        return next(
            (button for button in self.buttons(frame_width) if button.rect.contains(point)),
            None,
        )

    def activate(self, point: Point, frame_width: int) -> ToolbarAction | None:
        """Activate a control immediately, as used by mouse clicks."""
        button = self.hit_test(point, frame_width)
        if button is None:
            return None
        if button.kind == "clear":
            return ToolbarAction.CLEAR
        if button.kind == "color":
            assert isinstance(button.value, tuple)
            self.selected_color = button.value
            return ToolbarAction.STYLE_CHANGED
        if button.kind == "size":
            assert isinstance(button.value, int)
            self.brush_thickness = button.value
            return ToolbarAction.STYLE_CHANGED
        return None

    def update_pointer(
        self,
        point: Point | None,
        selecting: bool,
        timestamp_ms: int,
        frame_width: int,
    ) -> ToolbarAction | None:
        """Select a hovered control after a continuous MOVE-gesture dwell."""
        button = self.hit_test(point, frame_width) if selecting else None
        if button is None:
            self.reset_hover()
            return None
        if button.key != self._hover_key:
            self._hover_key = button.key
            self._hover_since_ms = timestamp_ms
            self._hover_latched = False
            return None
        if self._hover_latched or timestamp_ms - self._hover_since_ms < self.dwell_ms:
            return None
        self._hover_latched = True
        assert point is not None
        return self.activate(point, frame_width)

    def reset_hover(self) -> None:
        self._hover_key = None
        self._hover_since_ms = 0
        self._hover_latched = False

    def render(self, frame: Any, mode_label: str, timestamp_ms: int) -> None:
        """Render a translucent, camera-friendly toolbar above the scene."""
        frame_height, frame_width = frame.shape[:2]
        panel_height = min(self.height, frame_height)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, panel_height), (18, 23, 32), cv2.FILLED)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        cv2.line(frame, (0, panel_height - 1), (frame_width, panel_height - 1), (75, 84, 99), 1)

        cv2.putText(
            frame, "AIR WRITER", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"MODE  {mode_label.upper()}",
            (16, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (176, 190, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, "COLORS", (290, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (176, 190, 210), 1)
        cv2.putText(frame, "BRUSH", (690, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (176, 190, 210), 1)

        for button in self.buttons(frame_width):
            self._render_button(frame, button, timestamp_ms)

        if frame_width >= 1050:
            cv2.putText(
                frame,
                "Use MOVE gesture or mouse to select",
                (925, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (205, 213, 225),
                1,
                cv2.LINE_AA,
            )

    def _render_button(self, frame: Any, button: ToolbarButton, timestamp_ms: int) -> None:
        rect = button.rect
        hovered = button.key == self._hover_key
        border = (0, 220, 255) if hovered else (91, 101, 118)
        cv2.rectangle(
            frame,
            (rect.x, rect.y),
            (rect.x + rect.width, rect.y + rect.height),
            (35, 43, 56),
            cv2.FILLED,
        )
        cv2.rectangle(
            frame,
            (rect.x, rect.y),
            (rect.x + rect.width, rect.y + rect.height),
            border,
            2 if hovered else 1,
        )

        center = (rect.x + rect.width // 2, rect.y + rect.height // 2)
        selected = False
        if button.kind == "color":
            assert isinstance(button.value, tuple)
            cv2.circle(frame, center, 16, button.value, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, center, 17, (225, 230, 238), 1, cv2.LINE_AA)
            selected = button.value == self.selected_color
        elif button.kind == "size":
            assert isinstance(button.value, int)
            cv2.line(
                frame,
                (rect.x + 13, center[1]),
                (rect.x + rect.width - 13, center[1]),
                self.selected_color,
                button.value,
                cv2.LINE_AA,
            )
            selected = button.value == self.brush_thickness
        else:
            cv2.putText(
                frame,
                button.label,
                (rect.x + 10, center[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 238, 244),
                1,
                cv2.LINE_AA,
            )

        if selected:
            cv2.rectangle(
                frame,
                (rect.x - 2, rect.y - 2),
                (rect.x + rect.width + 2, rect.y + rect.height + 2),
                (255, 255, 255),
                2,
            )

        if hovered and self.dwell_ms > 0:
            progress = min(1.0, max(0.0, (timestamp_ms - self._hover_since_ms) / self.dwell_ms))
            progress_x = rect.x + round(rect.width * progress)
            cv2.line(
                frame,
                (rect.x, rect.y + rect.height - 2),
                (progress_x, rect.y + rect.height - 2),
                (0, 220, 255),
                4,
            )
