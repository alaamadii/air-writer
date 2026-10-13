import unittest

import numpy as np

from gesture_logic import GestureMode
from main import _apply_canvas_segment


class CanvasActionTests(unittest.TestCase):
    def test_draw_segment_uses_selected_color_and_thickness(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        _apply_canvas_segment(
            canvas,
            GestureMode.DRAW,
            ((10, 50), (90, 50)),
            brush_color=(255, 0, 0),
            brush_thickness=4,
        )
        self.assertTrue(np.array_equal(canvas[50, 50], (255, 0, 0)))

    def test_erase_segment_removes_existing_drawing(self):
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        _apply_canvas_segment(canvas, GestureMode.DRAW, ((20, 100), (180, 100)))
        self.assertTrue(np.any(canvas[100, 20:181] != 0))

        _apply_canvas_segment(canvas, GestureMode.ERASE, ((90, 100), (110, 100)))

        self.assertTrue(np.all(canvas[100, 90:111] == 0))
        self.assertTrue(np.any(canvas[100, 20:50] != 0))


if __name__ == "__main__":
    unittest.main()
