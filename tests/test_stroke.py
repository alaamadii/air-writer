import unittest

from gesture_logic import GestureMode
from stroke import StrokeController


class StrokeControllerTests(unittest.TestCase):
    def test_starts_and_continues_only_within_same_mode(self):
        strokes = StrokeController(max_jump_pixels=50)
        self.assertEqual(strokes.update(GestureMode.DRAW, (10, 10)), ((10, 10), (10, 10)))
        self.assertEqual(strokes.update(GestureMode.DRAW, (20, 20)), ((10, 10), (20, 20)))
        self.assertEqual(strokes.update(GestureMode.ERASE, (30, 30)), ((30, 30), (30, 30)))
        self.assertEqual(strokes.update(GestureMode.ERASE, (35, 35)), ((30, 30), (35, 35)))

    def test_non_drawing_mode_and_hand_loss_reset_continuity(self):
        strokes = StrokeController(max_jump_pixels=50)
        strokes.update(GestureMode.DRAW, (10, 10))
        self.assertIsNone(strokes.update(GestureMode.MOVE, (20, 20)))
        self.assertEqual(strokes.update(GestureMode.DRAW, (30, 30)), ((30, 30), (30, 30)))
        self.assertIsNone(strokes.update(GestureMode.DRAW, None))
        self.assertEqual(strokes.update(GestureMode.DRAW, (40, 40)), ((40, 40), (40, 40)))

    def test_large_jump_is_not_connected_and_restarts(self):
        strokes = StrokeController(max_jump_pixels=20)
        strokes.update(GestureMode.DRAW, (0, 0))
        self.assertIsNone(strokes.update(GestureMode.DRAW, (100, 100)))
        self.assertEqual(strokes.update(GestureMode.DRAW, (105, 105)), ((100, 100), (105, 105)))


if __name__ == "__main__":
    unittest.main()
