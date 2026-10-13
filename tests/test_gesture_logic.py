import unittest

from gesture_logic import GestureMode, classify_gesture


class GestureLogicTests(unittest.TestCase):
    def test_draw_mode(self):
        self.assertEqual(classify_gesture([0, 1, 0, 0, 0]), GestureMode.DRAW)
        self.assertEqual(classify_gesture([1, 1, 0, 1, 1]), GestureMode.DRAW)

    def test_move_mode(self):
        self.assertEqual(classify_gesture([0, 1, 1, 0, 0]), GestureMode.MOVE)

    def test_erase_mode(self):
        self.assertEqual(classify_gesture([0, 1, 1, 1, 0]), GestureMode.ERASE)

    def test_clear_mode_has_priority(self):
        self.assertEqual(classify_gesture([1, 1, 1, 1, 1]), GestureMode.CLEAR)
        self.assertEqual(classify_gesture([0, 1, 1, 1, 1]), GestureMode.CLEAR)

    def test_idle_for_unrecognized_or_invalid_state(self):
        self.assertEqual(classify_gesture([0, 0, 0, 0, 0]), GestureMode.IDLE)
        self.assertEqual(classify_gesture([]), GestureMode.IDLE)
        self.assertEqual(classify_gesture([1, 1]), GestureMode.IDLE)


if __name__ == "__main__":
    unittest.main()
