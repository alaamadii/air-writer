import unittest

from gesture_logic import GestureMode, GestureStateMachine


class GestureStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = GestureStateMachine(
            stabilization_ms=80,
            release_grace_ms=100,
            clear_hold_ms=600,
            max_gap_ms=250,
        )

    def test_action_requires_stabilization(self):
        self.assertEqual(self.machine.update(GestureMode.DRAW, 0).mode, GestureMode.IDLE)
        self.assertEqual(self.machine.update(GestureMode.DRAW, 79).mode, GestureMode.IDLE)
        self.assertEqual(self.machine.update(GestureMode.DRAW, 80).mode, GestureMode.DRAW)

    def test_release_grace_survives_isolated_missing_result(self):
        self.machine.update(GestureMode.MOVE, 0)
        self.machine.update(GestureMode.MOVE, 80)
        self.assertEqual(self.machine.update(GestureMode.IDLE, 100).mode, GestureMode.MOVE)
        self.assertEqual(self.machine.update(GestureMode.MOVE, 130).mode, GestureMode.MOVE)
        self.machine.update(GestureMode.IDLE, 150)
        self.assertEqual(self.machine.update(GestureMode.IDLE, 250).mode, GestureMode.IDLE)

    def test_duplicate_timestamp_does_not_advance_candidate(self):
        self.machine.update(GestureMode.DRAW, 100)
        self.machine.update(GestureMode.DRAW, 100)
        self.assertEqual(self.machine.mode, GestureMode.IDLE)
        self.assertEqual(self.machine.update(GestureMode.DRAW, 179).mode, GestureMode.IDLE)
        self.assertEqual(self.machine.update(GestureMode.DRAW, 180).mode, GestureMode.DRAW)

    def test_large_gap_resets_state(self):
        self.machine.update(GestureMode.DRAW, 0)
        self.machine.update(GestureMode.DRAW, 80)
        self.assertEqual(self.machine.mode, GestureMode.DRAW)
        self.assertEqual(self.machine.update(GestureMode.DRAW, 400).mode, GestureMode.IDLE)

    def test_clear_requires_hold_and_fires_once(self):
        self.machine.update(GestureMode.CLEAR, 0)
        self.machine.update(GestureMode.CLEAR, 200)
        self.machine.update(GestureMode.CLEAR, 400)
        self.assertFalse(self.machine.update(GestureMode.CLEAR, 599).clear_triggered)
        fired = self.machine.update(GestureMode.CLEAR, 600)
        self.assertTrue(fired.clear_triggered)
        self.assertEqual(fired.mode, GestureMode.CLEAR)
        self.assertFalse(self.machine.update(GestureMode.CLEAR, 700).clear_triggered)

        self.machine.update(GestureMode.IDLE, 720)
        self.machine.update(GestureMode.CLEAR, 800)
        self.machine.update(GestureMode.CLEAR, 1000)
        self.machine.update(GestureMode.CLEAR, 1200)
        self.assertTrue(self.machine.update(GestureMode.CLEAR, 1400).clear_triggered)

    def test_reset_prevents_stale_time_from_advancing_hold(self):
        self.machine.update(GestureMode.CLEAR, 0)
        self.machine.reset()
        self.assertFalse(self.machine.update(GestureMode.CLEAR, 1000).clear_triggered)
        self.machine.update(GestureMode.CLEAR, 1200)
        self.machine.update(GestureMode.CLEAR, 1400)
        self.assertFalse(self.machine.update(GestureMode.CLEAR, 1599).clear_triggered)
        self.assertTrue(self.machine.update(GestureMode.CLEAR, 1600).clear_triggered)


if __name__ == "__main__":
    unittest.main()
