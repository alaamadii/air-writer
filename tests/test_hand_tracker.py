import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from mediapipe.tasks.python import vision

from hand_tracker import HandTracker


class HandTrackerTests(unittest.TestCase):
    @patch("hand_tracker.mp.Image")
    @patch("hand_tracker.vision.HandLandmarker.create_from_options")
    def test_live_stream_options_and_downscaled_async_submission(
        self, create_detector, create_image
    ):
        detector = MagicMock()
        create_detector.return_value = detector
        media_image = object()
        create_image.return_value = media_image
        tracker = HandTracker()
        options = create_detector.call_args.args[0]

        self.assertEqual(options.running_mode, vision.RunningMode.LIVE_STREAM)
        self.assertIsNotNone(options.result_callback)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch("hand_tracker.monotonic_ms", return_value=100):
            self.assertEqual(tracker.process_frame(frame), 100)
        image_data = create_image.call_args.kwargs["data"]
        self.assertEqual(image_data.shape, (540, 960, 3))
        image, timestamp = detector.detect_async.call_args.args
        self.assertIs(image, media_image)
        self.assertEqual(timestamp, 100)
        tracker.close()

    @patch("hand_tracker.mp.Image")
    @patch("hand_tracker.vision.HandLandmarker.create_from_options")
    def test_submission_timestamps_are_strictly_increasing(self, create_detector, create_image):
        create_detector.return_value = MagicMock()
        create_image.return_value = object()
        tracker = HandTracker()
        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        with patch("hand_tracker.monotonic_ms", return_value=500):
            self.assertEqual(tracker.process_frame(frame), 500)
            self.assertEqual(tracker.process_frame(frame), 501)
        tracker.close()

    def test_latest_callback_result_replaces_older_and_expires(self):
        tracker = HandTracker.__new__(HandTracker)
        tracker._state_lock = threading.Lock()
        tracker._closed = False
        tracker._latest = None
        tracker._completion_times_ms = deque()
        newer = SimpleNamespace(hand_landmarks=[])
        older = SimpleNamespace(hand_landmarks=[])

        with patch("hand_tracker.monotonic_ms", side_effect=(1_010, 1_020)):
            tracker._result_callback(newer, None, 1_000)
            tracker._result_callback(older, None, 999)

        self.assertIs(tracker.get_latest_result(now_ms=1_100).result, newer)
        self.assertIsNone(tracker.get_latest_result(now_ms=1_201))

    def test_callback_does_not_publish_after_close_begins(self):
        tracker = HandTracker.__new__(HandTracker)
        tracker._state_lock = threading.Lock()
        tracker._closed = True
        tracker._latest = None
        tracker._completion_times_ms = deque()

        with patch("hand_tracker.monotonic_ms", return_value=1_010):
            tracker._result_callback(SimpleNamespace(), None, 1_000)
        self.assertIsNone(tracker._latest)


if __name__ == "__main__":
    unittest.main()
