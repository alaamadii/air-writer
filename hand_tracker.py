from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


class HandTracker:
    INDEX_FINGER_TIP = 8

    def __init__(self):
        model_path = Path(__file__).resolve().with_name("hand_landmarker.task")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Make sure hand_landmarker.task is included with the project."
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.results = None

    def process_frame(self, frame):
        """Detect a hand and draw the tracked landmarks on the frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.detector.detect(mp_image)

        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                height, width, _ = frame.shape

                for connection in self.connections:
                    start_lm = hand_landmarks[connection.start]
                    end_lm = hand_landmarks[connection.end]
                    start_pos = (int(start_lm.x * width), int(start_lm.y * height))
                    end_pos = (int(end_lm.x * width), int(end_lm.y * height))
                    cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

                for landmark in hand_landmarks:
                    pos = (int(landmark.x * width), int(landmark.y * height))
                    cv2.circle(frame, pos, 4, (255, 0, 0), -1)

        return frame

    def get_index_finger_tip(self, width, height):
        """Return index-fingertip pixel coordinates when a hand is visible."""
        if not self.results or not self.results.hand_landmarks:
            return None

        hand_landmarks = self.results.hand_landmarks[0]
        index_tip = hand_landmarks[self.INDEX_FINGER_TIP]
        return int(index_tip.x * width), int(index_tip.y * height)

    def get_fingers_up(self):
        """Return finger states in thumb, index, middle, ring, pinky order."""
        if not self.results or not self.results.hand_landmarks:
            return []

        hand_landmarks = self.results.hand_landmarks[0]
        handedness = self.results.handedness[0][0].category_name
        fingers = []

        thumb_is_up = (
            hand_landmarks[4].x < hand_landmarks[3].x
            if handedness == "Right"
            else hand_landmarks[4].x > hand_landmarks[3].x
        )
        fingers.append(int(thumb_is_up))

        for tip_id in (8, 12, 16, 20):
            fingers.append(int(hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y))

        return fingers

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()
