import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

class HandTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Landmark 8 is the tip of the index finger
        self.INDEX_FINGER_TIP = 8

        # For drawing manually
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def process_frame(self, frame):
        """Processes an image frame and returns hand landmarks and the frame with drawn landmarks."""
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hands using Tasks API
        self.results = self.detector.detect(mp_image)
        
        # Draw hand landmarks manually if detected (Optional, for visual feedback)
        if self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                h, w, _ = frame.shape
                
                # Draw lines between connected landmarks
                for connection in self.connections:
                    start_idx = connection.start
                    end_idx = connection.end
                    
                    start_lm = hand_landmarks[start_idx]
                    end_lm = hand_landmarks[end_idx]
                    
                    start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                    
                    cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
                
                # Draw dots for each landmark
                for landmark in hand_landmarks:
                    pos = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(frame, pos, 4, (255, 0, 0), -1)

        return frame

    def get_index_finger_tip(self, w, h):
        """Returns the (x, y) coordinates of the index finger tip, if visible."""
        if self.results and self.results.hand_landmarks and len(self.results.hand_landmarks) > 0:
            hand_landmarks = self.results.hand_landmarks[0]
            index_tip = hand_landmarks[self.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates back to pixel values
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            return (cx, cy)
            
        return None

    def get_fingers_up(self):
        """Returns a list of 5 integers (0 or 1) indicating which fingers are up.
           Order: [Thumb, Index, Middle, Ring, Pinky]"""
        if not self.results or not self.results.hand_landmarks or len(self.results.hand_landmarks) == 0:
            return []
            
        hand_landmarks = self.results.hand_landmarks[0]
        handedness = self.results.handedness[0][0].category_name # "Left" or "Right"
        
        tips_ids = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
        fingers = []
        
        # Thumb
        if handedness == "Right":
            if hand_landmarks[4].x < hand_landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks[4].x > hand_landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # Other fingers
        for tip_id in tips_ids:
            if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
