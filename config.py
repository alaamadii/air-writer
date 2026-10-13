# Application Configuration

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# MediaPipe receives a smaller same-aspect-ratio copy; display remains full size.
INFERENCE_WIDTH = 960
INFERENCE_HEIGHT = 540

# Drawing settings
DEFAULT_BRUSH_COLOR = (0, 0, 255)  # Red in BGR format
DEFAULT_BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40
MAX_STROKE_JUMP_PIXELS = 160

# On-screen toolbar settings. Hold the MOVE gesture over a control to select it.
TOOLBAR_HEIGHT = 112
TOOLBAR_DWELL_MS = 450

# MediaPipe settings
# Detection starts a hand track, presence decides whether a track still contains
# a hand, and tracking measures landmark continuity. They are distinct signals.
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MAX_RESULT_AGE_MS = 200
DRAW_HAND_LANDMARKS = True

# One Euro fingertip smoothing. Raise MIN_CUTOFF for less smoothing; raise BETA
# for faster response during quick motion. Long gaps reset instead of bridging.
SMOOTHING_MIN_CUTOFF = 2.0
SMOOTHING_BETA = 5.0
SMOOTHING_DERIVATIVE_CUTOFF = 1.0
SMOOTHING_MAX_GAP_MS = 250

# Gesture timings use MediaPipe result timestamps, never display-frame counts.
GESTURE_STABILIZATION_MS = 50
GESTURE_RELEASE_GRACE_MS = 100
CLEAR_HOLD_MS = 600
GESTURE_MAX_GAP_MS = 250

# Optional rolling diagnostics; disabled by default to keep rendering minimal.
SHOW_DIAGNOSTICS = False
METRICS_WINDOW_MS = 1_000
