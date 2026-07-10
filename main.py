import cv2
import numpy as np
import config
from hand_tracker import HandTracker

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Check camera permissions or update CAMERA_INDEX in config.py."
        )

    # Initialize Hand Tracker
    tracker = HandTracker()

    # Create a canvas for drawing (same size as camera frame)
    # Start with a dummy canvas, it will be resized when the first frame is read
    canvas = None

    # Track previous point for drawing lines
    prev_x, prev_y = 0, 0

    print("Air-Writing started!")
    print("- Raise your index finger to draw.")
    print("- Press 'c' to clear the canvas.")
    print("- Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera. Exiting...")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Initialize canvas if it's the first frame
        if canvas is None:
            canvas = np.zeros((h, w, c), dtype=np.uint8)

        # Process the frame to find hands
        frame = tracker.process_frame(frame)

        # Get index finger position and fingers state
        index_tip = tracker.get_index_finger_tip(w, h)
        fingers = tracker.get_fingers_up()

        if index_tip is not None and len(fingers) == 5:
            cx, cy = index_tip
            
            # If it's the first point detected, just update prev coordinates
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Check Gestures based on fingers up: [Thumb, Index, Middle, Ring, Pinky]
            if fingers[1] == 1 and fingers[2] == 0:
                # 1. DRAW MODE: Index finger up, Middle down
                cv2.circle(frame, (cx, cy), 10, config.DEFAULT_BRUSH_COLOR, cv2.FILLED)
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), config.DEFAULT_BRUSH_COLOR, config.DEFAULT_BRUSH_THICKNESS)
                prev_x, prev_y = cx, cy
            
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                # 2. MOVE MODE: Index and Middle fingers up
                # Do not draw, just update previous coordinates to prevent jump
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED) # Blue circle for move mode
                prev_x, prev_y = cx, cy
            
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                # 3. ERASE MODE: Index, Middle, and Ring fingers up
                # Draw black on the canvas to act as an eraser
                cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 2) # White circle for eraser cursor
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), config.ERASER_THICKNESS)
                prev_x, prev_y = cx, cy
                
            elif fingers == [1, 1, 1, 1, 1]:
                # 4. CLEAR ALL MODE: All 5 fingers open
                canvas = np.zeros((h, w, c), dtype=np.uint8)
                prev_x, prev_y = cx, cy
            else:
                # Idle or unrecognized gesture, just update prev coordinates
                prev_x, prev_y = cx, cy
        else:
            # If no hand detected, reset previous points
            prev_x, prev_y = 0, 0

        # Combine the frame and the canvas
        # Where canvas is not black, use canvas; otherwise use frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
        
        # Black out the area of the drawing in the frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add the drawing canvas to the frame
        final_frame = cv2.add(frame_bg, canvas)

        # Display the result
        cv2.imshow("Air-Writing", final_frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear the canvas
            canvas = np.zeros((h, w, c), dtype=np.uint8)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
