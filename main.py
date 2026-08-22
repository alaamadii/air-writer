import cv2
import numpy as np

import config
from gesture_logic import GestureMode, classify_gesture
from hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Check camera permissions or update CAMERA_INDEX in config.py."
        )

    tracker = HandTracker()
    canvas = None
    prev_x, prev_y = 0, 0

    print("Air Writing started")
    print("- Index finger: draw")
    print("- Index + middle: move")
    print("- Index + middle + ring: erase")
    print("- Open palm: clear")
    print("- C: clear canvas")
    print("- Q: quit")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            height, width, channels = frame.shape

            if canvas is None:
                canvas = np.zeros((height, width, channels), dtype=np.uint8)

            frame = tracker.process_frame(frame)
            index_tip = tracker.get_index_finger_tip(width, height)
            fingers = tracker.get_fingers_up()
            mode = classify_gesture(fingers)

            if index_tip is not None:
                cx, cy = index_tip

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                if mode == GestureMode.DRAW:
                    cv2.circle(
                        frame,
                        (cx, cy),
                        10,
                        config.DEFAULT_BRUSH_COLOR,
                        cv2.FILLED,
                    )
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (cx, cy),
                        config.DEFAULT_BRUSH_COLOR,
                        config.DEFAULT_BRUSH_THICKNESS,
                    )
                elif mode == GestureMode.MOVE:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                elif mode == GestureMode.ERASE:
                    cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 2)
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (cx, cy),
                        (0, 0, 0),
                        config.ERASER_THICKNESS,
                    )
                elif mode == GestureMode.CLEAR:
                    canvas = np.zeros((height, width, channels), dtype=np.uint8)

                prev_x, prev_y = cx, cy
            else:
                prev_x, prev_y = 0, 0

            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
            final_frame = cv2.add(frame_bg, canvas)

            cv2.imshow("Air Writing", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                canvas = np.zeros((height, width, channels), dtype=np.uint8)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
