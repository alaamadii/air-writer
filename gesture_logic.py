"""Pure gesture classification helpers for the air-writing application."""

from enum import Enum


class GestureMode(str, Enum):
    DRAW = "draw"
    MOVE = "move"
    ERASE = "erase"
    CLEAR = "clear"
    IDLE = "idle"


def classify_gesture(fingers: list[int]) -> GestureMode:
    """Map a five-finger state to the application mode.

    Finger order is: thumb, index, middle, ring, pinky.
    """
    if len(fingers) != 5:
        return GestureMode.IDLE

    _, index, middle, ring, pinky = fingers

    if fingers == [1, 1, 1, 1, 1]:
        return GestureMode.CLEAR
    if index == 1 and middle == 1 and ring == 1 and pinky == 0:
        return GestureMode.ERASE
    if index == 1 and middle == 1 and ring == 0:
        return GestureMode.MOVE
    if index == 1 and middle == 0:
        return GestureMode.DRAW

    return GestureMode.IDLE
