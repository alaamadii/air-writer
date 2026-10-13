"""Pure hand-landmark geometry used by gesture recognition."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median
from typing import Any

Point3D = tuple[float, float, float]

_FINGER_CHAINS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def _as_point(value: Any) -> Point3D | None:
    try:
        if hasattr(value, "x"):
            coordinates = (float(value.x), float(value.y), float(getattr(value, "z", 0.0)))
        else:
            coordinates = (
                float(value[0]),
                float(value[1]),
                float(value[2]) if len(value) > 2 else 0.0,
            )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return coordinates if all(math.isfinite(item) for item in coordinates) else None


def _vector(start: Point3D, end: Point3D) -> Point3D:
    return end[0] - start[0], end[1] - start[1], end[2] - start[2]


def distance(first: Any, second: Any) -> float | None:
    """Return the 3D Euclidean distance, or ``None`` for invalid points."""
    a = _as_point(first)
    b = _as_point(second)
    if a is None or b is None:
        return None
    delta = _vector(a, b)
    return math.sqrt(sum(component * component for component in delta))


def joint_angle(first: Any, vertex: Any, third: Any) -> float | None:
    """Return the smaller angle at ``vertex`` in degrees.

    Degenerate or non-finite inputs return ``None`` instead of producing an
    unstable classification.
    """
    a = _as_point(first)
    b = _as_point(vertex)
    c = _as_point(third)
    if a is None or b is None or c is None:
        return None
    first_vector = _vector(b, a)
    second_vector = _vector(b, c)
    first_length = math.sqrt(sum(item * item for item in first_vector))
    second_length = math.sqrt(sum(item * item for item in second_vector))
    if first_length <= 1e-9 or second_length <= 1e-9:
        return None
    cosine = sum(x * y for x, y in zip(first_vector, second_vector, strict=True))
    cosine /= first_length * second_length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _valid_landmarks(landmarks: Sequence[Any] | None) -> list[Point3D] | None:
    if landmarks is None or len(landmarks) < 21:
        return None
    points = [_as_point(landmark) for landmark in landmarks[:21]]
    if any(point is None for point in points):
        return None
    return [point for point in points if point is not None]


def _palm_size(points: Sequence[Point3D]) -> float | None:
    # Several palm spans make the scale robust to one foreshortened axis.
    spans = [distance(points[0], points[index]) for index in (5, 9, 13, 17)] + [
        distance(points[5], points[17]),
        distance(points[0], points[9]),
    ]
    valid_spans = [span for span in spans if span is not None and span > 1e-9]
    return median(valid_spans) if valid_spans else None


def finger_states(
    normalized_landmarks: Sequence[Any] | None,
    world_landmarks: Sequence[Any] | None = None,
) -> list[int]:
    """Classify thumb-to-pinky extension using joint angles and palm scale.

    Metric 3D world landmarks are preferred. Normalized landmarks are a safe
    fallback when world coordinates are absent or invalid. The rules do not
    depend on screen x/y direction or handedness, so in-plane rotation and
    mirroring do not invert the result.
    """
    normalized_points = _valid_landmarks(normalized_landmarks)
    if normalized_points is None:
        return []
    world_points = _valid_landmarks(world_landmarks)
    world_palm_size = _palm_size(world_points) if world_points is not None else None
    if world_points is not None and world_palm_size is not None:
        points = world_points
        palm_size = world_palm_size
    else:
        points = normalized_points
        palm_size = _palm_size(points)
    if palm_size is None or palm_size <= 1e-9:
        return []

    thumb_mcp_angle = joint_angle(points[1], points[2], points[3])
    thumb_ip_angle = joint_angle(points[2], points[3], points[4])
    thumb_length = distance(points[1], points[4])
    thumb_spread = distance(points[4], points[5])
    thumb_extended = all(
        (
            thumb_mcp_angle is not None and thumb_mcp_angle >= 135.0,
            thumb_ip_angle is not None and thumb_ip_angle >= 145.0,
            thumb_length is not None and thumb_length / palm_size >= 0.55,
            thumb_spread is not None and thumb_spread / palm_size >= 0.45,
        )
    )

    states = [int(thumb_extended)]
    for mcp, pip, dip, tip in _FINGER_CHAINS:
        pip_angle = joint_angle(points[mcp], points[pip], points[dip])
        dip_angle = joint_angle(points[pip], points[dip], points[tip])
        reach = distance(points[mcp], points[tip])
        extended = all(
            (
                pip_angle is not None and pip_angle >= 135.0,
                dip_angle is not None and dip_angle >= 135.0,
                reach is not None and reach / palm_size >= 0.62,
            )
        )
        states.append(int(extended))
    return states
