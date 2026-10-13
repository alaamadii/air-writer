import math
import unittest

from hand_geometry import finger_states, joint_angle


def make_hand(extended: set[str]) -> list[tuple[float, float, float]]:
    points = [(0.0, 0.0, 0.0)] * 21
    points[0] = (0.0, 0.0, 0.0)
    points[1:5] = [
        (-0.20, -0.05, 0.0),
        (-0.40, -0.10, 0.0),
        (-0.62, -0.13, 0.0),
        (-0.85, -0.15, 0.0),
    ]
    if "thumb" not in extended:
        points[1:5] = [
            (-0.20, -0.05, 0.0),
            (-0.32, -0.15, 0.0),
            (-0.15, -0.22, 0.0),
            (0.02, -0.20, 0.0),
        ]

    finger_data = (
        ("index", 5, -0.30),
        ("middle", 9, -0.10),
        ("ring", 13, 0.10),
        ("pinky", 17, 0.28),
    )
    for name, start, x in finger_data:
        points[start] = (x, -0.38, 0.0)
        if name in extended:
            points[start + 1] = (x, -0.68, 0.0)
            points[start + 2] = (x, -0.94, 0.0)
            points[start + 3] = (x, -1.20, 0.0)
        else:
            points[start + 1] = (x, -0.64, 0.0)
            points[start + 2] = (x + 0.18, -0.48, 0.0)
            points[start + 3] = (x + 0.20, -0.28, 0.0)
    return points


def rotate(points: list[tuple[float, float, float]], degrees: float):
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    return [(x * cosine - y * sine, x * sine + y * cosine, z) for x, y, z in points]


class HandGeometryTests(unittest.TestCase):
    def test_joint_angles_and_degenerate_input(self):
        self.assertAlmostEqual(joint_angle((1, 0, 0), (0, 0, 0), (-1, 0, 0)), 180.0)
        self.assertAlmostEqual(joint_angle((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0)
        self.assertIsNone(joint_angle((0, 0, 0), (0, 0, 0), (1, 0, 0)))
        self.assertIsNone(joint_angle((math.nan, 0, 0), (0, 0, 0), (1, 0, 0)))

    def test_representative_finger_states(self):
        cases = (
            ({"thumb", "index", "middle", "ring", "pinky"}, [1, 1, 1, 1, 1]),
            (set(), [0, 0, 0, 0, 0]),
            ({"index"}, [0, 1, 0, 0, 0]),
            ({"index", "middle"}, [0, 1, 1, 0, 0]),
            ({"index", "middle", "ring"}, [0, 1, 1, 1, 0]),
        )
        for extended, expected in cases:
            with self.subTest(extended=extended):
                self.assertEqual(finger_states(make_hand(extended)), expected)

    def test_rotation_does_not_change_states(self):
        for extended in (
            {"index"},
            {"index", "middle", "ring"},
            {"thumb", "index", "middle", "ring", "pinky"},
        ):
            hand = make_hand(extended)
            expected = finger_states(hand)
            for degrees in (37, 90, 173, 251):
                self.assertEqual(finger_states(rotate(hand, degrees)), expected)

    def test_world_landmarks_are_preferred(self):
        closed = make_hand(set())
        opened = make_hand({"thumb", "index", "middle", "ring", "pinky"})
        self.assertEqual(finger_states(closed, opened), [1, 1, 1, 1, 1])

    def test_degenerate_world_landmarks_fall_back_to_normalized(self):
        index_only = make_hand({"index"})
        self.assertEqual(finger_states(index_only, [(0.0, 0.0, 0.0)] * 21), [0, 1, 0, 0, 0])

    def test_invalid_landmarks_fail_closed(self):
        self.assertEqual(finger_states(None), [])
        self.assertEqual(finger_states([(0.0, 0.0)] * 20), [])
        self.assertEqual(finger_states([(0.0, 0.0)] * 21), [])
        invalid = make_hand({"index"})
        invalid[8] = (math.inf, 0.0, 0.0)
        self.assertEqual(finger_states(invalid), [])


if __name__ == "__main__":
    unittest.main()
