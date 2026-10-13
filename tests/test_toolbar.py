import unittest

import numpy as np

from toolbar import DrawingToolbar, ToolbarAction


class DrawingToolbarTests(unittest.TestCase):
    def setUp(self):
        self.toolbar = DrawingToolbar((0, 0, 255), 8, dwell_ms=450)
        self.width = 1280

    def _center(self, key):
        button = next(item for item in self.toolbar.buttons(self.width) if item.key == key)
        return (
            button.rect.x + button.rect.width // 2,
            button.rect.y + button.rect.height // 2,
        )

    def test_mouse_activation_changes_color_and_size(self):
        color_point = self._center("color_1")
        size_point = self._center("size_2")

        self.assertEqual(
            self.toolbar.activate(color_point, self.width), ToolbarAction.STYLE_CHANGED
        )
        self.assertEqual(self.toolbar.selected_color, self.toolbar.COLORS[1][1])
        self.assertEqual(self.toolbar.activate(size_point, self.width), ToolbarAction.STYLE_CHANGED)
        self.assertEqual(self.toolbar.brush_thickness, 14)

    def test_clear_button_returns_clear_action(self):
        self.assertEqual(
            self.toolbar.activate(self._center("clear"), self.width), ToolbarAction.CLEAR
        )

    def test_hand_selection_requires_continuous_dwell(self):
        point = self._center("color_2")
        self.assertIsNone(self.toolbar.update_pointer(point, True, 1000, self.width))
        self.assertIsNone(self.toolbar.update_pointer(point, True, 1449, self.width))
        self.assertEqual(
            self.toolbar.update_pointer(point, True, 1450, self.width),
            ToolbarAction.STYLE_CHANGED,
        )
        self.assertIsNone(self.toolbar.update_pointer(point, True, 1600, self.width))

        self.toolbar.update_pointer(None, False, 1700, self.width)
        self.assertIsNone(self.toolbar.update_pointer(point, True, 1800, self.width))

    def test_toolbar_region_and_rendering(self):
        self.assertTrue(self.toolbar.contains((10, 10)))
        self.assertFalse(self.toolbar.contains((10, 200)))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.toolbar.render(frame, "draw", 1000)
        self.assertTrue(np.any(frame[: self.toolbar.height] != 0))
        self.assertTrue(np.all(frame[self.toolbar.height + 10 :] == 0))


if __name__ == "__main__":
    unittest.main()
