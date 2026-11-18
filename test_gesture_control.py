import unittest
from types import SimpleNamespace

from gesture_control import GestureController
from main import (
    HEALTH_CHEAT,
    classify_palm,
    draw_activation_message,
    draw_hand_landmarks,
    get_hand_bounds,
)


def hand_landmarks(tip_y: float) -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.8)
    for pip_index, tip_index in ((6, 8), (10, 12), (14, 16), (18, 20)):
        points[pip_index] = SimpleNamespace(x=0.5, y=0.5)
        points[tip_index] = SimpleNamespace(x=0.5, y=tip_y)
    return points


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.rectangles = []
        self.text = []

    def rectangle(self, frame, start, end, color, thickness):
        self.rectangles.append((start, end, color, thickness))

    def putText(self, frame, text, origin, font, scale, color, thickness):
        self.text.append((text, origin, color, scale))

    def line(self, frame, start, end, color, thickness):
        pass

    def circle(self, frame, center, radius, color, thickness):
        pass


class GestureControllerTests(unittest.TestCase):
    def test_health_cheat_is_typed_in_lowercase(self):
        self.assertEqual(HEALTH_CHEAT, "gesundheit")

    def test_detects_a_fist(self):
        self.assertEqual(classify_palm(hand_landmarks(0.65), 640, 480), "closed")

    def test_detects_an_open_palm(self):
        self.assertEqual(classify_palm(hand_landmarks(0.2), 640, 480), "open")

    def test_does_not_misclassify_an_unclear_hand_pose(self):
        self.assertIsNone(classify_palm(hand_landmarks(0.5), 640, 480))

    def test_hand_bounds_include_padding_and_stay_inside_frame(self):
        points = hand_landmarks(0.2)
        points[0] = SimpleNamespace(x=0.25, y=0.25)
        points[1] = SimpleNamespace(x=0.75, y=0.75)

        self.assertEqual(get_hand_bounds(points, 640, 480), (148, 84, 492, 372))
        points[0] = SimpleNamespace(x=-0.2, y=-0.2)
        points[1] = SimpleNamespace(x=1.2, y=1.2)
        self.assertEqual(get_hand_bounds(points, 640, 480), (0, 0, 639, 479))

    def test_draws_green_box_and_label_for_open_and_closed_palms(self):
        frame = SimpleNamespace(shape=(480, 640, 3))

        for state in ("open", "closed"):
            with self.subTest(state=state):
                cv2 = FakeCv2()
                draw_hand_landmarks(cv2, frame, hand_landmarks(0.5), state)

                self.assertEqual(
                    cv2.rectangles[0],
                    ((308, 228), (332, 396), (0, 255, 0), 2),
                )
                self.assertEqual(cv2.text[0][0], f"{state.upper()} PALM DETECTED")
                self.assertGreaterEqual(cv2.text[0][3], 0.85)

    def test_draws_health_cheat_activation_message(self):
        cv2 = FakeCv2()
        frame = SimpleNamespace(shape=(480, 640, 3))

        draw_activation_message(cv2, frame)

        self.assertEqual(
            [message[0] for message in cv2.text],
            ["Activating Health Cheat", "Closed Palm Detected!"],
        )

    def test_requires_open_hand_before_arming(self):
        controller = GestureController(debounce_seconds=0.2, cooldown_seconds=1.0)

        self.assertFalse(controller.update(True, now=0.0))
        self.assertFalse(controller.update(True, now=0.3))
        self.assertTrue(controller.stable_closed)

    def test_fires_once_for_a_debounced_fist_transition(self):
        controller = GestureController(debounce_seconds=0.2, cooldown_seconds=1.0)

        self.assertFalse(controller.update(False, now=0.0))
        self.assertFalse(controller.update(False, now=0.21))
        self.assertFalse(controller.update(True, now=0.3))
        self.assertTrue(controller.update(True, now=0.51))
        self.assertFalse(controller.update(True, now=2.0))

    def test_requires_reopening_even_after_cooldown(self):
        controller = GestureController(debounce_seconds=0.1, cooldown_seconds=1.0)

        controller.update(False, now=0.0)
        controller.update(False, now=0.11)
        controller.update(True, now=0.2)
        self.assertTrue(controller.update(True, now=0.31))
        self.assertFalse(controller.update(True, now=5.0))

    def test_cooldown_blocks_a_new_transition(self):
        controller = GestureController(debounce_seconds=0.1, cooldown_seconds=2.0)

        controller.update(False, now=0.0)
        controller.update(False, now=0.11)
        controller.update(True, now=0.2)
        self.assertTrue(controller.update(True, now=0.31))
        controller.update(False, now=0.4)
        controller.update(False, now=0.51)
        controller.update(True, now=0.6)
        self.assertFalse(controller.update(True, now=0.71))
        self.assertAlmostEqual(controller.cooldown_remaining(now=0.71), 1.6)

    def test_missing_hand_cancels_pending_transition(self):
        controller = GestureController(debounce_seconds=0.2, cooldown_seconds=1.0)

        controller.update(False, now=0.0)
        controller.update(False, now=0.21)
        controller.update(True, now=0.3)
        self.assertFalse(controller.update(None, now=0.4))
        self.assertFalse(controller.update(True, now=0.5))
        self.assertTrue(controller.update(True, now=0.8))

    def test_rejects_negative_timing_configuration(self):
        with self.assertRaises(ValueError):
            GestureController(debounce_seconds=-0.1)
        with self.assertRaises(ValueError):
            GestureController(cooldown_seconds=-0.1)


if __name__ == "__main__":
    unittest.main()
