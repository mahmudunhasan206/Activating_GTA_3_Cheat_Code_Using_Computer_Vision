import argparse
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

from gesture_control import GestureController


HEALTH_CHEAT = "gesundheit"
ACTIVATION_MESSAGE_LINES = (
    "Activating Health Cheat",
    "Closed Palm Detected!",
)
ACTIVATION_MESSAGE_DURATION = 2.5
UI_FONT_SCALE = 0.9
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandLandmark(Protocol):
    x: float
    y: float


def classify_palm(
    hand_landmarks: Sequence[HandLandmark],
    frame_width: int,
    frame_height: int,
) -> str | None:
    """Return open or closed only when at least three fingers agree."""
    points = hand_landmarks
    wrist = (points[0].x * frame_width, points[0].y * frame_height)
    finger_pairs = ((6, 8), (10, 12), (14, 16), (18, 20))
    curled_fingers = 0
    extended_fingers = 0

    for pip_index, tip_index in finger_pairs:
        pip = (points[pip_index].x * frame_width, points[pip_index].y * frame_height)
        tip = (points[tip_index].x * frame_width, points[tip_index].y * frame_height)
        pip_distance = math.dist(wrist, pip)
        tip_distance = math.dist(wrist, tip)
        if tip_distance < pip_distance * 0.95:
            curled_fingers += 1
        elif tip_distance > pip_distance * 1.05:
            extended_fingers += 1

    if curled_fingers >= 3:
        return "closed"
    if extended_fingers >= 3:
        return "open"
    return None


def get_hand_bounds(
    hand_landmarks: Sequence[HandLandmark],
    frame_width: int,
    frame_height: int,
    padding: int = 12,
) -> tuple[int, int, int, int]:
    x_values = [int(point.x * frame_width) for point in hand_landmarks]
    y_values = [int(point.y * frame_height) for point in hand_landmarks]
    return (
        max(0, min(frame_width - 1, min(x_values) - padding)),
        max(0, min(frame_height - 1, min(y_values) - padding)),
        max(0, min(frame_width - 1, max(x_values) + padding)),
        max(0, min(frame_height - 1, max(y_values) + padding)),
    )


def blur_background(cv2, np, frame, hand_landmarks: Sequence[HandLandmark] | None):
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=25)
    if hand_landmarks is None:
        return blurred

    height, width = frame.shape[:2]
    points = np.array(
        [
            [int(point.x * width), int(point.y * height)]
            for point in hand_landmarks
        ],
        dtype=np.int32,
    )
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=12)
    alpha = mask[:, :, None].astype(np.float32) / 255.0
    sharp_frame = frame.astype(np.float32)
    blurred_frame = blurred.astype(np.float32)
    return (sharp_frame * alpha + blurred_frame * (1.0 - alpha)).astype(np.uint8)


def get_model_path() -> Path:
    """Download and cache the MediaPipe hand model on its first run."""
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    model_path = cache_root / "GTA3HandGesture" / "hand_landmarker.task"
    if model_path.is_file() and model_path.stat().st_size > 1_000_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    print("Downloading the hand-tracking model (first run only)...")
    try:
        request = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "GTA3HandGesture/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as model_file:
                shutil.copyfileobj(response, model_file)
        if temporary_path.stat().st_size <= 1_000_000:
            raise OSError("the downloaded hand-tracking model is incomplete")
        temporary_path.replace(model_path)
    except (OSError, urllib.error.URLError):
        temporary_path.unlink(missing_ok=True)
        raise

    return model_path


def draw_hand_landmarks(
    cv2,
    frame,
    hand_landmarks: Sequence[HandLandmark],
    palm_state: str | None,
) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]
    left, top, right, bottom = get_hand_bounds(hand_landmarks, width, height)
    detected_palm = palm_state is not None
    color = (0, 255, 0) if detected_palm else (0, 165, 255)
    label = (
        f"{palm_state.upper()} PALM DETECTED"
        if detected_palm
        else "HAND DETECTED"
    )
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(
        frame,
        label,
        (left, max(20, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        UI_FONT_SCALE,
        color,
        2,
    )
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def draw_activation_message(cv2, frame) -> None:
    for line_number, message in enumerate(ACTIVATION_MESSAGE_LINES):
        cv2.putText(
            frame,
            message,
            (12, 65 + line_number * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            UI_FONT_SCALE,
            (0, 255, 0),
            2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Type the GTA III health cheat when a closed fist is detected."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.3,
        help="seconds a hand state must remain stable (default: 0.3)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="minimum seconds between cheat activations (default: 3)",
    )
    parser.add_argument(
        "--key-interval",
        type=float,
        default=0.08,
        help="delay between typed cheat characters (default: 0.08)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.key_interval < 0:
        print("Error: --key-interval must not be negative.", file=sys.stderr)
        return 2
    try:
        controller = GestureController(args.debounce, args.cooldown)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import pyautogui
    except ImportError as exc:
        print(
            "Missing application dependency. Install them with "
            "`python -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            "A native application dependency could not be loaded. Reinstall "
            "the requirements using the same Python interpreter used to run "
            "this app.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    try:
        model_path = get_model_path()
    except (OSError, urllib.error.URLError) as exc:
        print(f"Error downloading the hand-tracking model: {exc}", file=sys.stderr)
        return 1

    pyautogui.FAILSAFE = True
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(
            f"Error: could not open camera {args.camera}. "
            "Check the camera index and camera permissions.",
            file=sys.stderr,
        )
        return 1

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    try:
        activation_message_until = 0.0
        with vision.HandLandmarker.create_from_options(options) as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    print("Error: the camera stopped producing frames.", file=sys.stderr)
                    return 1

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                results = hands.detect(image)
                hand_landmarks = (
                    results.hand_landmarks[0]
                    if results.hand_landmarks
                    else None
                )
                palm_state = (
                    classify_palm(hand_landmarks, width, height)
                    if hand_landmarks is not None
                    else None
                )
                observed_closed = (
                    palm_state == "closed" if palm_state is not None else None
                )

                if controller.update(observed_closed):
                    activation_message_until = (
                        time.monotonic() + ACTIVATION_MESSAGE_DURATION
                    )
                    print("Activating Health Cheat - Closed Palm Detected!")
                    pyautogui.write(HEALTH_CHEAT, interval=args.key_interval)

                frame = blur_background(cv2, np, frame, hand_landmarks)
                if hand_landmarks is not None:
                    draw_hand_landmarks(cv2, frame, hand_landmarks, palm_state)

                if controller.stable_closed is None:
                    status = "Show an open hand to arm"
                elif controller.stable_closed:
                    status = "Fist detected - open hand to re-arm"
                else:
                    status = "Armed - close your hand to activate"

                remaining = controller.cooldown_remaining()
                if remaining > 0:
                    status += f" | Cooldown: {remaining:.1f}s"
                cv2.putText(
                    frame,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    UI_FONT_SCALE,
                    (0, 255, 0) if not controller.stable_closed else (0, 165, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    UI_FONT_SCALE,
                    (255, 255, 255),
                    1,
                )
                if time.monotonic() < activation_message_until:
                    draw_activation_message(cv2, frame)
                cv2.imshow("GTA III Hand Gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except pyautogui.FailSafeException:
        print("Stopped by PyAutoGUI failsafe.", file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
