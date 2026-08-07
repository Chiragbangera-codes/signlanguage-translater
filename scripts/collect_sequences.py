"""Collect MediaPipe hand-landmark sequences for one digit label.

Each saved sample contains 30 frames with 127 values per frame:
uses_two_hands, left-hand landmarks (63), and right-hand landmarks (63).
"""

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 127


def open_camera(camera_index: int, source: str | None = None) -> cv2.VideoCapture:
    """Opens a camera with DirectShow on Windows, which supports virtual webcams."""
    if source:
        return cv2.VideoCapture(source)
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    return cv2.VideoCapture(camera_index, backend)


def list_available_cameras(max_index: int) -> None:
    """Prints camera indexes that can open and return a video frame."""
    print("Checking available cameras...")
    found = []
    for camera_index in range(max_index + 1):
        camera = open_camera(camera_index)
        opened, _ = camera.read() if camera.isOpened() else (False, None)
        camera.release()
        if opened:
            found.append(camera_index)
            print(f"  Camera {camera_index}: available")

    if not found:
        print("No camera returned a frame. Connect and start your phone webcam app, then try again.")


def landmarks_to_frame(results) -> np.ndarray | None:
    """Converts one MediaPipe result into the feature order used by the app."""
    landmarks_list = results.multi_hand_landmarks or []
    handedness_list = results.multi_handedness or []
    if not landmarks_list:
        return None

    left_hand = np.full(63, -1.0, dtype=np.float32)
    right_hand = np.full(63, -1.0, dtype=np.float32)
    uses_two_hands = 1.0 if len(landmarks_list) >= 2 else 0.0

    if len(landmarks_list) == 1:
        left_hand = np.asarray(
            [value for landmark in landmarks_list[0].landmark for value in (landmark.x, landmark.y, landmark.z)],
            dtype=np.float32,
        )
    else:
        for landmarks, handedness in zip(landmarks_list[:2], handedness_list[:2]):
            coords = np.asarray(
                [value for landmark in landmarks.landmark for value in (landmark.x, landmark.y, landmark.z)],
                dtype=np.float32,
            )
            label = handedness.classification[0].label
            # Match the non-mirrored MediaPipe mapping used by CameraCard.tsx.
            if label == "Right":
                left_hand = coords
            elif label == "Left":
                right_hand = coords

    return np.concatenate(([uses_two_hands], left_hand, right_hand)).reshape(FEATURES_PER_FRAME)


def next_sequence_path(label_dir: Path) -> Path:
    existing_numbers = []
    for path in label_dir.glob("sequence_*.npy"):
        try:
            existing_numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return label_dir / f"sequence_{max(existing_numbers, default=0) + 1:03d}.npy"


def collect(label: str, samples: int, dataset_dir: Path, camera_index: int, source: str | None = None) -> None:
    label_dir = dataset_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    camera = open_camera(camera_index, source)
    if not camera.isOpened():
        camera_name = source or f"camera {camera_index}"
        raise RuntimeError(f"Could not open {camera_name}.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    collected = 0
    recording = False
    sequence: list[np.ndarray] = []
    print(f"Press SPACE to start collecting {samples} samples, then they are collected continuously. Pausing: SPACE, discard current sample: R, quit: Q.")

    try:
        while collected < samples:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feature_frame = landmarks_to_frame(results)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            if recording and feature_frame is not None:
                sequence.append(feature_frame)
                if len(sequence) == FRAMES_PER_SEQUENCE:
                    output_path = next_sequence_path(label_dir)
                    np.save(output_path, np.stack(sequence).astype(np.float32))
                    collected += 1
                    sequence = []
                    print(f"Saved {output_path} ({collected}/{samples})")

            display = cv2.flip(frame, 1)
            if recording:
                state = "RECORDING"
            elif collected == 0:
                state = "Press SPACE to start"
            else:
                state = "PAUSED"
            status = f"Label {label} | sample {collected + 1}/{samples} | {state}"
            cv2.putText(display, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            if recording:
                cv2.putText(
                    display,
                    f"Frames: {len(sequence)}/{FRAMES_PER_SEQUENCE}",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("SignSpeak dataset collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                sequence = []
            if key == ord(" "):
                if not recording:
                    sequence = []
                recording = not recording
    finally:
        hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect 30-frame MediaPipe gesture sequences for one digit.")
    parser.add_argument("--label", choices=[str(number) for number in range(10)])
    parser.add_argument("--samples", type=int, default=50, help="Number of sequences to collect (default: 50).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--source",
        help="Optional video URL, for example DroidCam's http://PHONE_IP:4747/video stream.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List usable camera indexes and exit.")
    parser.add_argument("--max-camera-index", type=int, default=5, help="Highest index to check with --list-cameras.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "digits",
    )
    args = parser.parse_args()

    if args.list_cameras:
        list_available_cameras(args.max_camera_index)
        raise SystemExit(0)
    if args.label is None:
        parser.error("--label is required unless --list-cameras is used.")
    if args.samples < 1:
        parser.error("--samples must be at least 1.")
    collect(args.label, args.samples, args.dataset_dir, args.camera, args.source)
