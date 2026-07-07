import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def most_frequent(items):
    return max(set(items), key=items.count)


def get_most_frequent_emotion(output):
    emotions = []
    for frame_data in output.values():
        for face_data in frame_data.values():
            emotions.append(face_data["emotion"])
    return most_frequent(emotions) if emotions else None


def validate_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def extract_video_frames(video_path, frames_dir):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_paths = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_path = frames_dir / f"frame_{frame_index:010d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_index += 1

    capture.release()

    if not frame_paths:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frame_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Detect emotions from an image or video.")
    parser.add_argument("input_path", help="Path to an image or video file.")
    return parser


def process(input_path):
    input_path = Path(input_path).resolve()
    validate_file(input_path, "Input file")

    detection_model_path = TRAINED_MODELS_DIR / "detection_models" / "haarcascade_frontalface_default.xml"
    emotion_model_path = TRAINED_MODELS_DIR / "emotion_models" / "fer2013_mini_XCEPTION.102-0.66.hdf5"
    validate_file(detection_model_path, "Face detection model")
    validate_file(emotion_model_path, "Emotion classification model")

    emotion_labels = get_labels("fer2013")
    emotion_offsets = (0, 0)

    face_detection = load_detection_model(str(detection_model_path))
    emotion_classifier = load_model(str(emotion_model_path), compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    frames_dir = None
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        images_list = [input_path]
    else:
        frames_dir = Path(tempfile.mkdtemp(prefix="emotion_frames_"))
        images_list = extract_video_frames(input_path, frames_dir)

    output = {}
    try:
        for frame_index, image_path in enumerate(images_list):
            gray_image = load_image(str(image_path), grayscale=True)
            gray_image = np.squeeze(gray_image).astype("uint8")
            faces = detect_faces(face_detection, gray_image)

            frame_output = {}
            for face_index, face_coordinates in enumerate(faces):
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]

                try:
                    gray_face = cv2.resize(gray_face, emotion_target_size)
                except cv2.error:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)

                prediction = emotion_classifier.predict(gray_face, verbose=0)[0]
                emotion_label_arg = int(np.argmax(prediction))
                x, y, w, h = [int(value) for value in face_coordinates]
                frame_output[face_index] = {
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "emotion": emotion_labels[emotion_label_arg],
                    "score": float(np.max(prediction)),
                }

            output[frame_index] = frame_output
    finally:
        if frames_dir and frames_dir.exists():
            shutil.rmtree(frames_dir)

    return output, get_most_frequent_emotion(output)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        output, most_frequent_emotion = process(args.input_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for key, value in output.items():
        print({key: value})
    print(most_frequent_emotion)
