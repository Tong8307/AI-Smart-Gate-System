"""
AI Smart Gate System
Emotion Engine

Optional, separate from face_engine.py on purpose: InsightFace (ArcFace /
buffalo_l) only does detection, landmarks, recognition and gender/age — it
has no emotion model. This module bolts on a small standalone ONNX
classifier (FER+ by default) and runs it on a face crop that's already
been detected elsewhere, usually by face_engine.detect_faces().

This module does NOT do its own face detection. Call it with a frame and
a bbox you already have (e.g. from an InsightFace `face.bbox`) — that
keeps this file simple and avoids running detection twice per frame.

Usage:
    import emotion_engine as ee
    result = ee.detect_emotion(frame, face.bbox)
    # -> {"emotion": "happiness", "confidence": 0.83, "scores": {...}} or None

Setup:
    1. Download an emotion ONNX model, e.g. FER+ from the ONNX Model Zoo:
       https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus
    2. Save it as models/emotion-ferplus.onnx relative to this file (or
       point EMOTION_MODEL_PATH at wherever you keep it).
    3. pip install onnxruntime (already a dependency of face_engine.py,
       so if the face recognition side works, this will too).

Accuracy note: FER+ was trained on posed, front-facing, well-lit faces.
Real gate-camera footage (angle, motion blur, low light) will be noisier
than that, so treat this as a rough signal, not a precise reading.
"""

import os
import threading

import cv2
import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point this at your .onnx file. Override with EMOTION_MODEL_PATH if you
# keep models somewhere else.
EMOTION_MODEL_PATH = os.environ.get(
    "EMOTION_MODEL_PATH",
    os.path.join(_PROJECT_DIR, "models", "emotion-ferplus.onnx"),
)

# FER+ output order (8 classes). If you swap in a different model, update
# this to match its label order.
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

# Crop is padded a bit around the face bbox before resizing, so the model
# sees a bit of forehead/chin/jaw instead of a tight face-only box —
# FER+ was trained on crops with some margin, not just the raw bbox.
CROP_PADDING = 0.15
INPUT_SIZE = 64  # FER+ expects a 64x64 grayscale input

_session = None
_session_lock = threading.Lock()


def get_emotion_model():
    """Lazily loads the ONNX emotion model once and reuses it, same
    pattern as face_engine.get_app() for the face model."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                if not os.path.exists(EMOTION_MODEL_PATH):
                    raise FileNotFoundError(
                        f"Emotion model not found at {EMOTION_MODEL_PATH}. "
                        f"Download a FER+ .onnx file and place it there, or "
                        f"set the EMOTION_MODEL_PATH environment variable."
                    )
                import onnxruntime as ort
                _session = ort.InferenceSession(
                    EMOTION_MODEL_PATH,
                    providers=["CPUExecutionProvider"],
                )
    return _session


def is_available():
    """Cheap check for callers that want to skip emotion detection
    entirely when no model file has been set up yet, instead of hitting
    the FileNotFoundError from get_emotion_model()."""
    return os.path.exists(EMOTION_MODEL_PATH)


def _crop_face(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * CROP_PADDING
    x2 += bw * CROP_PADDING
    y1 -= bh * CROP_PADDING
    y2 += bh * CROP_PADDING
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def detect_emotion(frame, bbox):
    """
    frame: BGR numpy array (a full camera frame, e.g. from cv2/InsightFace).
    bbox: [x1, y1, x2, y2] face box, e.g. face.bbox from an InsightFace
          Face object (face_engine.detect_faces()).

    Returns {"emotion": str, "confidence": float, "scores": {label: prob}}
    or None if there was no usable face crop, or if the model file isn't
    set up (use is_available() to check that ahead of time and avoid the
    exception path in a hot loop).
    """
    if not is_available():
        return None

    crop = _crop_face(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    inp = resized.reshape(1, 1, INPUT_SIZE, INPUT_SIZE)

    sess = get_emotion_model()
    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: inp})[0][0]

    # softmax
    exp = np.exp(raw - np.max(raw))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    scores = {label: round(float(p), 3) for label, p in zip(EMOTION_LABELS, probs)}

    return {
        "emotion": EMOTION_LABELS[idx],
        "confidence": round(float(probs[idx]), 3),
        "scores": scores,
    }
