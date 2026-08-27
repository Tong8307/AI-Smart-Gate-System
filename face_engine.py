"""
AI Smart Gate System
Face Engine

Shared logic used by the Flask app (app.py) for:
- Loading the ArcFace / InsightFace model once (singleton)
- Loading & saving the face database (supports many registered users)
- Extracting embeddings from an uploaded photo or a camera frame
- Comparing a probe embedding against the database
- Grabbing a single frame from a local camera

Kept separate from app.py so register.py / recognize.py (or a CLI)
could still import the same logic if you want them later.

The anti-tailgating "how many people are in frame" check is YOLO-based
and lives in person_detector.py instead of here — see count_persons()
below, which is just a thin pass-through to that module.
"""

import os
import pickle
import threading
import time
import uuid
from datetime import datetime

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

import person_detector
import emotion_engine as ee

# Always resolve relative to this file's own folder (not the terminal's
# current directory), so the database is found in the same place no
# matter where you launch the app from.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("FACE_DB_PATH", os.path.join(_PROJECT_DIR, "face_database.pkl"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
# One physical camera is used for the prototype. Entry, Exit and CCTV are
# logical modes selected by the dashboard, not separate physical cameras.
CAMERA_INDEX_ENTRY = CAMERA_INDEX
CAMERA_INDEX_EXIT = CAMERA_INDEX
CAMERA_INDEX_CCTV = int(os.environ.get("CAMERA_INDEX_CCTV", "1"))
# ctx_id=-1 -> CPU, ctx_id=0 -> first GPU. Default to CPU so this runs
# out of the box; set FACE_CTX_ID=0 in the environment if you have a GPU.
CTX_ID = int(os.environ.get("FACE_CTX_ID", "-1"))
# Detector input size. 640x640 is InsightFace's accurate default but is
# slow on CPU. Drop it to e.g. 320x320 (set FACE_DET_SIZE=320) for a big
# CPU speedup at a small accuracy cost — plenty for a single face up close
# at a gate.
_det_size = int(os.environ.get("FACE_DET_SIZE", "320"))
DET_SIZE = (_det_size, _det_size)
DEFAULT_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.60"))
# Used for the "is this still the same person?" check across the 3
# registration photos (see same_person below) — deliberately a bit looser
# than DEFAULT_THRESHOLD. DEFAULT_THRESHOLD is tuned for the gate's own
# angle/distance recognition; here we're only checking a few shots taken
# seconds apart, but at 3 different distances (normal/far/close) on
# purpose, so cosine similarity naturally runs a little lower than a
# same-distance comparison would.
SAME_PERSON_THRESHOLD = float(os.environ.get("SAME_PERSON_THRESHOLD", "0.50"))
# Camera capture resolution. Smaller frames = less work per detection pass
# AND a lighter MJPEG stream. 0 leaves the camera's default resolution.
CAM_WIDTH = int(os.environ.get("CAM_WIDTH", "640"))
CAM_HEIGHT = int(os.environ.get("CAM_HEIGHT", "480"))

# Where captured "evidence" photos go (unknown/unauthorized faces caught by
# the gate or CCTV). Served by Flask's default /static route since this
# folder lives at <project>/static/snapshots.
SNAPSHOT_DIR = os.path.join(_PROJECT_DIR, "static", "snapshots")

_app = None
_app_lock = threading.Lock()
# RLock (not Lock) because add_user/set_user_active/delete_user call
# load_database() while already holding this lock, and load_database()
# itself may need to save (see the migration step below).
_db_lock = threading.RLock()


def get_app():
    """Lazily load the InsightFace model once and reuse it for every request."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis

                available = ort.get_available_providers()
                wants_gpu = CTX_ID >= 0
                has_cuda = "CUDAExecutionProvider" in available

                fa = FaceAnalysis(name="buffalo_l")
                fa.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)
                _app = fa

                # Loud, unmissable startup line so you don't have to guess.
                if wants_gpu and has_cuda:
                    print(f"[face_engine] Running on GPU (ctx_id={CTX_ID}). "
                          f"Providers available: {available}")
                elif wants_gpu and not has_cuda:
                    print(f"[face_engine] WARNING: FACE_CTX_ID={CTX_ID} requested GPU, "
                          f"but CUDAExecutionProvider is NOT available. Falling back to "
                          f"CPU behavior. Providers available: {available}. "
                          f"Check your onnxruntime-gpu / CUDA / cuDNN install.")
                else:
                    print(f"[face_engine] Running on CPU (ctx_id={CTX_ID}). "
                          f"Providers available: {available}")
    return _app


def get_gpu_status():
    """Reports what's configured vs. what's actually available, for the /api/gpu-status route."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    wants_gpu = CTX_ID >= 0
    has_cuda = "CUDAExecutionProvider" in available
    actually_on_gpu = wants_gpu and has_cuda

    return {
        "ctx_id": CTX_ID,
        "requested_gpu": wants_gpu,
        "cuda_provider_available": has_cuda,
        "available_providers": available,
        "actually_running_on_gpu": actually_on_gpu,
        "det_size": DET_SIZE,
        "model_loaded": _app is not None,
    }


def count_persons(frame):
    """Returns (count, boxes) — the number of *people* (full bodies, not
    just faces) visible in `frame`. Thin pass-through to
    person_detector.count_persons(), supplying this module's own face
    detector as the fallback for when YOLO/ultralytics isn't installed.
    Used by the main gate to refuse access when more than one person is
    presenting to the camera at once (anti-tailgating)."""
    return person_detector.count_persons(frame, fallback_faces=lambda: detect_faces(frame))


# -----------------------------
# Database helpers
# -----------------------------

def load_database(path=DB_PATH):
    """
    Returns a list of records:
        {"id": str, "name": str, "role": str, "embeddings": [np.ndarray, ...],
         "registered_at": iso str, "active": bool}

    A person can have MULTIPLE embeddings (one per registration photo —
    see add_user/register()), which makes matching far more robust to
    lighting, angle, glasses, etc. than a single reference photo: at
    match time we compare the probe face against every stored embedding
    for that person and take their best score (see match_embedding).

    Transparently upgrades older formats so existing databases keep
    working after this change:
      - the very old single-user dict format from the original register.py
      - the later list-of-records format that stored one "embedding" per
        person instead of an "embeddings" list
    and backfills "id"/"active" on any record that predates those fields.
    "active" is the access-control flag: set it False (see
    set_user_active) to revoke someone's access without deleting their
    face data, or delete_user() to remove them entirely.
    """
    if not os.path.exists(path):
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "embedding" in data:
        # Very old single-user format -> wrap it in the new list format.
        data = [{
            "name": data.get("name", "Registered User"),
            "role": data.get("role", "Resident"),
            "embeddings": [data["embedding"]],
            "registered_at": datetime.now().isoformat(timespec="seconds"),
        }]

    if not isinstance(data, list):
        return []

    changed = False
    for rec in data:
        if "id" not in rec:
            rec["id"] = uuid.uuid4().hex
            changed = True
        if "active" not in rec:
            rec["active"] = True
            changed = True
        if "embeddings" not in rec:
            # One-photo-per-person records from before multi-photo
            # registration was added — migrate their single embedding
            # into the new list-based format.
            old = rec.pop("embedding", None)
            rec["embeddings"] = [old] if old is not None else []
            changed = True
    if changed:
        save_database(data, path)

    return data


def save_database(records, path=DB_PATH):
    with _db_lock:
        with open(path, "wb") as f:
            pickle.dump(records, f)


def add_user(name, role, embeddings, path=DB_PATH):
    """Adds a new person and returns their full record (including the new
    "id", which the caller needs to reference this person later — e.g. to
    revoke or delete their access).

    `embeddings` is a list of one or more embeddings — one per accepted
    registration photo (see register() in app.py, which asks for up to 3
    photos: front-facing, slight left turn, slight right turn). Storing
    several angles per person means a probe face only needs to be close
    to ONE of them to match, instead of forcing every future gate photo
    to resemble a single reference shot."""
    if not isinstance(embeddings, (list, tuple)):
        embeddings = [embeddings]
    with _db_lock:
        records = load_database(path)
        record = {
            "id": uuid.uuid4().hex,
            "name": name,
            "role": role,
            "embeddings": list(embeddings),
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "active": True,
        }
        records.append(record)
        with open(path, "wb") as f:
            pickle.dump(records, f)
        return record


def list_users(path=DB_PATH):
    """Dashboard-safe view of the database (no embeddings)."""
    return [
        {
            "id": r["id"],
            "name": r.get("name", "Unknown"),
            "role": r.get("role", "Resident"),
            "registered_at": r.get("registered_at", ""),
            "active": r.get("active", True),
        }
        for r in load_database(path)
    ]


def get_user(user_id, path=DB_PATH):
    for r in load_database(path):
        if r.get("id") == user_id:
            return r
    return None


def set_user_active(user_id, active, path=DB_PATH):
    """Enables/revokes a person's access without touching their stored
    face data. A revoked (active=False) person is excluded from matching
    by match_embedding/match_all_faces below, so they simply stop being
    recognized at the gate and on CCTV — showing up as 'Unknown' — until
    re-enabled. Returns the updated record, or None if no such user."""
    with _db_lock:
        records = load_database(path)
        updated = None
        for r in records:
            if r.get("id") == user_id:
                r["active"] = bool(active)
                updated = r
                break
        if updated is not None:
            with open(path, "wb") as f:
                pickle.dump(records, f)
        return updated


def delete_user(user_id, path=DB_PATH):
    """Permanently removes a person (and their face data). Returns True
    if someone was deleted, False if the id wasn't found."""
    with _db_lock:
        records = load_database(path)
        remaining = [r for r in records if r.get("id") != user_id]
        deleted = len(remaining) != len(records)
        if deleted:
            with open(path, "wb") as f:
                pickle.dump(remaining, f)
        return deleted


# -----------------------------
# Face detection / embedding
# -----------------------------

def largest_face(faces):
    """Pick the biggest detected face (closest to camera) when several are found."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _safe_detect_emotion(frame, bbox):
    """Best-effort emotion read on a face crop (see emotion_engine.py).
    This is a logging add-on, not part of the access decision, so any
    failure here (model file missing, bad crop, etc.) is swallowed —
    it must never be able to take down face recognition or gate access.
    Returns emotion_engine's result dict, or None."""
    if frame is None:
        return None
    try:
        return ee.detect_emotion(frame, bbox)
    except Exception:
        return None


def detect_faces(image):
    """image: BGR numpy array (as read by cv2). Returns list of InsightFace Face objects."""
    return get_app().get(image)


def decode_image_from_bytes(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# -----------------------------
# Registration photo quality checks
# -----------------------------
# Used by both registration paths in the dashboard:
#   - Live Capture: polled continuously while the user positions
#     themselves, to drive the "too far / too close / correct" guide.
#   - Upload Image: run once per uploaded file so only photos that pass
#     all four checks are accepted into the 3-photo set.
# Neither path stores a rejected photo — this is pure validation, the
# frame/file itself is discarded either way at this stage.

MIN_FACE_RATIO = 0.12    # face bbox height / image height, below this = "too small / far"
MIN_SHARPNESS = 60.0     # Laplacian variance; below this = "too blurry"

# --- Upload-Image-only checks -------------------------------------------
# These run ONLY when assess_photo_quality is called with extra_checks=True
# — used exclusively by the Upload Image registration path (see app.py).
# Live Capture already gets a real-time framing guide and a same_person
# cross-check across its 3 shots, which a file picked from disk doesn't
# have, so uploaded files get a couple of extra checks a live camera
# frame doesn't need (grayscale, front-facing angle).
#
# The exception is face_features_visible() (check 7, masked/covered
# face) below: that one is NOT upload-only. It also runs for Live
# Capture via check_feature_visibility=True, since a real-time framing
# guide and a same_person check don't catch someone covering their
# nose/mouth — see assess_photo_quality's docstring. Live Capture uses
# a looser threshold (LIVE_FEATURE_MAX_COLOR_DIFF) than Upload Image
# does, since webcam lighting/compression pushes the underlying color
# diff higher even on a normal, uncovered face.
MIN_DET_SCORE = 0.55          # InsightFace detection confidence for the chosen face — a coarse
                               # "is this even a clear face" floor.
GRAYSCALE_CHANNEL_DIFF = 8.0  # mean abs difference between B/G/R channels; real color photos
                               # run well above this, black & white (or desaturated) ones don't.
DUPLICATE_HASH_DISTANCE = 5   # average-hash Hamming distance at/below this = "same picture",
                               # used to stop the same uploaded file being reused across the 3 slots.
MAX_FACE_YAW = 25.0            # degrees; how far the head may turn left/right and still count as
                                # "front-facing". Skipped gracefully if the loaded model doesn't
                                # expose pose (older insightface builds / lighter model packs).
FEATURE_MAX_COLOR_DIFF = float(os.environ.get("FACE_UPLOAD_FEATURE_COLOR_DIFF", "80.0"))
                                # see face_features_visible() below — how far the nose/mouth/chin
                                # region's average color may drift from the forehead's average
                                # skin tone before it's treated as covered/obscured. Used for the
                                # Upload Image path. Originally 35.0, but that was too strict for
                                # real-world photos with strong directional lighting — e.g. an
                                # outdoor/sunlit shot with a bright specular highlight on the
                                # forehead (sweat, gel, direct sun) next to a shadowed nose/mouth/
                                # chin produces a big color gap on its own, with nothing covering
                                # the face. Tune with FACE_UPLOAD_FEATURE_COLOR_DIFF.
LIVE_FEATURE_MAX_COLOR_DIFF = float(os.environ.get("FACE_LIVE_FEATURE_COLOR_DIFF", "100.0"))
                                # Looser version of the same threshold for Live Capture. A live
                                # webcam frame has more uneven lighting, its own white-balance/
                                # compression behavior, and things like facial hair, glasses shadow,
                                # or a slight downward camera angle all naturally widen this color
                                # gap even with nothing covering the face. Set well above the Upload
                                # Image threshold so it only catches an actually-covered nose/mouth
                                # (mask, hand, etc.) rather than ordinary lighting/skin variation —
                                # tune with FACE_LIVE_FEATURE_COLOR_DIFF if it's still too sensitive
                                # (raise it) or too loose (lower it) for your camera/lighting setup.

# Target face-height-ratio bands for each of the 3 live-capture slots.
# All three remain strictly frontal — only the target distance (and so
# the expected relative face size) changes between them, which gives
# the registered profile embeddings that better match how a gate camera
# will actually see the person up close, at arm's length, and further
# back. Tuned for a typical laptop/desk webcam framing; adjust if your
# camera's field of view differs a lot.
CAPTURE_SLOTS = {
    "normal": {"min_ratio": 0.28, "max_ratio": 0.42, "label": "Front / Normal"},
    "far":    {"min_ratio": 0.14, "max_ratio": 0.27, "label": "Front / Far"},
    "close":  {"min_ratio": 0.43, "max_ratio": 0.65, "label": "Front / Close"},
}


def is_grayscale_image(image, threshold=GRAYSCALE_CHANNEL_DIFF):
    """Cheap heuristic for 'this is a black & white (or desaturated) photo'.
    A real color capture has genuine per-pixel differences between its
    B/G/R channels; a grayscale image — even one re-saved as a 3-channel
    JPEG — has all three channels nearly identical everywhere. Sampling
    every 4th pixel keeps this fast on a full-resolution photo without
    losing accuracy for a decision this coarse. Upload-Image-only check."""
    if image.ndim < 3 or image.shape[2] < 3:
        return True
    b, g, r = cv2.split(image[::4, ::4].astype(np.int16))
    diff = (np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3.0
    return diff < threshold


def compute_image_hash(image, hash_size=8):
    """Average hash (aHash): shrink to a small grayscale thumbnail and
    record which pixels are above/below the average brightness. This is
    NOT for face matching (that's what the ArcFace embedding is for) —
    it's a cheap fingerprint of the picture itself, used only to catch
    'the same uploaded file was reused for more than one of the 3 slots'.
    Upload-Image-only check."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def images_are_duplicate(hash_a, hash_b, max_distance=DUPLICATE_HASH_DISTANCE):
    """True if two compute_image_hash() results are close enough to call
    them the same underlying photo (allows for minor re-compression, not
    for an actually different shot). Upload-Image-only check."""
    if hash_a is None or hash_b is None or hash_a.shape != hash_b.shape:
        return False
    return int(np.count_nonzero(hash_a != hash_b)) <= max_distance


def _face_is_frontal(face, max_yaw=MAX_FACE_YAW):
    """True if the face is turned close enough to straight-on. Reads
    InsightFace's `pose` attribute (yaw/pitch/roll in degrees), which
    buffalo_l's 3D landmark model provides. If the loaded model pack
    doesn't expose pose at all, this can't judge angle and returns True
    (fails open rather than blocking registration over a feature the
    model doesn't support). Upload-Image-only check."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return True
    yaw = float(pose[1]) if len(pose) > 1 else 0.0
    return abs(yaw) <= max_yaw


def face_features_visible(image, face, max_color_diff=FEATURE_MAX_COLOR_DIFF):
    """Checks whether the nose/mouth/chin area actually looks like skin,
    by comparing it against a forehead patch from the SAME photo (each
    person's real skin tone, whatever it is, so this isn't biased toward
    any one tone). A mask, hand, heavy scribble, or similar covering that
    area reads as a strong color/brightness anomaly against that
    person's own forehead. This exists because a plain detection-
    confidence check often doesn't drop much when only the lower face is
    covered — the detector can still be confident from eyes/eyebrows/
    hairline alone, so a covered mouth and nose can otherwise sail
    through undetected. Upload-Image-only check.

    Uses InsightFace's 5-point landmarks (kps): left eye, right eye,
    nose, left mouth corner, right mouth corner, in that order.
    Returns True if kps aren't available (fails open, same reasoning as
    _face_is_frontal above).
    """
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 5:
        return True

    h, w = image.shape[:2]
    le, re, nose, ml, mr = [np.array(p, dtype=np.float32) for p in kps[:5]]
    eye_span = float(np.linalg.norm(re - le)) or 1.0

    def _patch_mean(cx, cy, radius):
        x1, y1 = max(int(cx - radius), 0), max(int(cy - radius), 0)
        x2, y2 = min(int(cx + radius), w), min(int(cy + radius), h)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
        return patch.mean(axis=0)

    # Forehead: straight up from the midpoint between the eyes, roughly
    # one eye-span up — comfortably above eyebrows/hair for a typical
    # front-on portrait crop.
    eye_mid = (le + re) / 2.0
    forehead = _patch_mean(eye_mid[0], eye_mid[1] - eye_span, eye_span * 0.4)

    # Nose/mouth/chin: centered between the nose and the mouth-corner
    # midpoint, sized to comfortably cover that whole lower-face region.
    mouth_mid = (ml + mr) / 2.0
    lower_center = (nose + mouth_mid) / 2.0
    lower = _patch_mean(lower_center[0], lower_center[1] + eye_span * 0.15, eye_span * 0.55)

    if forehead is None or lower is None:
        return True

    color_diff = float(np.linalg.norm(forehead - lower))
    return color_diff <= max_color_diff


def compute_sharpness(image):
    """Variance of the Laplacian — a cheap, standard blur estimator.
    Lower values mean a smoother (blurrier / out-of-focus or motion-
    blurred) image; sharp, in-focus images score much higher."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_photo_quality(image, min_face_ratio=MIN_FACE_RATIO, min_sharpness=MIN_SHARPNESS,
                          extra_checks=False, check_feature_visibility=False):
    """Runs the registration checks on one decoded image:
      1. A face is detectable at all.
      2. Exactly one face is in frame (not zero, not several).
      3. The image is clear enough (not too blurry) — skipped if
         min_sharpness=0.
      4. The face is large enough in frame (not too far / too small) —
         skipped if min_face_ratio=0.

    When extra_checks=True (Upload Image path ONLY — see app.py), three
    more checks run:
      5. The photo is in color, not black & white / desaturated.
      6. The face is front-facing (not turned too far to the side).
      7. The nose/mouth/chin area actually looks like visible skin — not
         covered by a mask, hand, hair, heavy scribble/sticker, etc.

    check_feature_visibility=True runs ONLY check 7 (the nose/mouth/chin
    visibility check) on its own, without the other extra_checks, using
    a looser color-diff threshold (LIVE_FEATURE_MAX_COLOR_DIFF) than the
    Upload Image path uses — a live webcam frame's uneven lighting,
    white balance, compression, facial hair, or glasses shadow all
    naturally widen that gap without anything actually covering the
    face, so the stricter Upload threshold produced false rejections
    here. Live Capture sets this: a masked/covered face should still be
    rejected there just like an uploaded photo — a covered nose/mouth
    doesn't reliably drop face-detection confidence on its own (the
    detector can stay confident from eyes/eyebrows/hairline alone), so
    without this check a masked face can otherwise sail through Live
    Capture undetected.

    Live Capture calls this with min_sharpness=0 and min_face_ratio=0 —
    it doesn't want checks 3/4 at all (its own per-slot distance guide
    in app.py/CAPTURE_SLOTS already handles framing far more precisely
    than the blunt min_face_ratio floor, and a blur check isn't wanted
    there), leaving only "exactly one face" + facial-feature-visibility
    as what actually gates whether Live Capture can take the shot.
    face_ratio and sharpness are still computed and returned either way
    (face_ratio in particular is what drives the distance guide), just
    not turned into a rejection reason when their thresholds are 0.

    Returns a dict:
      {ok, reasons: [...], face_count, face_ratio, sharpness, face, image_hash}
    `face` is the accepted InsightFace Face object (bbox + embedding),
    set only when ok is True — the caller can use its .embedding
    directly without re-running detection. `image_hash` is only set when
    ok is True AND extra_checks is True — a compute_image_hash()
    fingerprint of the photo itself, for callers that want to check
    several accepted photos against each other for duplicates (see
    images_are_duplicate).
    """
    h, w = image.shape[:2]
    empty = {"face_count": 0, "face_ratio": None, "sharpness": None, "face": None, "image_hash": None}

    if extra_checks and is_grayscale_image(image):
        return {"ok": False,
                "reasons": ["Photo looks black & white — use a normal color photo, not a filtered/grayscale one."],
                **empty}

    faces = detect_faces(image)
    face_count = len(faces)

    if face_count == 0:
        return {"ok": False, "reasons": ["No face detected — face the camera directly with even lighting."],
                **empty}
    if face_count > 1:
        return {"ok": False, "reasons": [f"{face_count} faces detected — only one person should be in frame."],
                **{**empty, "face_count": face_count}}

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    face_ratio = float((y2 - y1) / h) if h else 0.0
    sharpness = compute_sharpness(image)

    reasons = []
    if min_sharpness > 0 and sharpness < min_sharpness:
        reasons.append("Image is too blurry — hold steady and check the lighting.")
    if min_face_ratio > 0 and face_ratio < min_face_ratio:
        reasons.append("Face is too small in frame — move closer to the camera.")

    if extra_checks:
        if face.det_score < MIN_DET_SCORE:
            reasons.append("Face isn't detected clearly enough — use a clearer, well-lit photo.")
        if not _face_is_frontal(face):
            reasons.append("Face must be front-facing — use a photo where you're looking straight at the camera.")

    if extra_checks and not face_features_visible(image, face, max_color_diff=FEATURE_MAX_COLOR_DIFF):
        reasons.append("Facial features aren't clearly visible — make sure nothing (a mask, hand, hair, etc.) "
                        "is covering your nose or mouth.")
    elif check_feature_visibility and not face_features_visible(image, face, max_color_diff=LIVE_FEATURE_MAX_COLOR_DIFF):
        reasons.append("Facial features aren't clearly visible — make sure nothing (a mask, hand, hair, etc.) "
                        "is covering your nose or mouth.")

    ok = not reasons
    return {
        "ok": ok,
        "reasons": reasons,
        "face_count": 1,
        "face_ratio": round(face_ratio, 3),
        "sharpness": round(sharpness, 1),
        "face": face if ok else None,
        "image_hash": compute_image_hash(image) if (ok and extra_checks) else None,
    }


def classify_capture_distance(face_ratio, slot):
    """Given a measured face-height ratio and which of the 3 live-capture
    slots is currently active ("normal" / "far" / "close"), returns
    "too_far", "too_close", or "correct" — the live guide's real-time
    verdict, computed from the actual detected face size rather than
    the user guessing their distance from the camera."""
    bounds = CAPTURE_SLOTS.get(slot, CAPTURE_SLOTS["normal"])
    if face_ratio < bounds["min_ratio"]:
        return "too_far"
    if face_ratio > bounds["max_ratio"]:
        return "too_close"
    return "correct"


def encode_jpeg(image, quality=92):
    """BGR numpy array -> raw JPEG bytes, or None on failure."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


# -----------------------------
# Matching
# -----------------------------

def match_embedding(embedding, records, threshold=DEFAULT_THRESHOLD):
    """
    Compares embedding against every record. Each record can hold several
    embeddings (one per registration photo — see add_user); a record's
    score is the BEST similarity across all of its own embeddings, since
    any one of a person's registered angles matching well is enough to
    recognize them.
    Returns (best_record_or_None, similarity_float).
    best_record is None if nothing clears the threshold (still returns the
    top similarity so the UI can show "closest but no match").

    Revoked users (active=False) are skipped entirely, so a person whose
    access has been turned off — or who has been deleted — is never
    matched again; they show up as an unrecognized/unknown face instead.
    """
    records = [r for r in records if r.get("active", True) and r.get("embeddings")]
    if not records:
        return None, 0.0

    best_record = None
    best_sim = -1.0
    probe = embedding.reshape(1, -1)

    for rec in records:
        rec_embeddings = np.stack([e.reshape(-1) for e in rec["embeddings"]])
        sims = cosine_similarity(probe, rec_embeddings)[0]
        sim = float(sims.max())
        if sim > best_sim:
            best_sim = sim
            best_record = rec

    if best_sim >= threshold:
        return best_record, best_sim
    return None, best_sim


def same_person(embedding_a, embedding_b, threshold=SAME_PERSON_THRESHOLD):
    """Cosine-similarity check for 'is this the same face as before?' —
    NOT the same thing as match_embedding, which checks a probe against
    the stored database. This compares two freshly-taken embeddings
    directly against each other, used during registration (Live Capture's
    3 distance shots, and Upload Image's 3 files) so someone else
    stepping into frame partway through doesn't get silently merged into
    the same person's record.

    Returns (is_same: bool, similarity: float).
    """
    a = embedding_a.reshape(1, -1)
    b = embedding_b.reshape(1, -1)
    sim = float(cosine_similarity(a, b)[0][0])
    return sim >= threshold, sim


# -----------------------------
# Unknown-visitor clustering
# -----------------------------
# An unrecognized face isn't in the database, so it has no id/name to
# group by — every previous version of this app labeled EVERY stranger
# "Unknown Visitor", which made the person-profiles view merge totally
# different people into one bucket. This gives each distinct unrecognized
# face a stable-ish numbered identity ("Unknown #3") by nearest-centroid
# matching against embeddings seen so far, so the SAME stranger walking
# past twice shows up as one profile with two scans, while a DIFFERENT
# stranger gets their own number. It's approximate (in-memory, resets on
# restart, no dedup against people who register later) — good enough for
# "was this the same person as before?", not a real identity system.
_unknown_clusters = []   # [{"id": int, "embedding": np.ndarray, "count": int}, ...]
_unknown_lock = threading.Lock()
UNKNOWN_CLUSTER_THRESHOLD = float(os.environ.get("UNKNOWN_CLUSTER_THRESHOLD", "0.5"))


def match_unknown_cluster(embedding):
    """Returns an int cluster id for this unrecognized face's embedding,
    creating a new cluster if it doesn't look like anyone seen before."""
    global _unknown_clusters
    probe = embedding.reshape(1, -1)
    with _unknown_lock:
        best_id, best_sim, best_cluster = None, -1.0, None
        for cluster in _unknown_clusters:
            sim = float(cosine_similarity(probe, cluster["embedding"].reshape(1, -1))[0][0])
            if sim > best_sim:
                best_sim, best_id, best_cluster = sim, cluster["id"], cluster

        if best_cluster is not None and best_sim >= UNKNOWN_CLUSTER_THRESHOLD:
            # Nudge the cluster's centroid toward this new sighting (running average),
            # so it drifts to represent the average of all their captured angles.
            n = best_cluster["count"]
            best_cluster["embedding"] = (best_cluster["embedding"] * n + embedding) / (n + 1)
            best_cluster["count"] = n + 1
            return best_id

        new_id = max((c["id"] for c in _unknown_clusters), default=0) + 1
        _unknown_clusters.append({"id": new_id, "embedding": embedding.copy(), "count": 1})
        return new_id


def match_all_faces(faces, records, threshold=DEFAULT_THRESHOLD, frame=None):
    """For CCTV mode: match every detected face, return a list of results.
    Pass `frame` to also attach a best-effort `emotion` read per face
    (logging only — see _safe_detect_emotion); omit it to skip that."""
    results = []
    for face in faces:
        rec, sim = match_embedding(face.embedding, records, threshold)
        bbox = [int(v) for v in face.bbox]
        results.append({
            "known": rec is not None,
            "name": rec["name"] if rec else "Unknown",
            "similarity": round(sim, 3),
            "bbox": bbox,
            "emotion": _safe_detect_emotion(frame, bbox),
        })
    return results


# -----------------------------
# Evidence snapshots (unknown / unauthorized faces)
# -----------------------------

def _draw_emotion_label(img, emotion):
    """Burns a readable emotion label onto the bottom of a snapshot crop,
    e.g. 'Happiness 83%' on a semi-transparent bar. Returns a new array —
    never mutates `img` in place, since callers may reuse the source
    frame elsewhere."""
    label_emotion = emotion.get("emotion")
    confidence = emotion.get("confidence")
    if not label_emotion or confidence is None:
        return img

    out = img.copy()
    h, w = out.shape[:2]
    label = f"{label_emotion.capitalize()} {int(round(confidence * 100))}%"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(0.7, w / 220))
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    bar_h = text_h + baseline + 10

    # Semi-transparent bar across the bottom so the label is legible
    # regardless of what's behind it in the photo.
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, label, (6, h - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def save_face_snapshot(frame, bbox=None, prefix="capture", padding=0.35, emotion=None):
    """Crops a little padding around `bbox` out of `frame` and saves it as
    a JPEG under SNAPSHOT_DIR — used to keep photographic evidence of an
    unrecognized or bypass attempt. If bbox is None/invalid, saves the
    full frame instead.

    `emotion` is the optional dict from emotion_engine.detect_emotion()
    (e.g. {"emotion": "happiness", "confidence": 0.83, ...}) — when given,
    it's burned onto the saved image as a readable label so the emotion
    is visible directly in the photo, not just in the log's text column.

    Returns a web-servable path like '/static/snapshots/denied_20260816_...jpg',
    or None on failure."""
    if frame is None:
        return None
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        crop = frame
        if bbox and len(bbox) == 4:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = (x2 - x1), (y2 - y1)
            x1 = max(0, int(x1 - bw * padding))
            y1 = max(0, int(y1 - bh * padding))
            x2 = min(w, int(x2 + bw * padding))
            y2 = min(h, int(y2 + bh * padding))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]

        if emotion:
            crop = _draw_emotion_label(crop, emotion)

        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        ok = cv2.imwrite(filepath, crop)
        if not ok:
            return None
        return f"/static/snapshots/{filename}"
    except Exception:
        return None


# -----------------------------
# Camera capture
# -----------------------------

def capture_frame(camera_index=CAMERA_INDEX_ENTRY):
    """Opens the local camera, grabs one frame, releases it. Returns a BGR frame or None."""
    cam = cv2.VideoCapture(camera_index)
    try:
        if not cam.isOpened():
            return None
        # Give cheap USB webcams a couple of frames to warm up / auto-expose.
        frame = None
        for _ in range(3):
            success, frame = cam.read()
            if not success:
                return None
        return frame
    finally:
        cam.release()


# -----------------------------
# Live camera manager
#
# Keeps ONE camera open in a background thread so the browser can show a
# real live feed while the same frames are also periodically fed through
# face recognition for automatic, no-click scanning.
# -----------------------------

class CameraManager:
    def __init__(self, camera_index=CAMERA_INDEX_ENTRY,
                 scan_interval=float(os.environ.get("FACE_SCAN_INTERVAL", "0.6")),
                 gate_cooldown=6.0, threshold=DEFAULT_THRESHOLD,
                 cctv_cooldown=2.0, multi_person_cooldown=2.5):
        self.camera_index = camera_index
        self.scan_interval = scan_interval   # seconds between recognition passes
        self.gate_cooldown = gate_cooldown    # don't re-fire the same granted person for this long
        self.cctv_cooldown = cctv_cooldown    # don't re-log the same CCTV face too often
        self.multi_person_cooldown = multi_person_cooldown
        self.threshold = threshold
        self.on_gate_event = None             # callback(id, name, granted, confidence, frame, bbox, emotion) -> app.py
        self.on_cctv_event = None             # callback(frame, detection_dict) -> app.py; fires continuously,
                                               # not just on manual "Detect now" clicks, but ONLY while
                                               # cctv_enabled is True (see enable_cctv/disable_cctv below)
        self.on_multi_person = None           # callback(frame, count) -> app.py; two+ people at the gate

        # There's only one physical camera. Entry/Exit/CCTV are logical
        # modes selected in the dashboard, not separate feeds — so CCTV
        # face logging must only happen while the user is actually
        # looking at the CCTV section, not any time the shared camera
        # loop happens to be running for Entry/Exit. app.py should call
        # enable_cctv()/disable_cctv() when the user opens/leaves that
        # section (see those methods below).
        self.cctv_enabled = False

        self._cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._latest_result = {"faces": [], "gate": None, "timestamp": None, "person_count": None}
        self._last_gate_time = {}     # identity key -> last-fired timestamp (see _run_recognition)
        self._last_gate_outcome = {}  # identity key -> {"opened": bool, "anomaly": bool, "note": str|None}
        self._last_cctv_time = {}   # name -> last-logged timestamp, so continuous tracking
                                     # doesn't spam the log every single scan pass
        self._last_multi_time = 0.0

    def is_running(self):
        return self._running

    def enable_cctv(self):
        """Call this when the user opens the CCTV section in the dashboard.
        Turns on continuous face logging for the CCTV tab. Does not open a
        new camera — the shared camera loop keeps running either way for
        Entry/Exit; this only controls whether faces get logged as CCTV
        events."""
        self.cctv_enabled = True

    def disable_cctv(self):
        """Call this when the user leaves the CCTV section (navigates away,
        closes the tab, etc). Stops CCTV face logging immediately, and
        clears the per-name cooldown state so a fresh session starts clean
        next time the section is reopened instead of silently reusing
        stale timestamps from the last visit."""
        self.cctv_enabled = False
        with self._lock:
            self._last_cctv_time = {}

    def start(self):
        with self._lock:
            if self._running:
                return
            cap = cv2.VideoCapture(self.camera_index)
            if CAM_WIDTH:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            if CAM_HEIGHT:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    "Could not open the camera. Check it's connected, not in use "
                    "by another app, and CAMERA_INDEX is correct."
                )
            self._cap = cap
            self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def _loop(self):
        last_scan = 0.0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame

            now = time.time()
            if now - last_scan >= self.scan_interval:
                last_scan = now
                try:
                    self._run_recognition(frame)
                except Exception:
                    # Never let a bad frame kill the capture loop.
                    pass

            time.sleep(0.02)

    def _run_recognition(self, frame):
        faces = detect_faces(frame)
        records = load_database()
        detections = match_all_faces(faces, records, self.threshold, frame=frame)

        # --- Continuous CCTV tracking ---------------------------------
        # Fires on every automatic recognition pass (not just when the
        # "Detect now" button is clicked), so passing faces get logged as
        # they happen. A short per-name cooldown stops the same person
        # from being logged again on every single pass while they linger
        # in frame.
        if self.on_cctv_event and self.cctv_enabled:
            now_cctv = time.time()
            # zip(faces, detections): match_all_faces() builds `detections`
            # by iterating `faces` in the same order, so the i-th detection
            # always corresponds to the i-th raw face (and its embedding).
            for face, det in zip(faces, detections):
                if not det["known"]:
                    # Give this stranger a stable per-person label instead of
                    # a generic "Unknown" that lumps every stranger together
                    # (see match_unknown_cluster) — this also fixes the
                    # cooldown below, which used to key off "Unknown" for
                    # EVERY unrecognized face, so a second different
                    # stranger arriving soon after the first wouldn't get
                    # logged at all.
                    det["name"] = f"Unknown #{match_unknown_cluster(face.embedding)}"
                key = det["name"]
                last = self._last_cctv_time.get(key, 0.0)
                if now_cctv - last >= self.cctv_cooldown:
                    self._last_cctv_time[key] = now_cctv
                    try:
                        self.on_cctv_event(frame, det)
                    except Exception:
                        pass

        # --- Multi-person guard (anti-tailgating) ----------------------
        # Counts full bodies (YOLO), not just faces, since a second person
        # standing beside/behind the one presenting to the camera may not
        # have a clean face angle. If 2+ people are in frame, nobody gets
        # access this pass, regardless of whether a face matched.
        # `faces` was already detected just above, so it's passed straight
        # through as the fallback instead of re-running face detection.
        person_count, _person_boxes = person_detector.count_persons(frame, fallback_faces=lambda: faces)
        if person_count >= 2:
            gate_result = {
                "granted": False,
                "name": "Multiple people detected",
                "id": None,
                "confidence": None,
                "bbox": None,
                "multi_person": True,
                "person_count": person_count,
            }
            with self._lock:
                self._latest_result = {
                    "faces": detections,
                    "gate": gate_result,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "person_count": person_count,
                }
            if self.on_multi_person:
                now_mp = time.time()
                if now_mp - self._last_multi_time >= self.multi_person_cooldown:
                    self._last_multi_time = now_mp
                    try:
                        self.on_multi_person(frame, person_count)
                    except Exception:
                        pass
            return  # skip normal single-person gate logic entirely

        gate_result = None
        if faces:
            face = largest_face(faces)
            rec, sim = match_embedding(face.embedding, records, self.threshold)
            if rec:
                gate_name, gate_id = rec["name"], rec["id"]
            else:
                # Give this stranger a stable per-person label/id instead of
                # the literal "Unknown Visitor" that every previous version
                # used for EVERY unmatched face (see match_unknown_cluster,
                # already used for this exact reason on the CCTV path
                # above). Reusing one shared string here meant the cooldown
                # check below treated any two different strangers arriving
                # within gate_cooldown of each other as "the same person
                # recently seen" and silently dropped the second one's
                # on_gate_event entirely — no log, no snapshot, no denial
                # recorded for an actual unauthorized visitor at the gate.
                unknown_id = match_unknown_cluster(face.embedding)
                gate_name, gate_id = f"Unknown #{unknown_id}", f"unknown_{unknown_id}"
            gate_result = {
                "granted": rec is not None,
                "name": gate_name,
                "id": rec["id"] if rec else None,
                "confidence": round(sim, 3),
                "bbox": [int(v) for v in face.bbox],
                "emotion": _safe_detect_emotion(frame, face.bbox),
            }

        if gate_result and self.on_gate_event:
            now = time.time()
            # Keyed by identity (registered id, or the unknown-cluster id
            # for a stranger) rather than the display name, so the
            # cooldown only suppresses re-firing for the SAME person —
            # a different known user with an identical name, or a
            # different unrecognized stranger, still gets their own
            # gate event even if one fired moments ago.
            cooldown_key = gate_id
            last_time = self._last_gate_time.get(cooldown_key, 0.0)
            same_person_recently = now - last_time < self.gate_cooldown
            if not same_person_recently:
                self._last_gate_time[cooldown_key] = now
                outcome = self.on_gate_event(gate_result["id"], gate_result["name"], gate_result["granted"],
                                              gate_result["confidence"], frame, gate_result["bbox"],
                                              gate_result.get("emotion"))
                if isinstance(outcome, dict):
                    self._last_gate_outcome[cooldown_key] = outcome
            # Apply the real outcome (this pass or, during cooldown, the
            # last one recorded for this specific person) so the live
            # status never claims "granted" for a match that didn't
            # actually open the gate — e.g. anti-passback blocks a
            # re-open but the face still matched the database.
            last_outcome = self._last_gate_outcome.get(cooldown_key)
            if last_outcome is not None:
                gate_result["granted"] = last_outcome.get("opened", gate_result["granted"])
                if last_outcome.get("anomaly"):
                    gate_result["anomaly"] = True
                if last_outcome.get("note"):
                    gate_result["note"] = last_outcome["note"]

        with self._lock:
            self._latest_result = {
                "faces": detections,
                "gate": gate_result,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "person_count": person_count,
            }

    def get_latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_latest_result(self):
        with self._lock:
            return dict(self._latest_result)

    def _draw_annotations(self, frame):
        """Draw annotations on a mirrored camera frame so all text stays readable.

        The camera preview is intentionally shown like a front-facing/selfie
        camera. We therefore mirror the frame on the server FIRST, then draw
        the face boxes, names, confidence values, and person count. This keeps
        the camera view mirrored while preventing labels such as ``Unknown``
        or a person's name from appearing backwards.
        """
        frame = frame.copy()

        # Mirror the camera image before drawing any text.
        frame = cv2.flip(frame, 1)

        with self._lock:
            faces = list(self._latest_result.get("faces") or [])

        h, w = frame.shape[:2]

        for face in faces:
            bbox = face.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # The recognition bbox belongs to the original camera frame.
            # Convert it to the mirrored frame before drawing.
            mx1 = max(0, int(w - x2))
            mx2 = min(w - 1, int(w - x1))
            y1 = max(0, int(y1))
            y2 = min(h - 1, int(y2))

            known = face.get("known", False)
            name = face.get("name", "Unknown")
            similarity = face.get("similarity")

            color = (0, 200, 0) if known else (0, 0, 255)  # BGR: green / red

            cv2.rectangle(frame, (mx1, y1), (mx2, y2), color, 2)

            label = name
            if similarity is not None:
                label = f"{name} ({similarity:.2f})"

            # Label background for readability over busy video.
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(y1 - 10, text_h + 4)

            cv2.rectangle(
                frame,
                (mx1, label_y - text_h - 6),
                (mx1 + text_w + 6, label_y + 2),
                color,
                -1,
            )

            # Text is drawn after mirroring, so it remains normal/readable.
            cv2.putText(
                frame,
                label,
                (mx1 + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        # Running count, top-left corner. No solid background block —
        # just the yellow text with a thin black outline (drawn thicker
        # first, then the yellow text on top) so it stays readable over
        # whatever's in the video without covering the frame with a box.
        count_label = f"Persons detected: {len(faces)}"
        cv2.putText(
            frame,
            count_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            frame,
            count_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return frame

    def get_frame_jpeg(self, quality=80, annotate=True):
        """`annotate=False` skips the gate-style overlay (bounding boxes,
        name/confidence labels, "Persons detected: N" box) and just
        encodes the raw camera frame. Used by the Register page's Live
        Capture preview, which draws its own guide outline + labels in
        the browser — the server-side overlay would otherwise sit right
        on top of those HTML labels and read as garbled double text."""
        frame = self.get_latest_frame()
        if frame is None:
            return None
        if annotate:
            frame = self._draw_annotations(frame)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buf.tobytes()


def mjpeg_generator(camera_manager, fps=10, annotate=True):
    """Yields a multipart/x-mixed-replace stream — set as a Flask Response mimetype.
    Pass annotate=False for a clean feed with no server-drawn overlay
    (see get_frame_jpeg)."""
    interval = 1.0 / fps
    while True:
        frame_bytes = camera_manager.get_frame_jpeg(annotate=annotate)
        if frame_bytes:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(interval)