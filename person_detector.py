"""
AI Smart Gate System
Person Detector (YOLO)

Full-body person detection used for the anti-tailgating "two people at
once" guard. InsightFace only ever looks for *faces*, so two people
standing shoulder-to-shoulder with one face turned away can still look
like "one face" to it. YOLO detects full person bodies (COCO class 0),
which is what we actually want to guard against.

Kept separate from face_engine.py so the YOLO/person-counting code can
be maintained, tested, replaced, or run on a different device
independently of the InsightFace face-matching code. This module has
no dependency on face_engine.py — if YOLO/ultralytics isn't available,
callers can optionally supply a `fallback_faces` callable (e.g.
face_engine.detect_faces) instead of this module importing it directly,
which would otherwise create a circular import between the two files.
"""

import os
import threading

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_PERSON_CONF = float(os.environ.get("YOLO_PERSON_CONF", "0.3"))

# --- NMS overlap threshold ---------------------------------------------
# ultralytics' model.predict() defaults to iou=0.7 when not given
# explicitly. That default is tuned for general object detection where
# two objects of the same class rarely overlap much. At a gate, two
# people standing close together (e.g. shoulder-to-shoulder tailgating,
# which is exactly the case we're trying to catch) can easily produce
# two person boxes whose IoU is well above 0.7 — NMS then treats the
# weaker box as a duplicate of the stronger one and suppresses it,
# so two real people collapse into a single detection. Lowering this
# tightens NMS (boxes must overlap LESS before one is suppressed), which
# keeps both people as separate detections. 0.45 is a reasonable
# starting point; if you still see close pairs merging, try 0.3-0.35 —
# if you start seeing ONE real person double-counted as two, raise it.
YOLO_PERSON_IOU = float(os.environ.get("YOLO_PERSON_IOU", "0.3"))

# --- Distance filter --------------------------------------------------
# A person's real-world height is roughly constant, so a bounding box
# that fills a large fraction of the frame means someone is standing
# close to the camera (i.e. at the gate); a small box means they're far
# away — e.g. someone walking past in the background — and shouldn't
# count toward "multiple people at the gate". This is expressed as a
# ratio of box height to frame height rather than a fixed pixel size, so
# it keeps working regardless of camera resolution.
#
# 0.35 is a starting point, not a universal constant — the right value
# depends on your camera's mounting height/angle and how far the gate is
# from the lens. Tune it by testing: temporarily log box_h / frame_h for
# a person actually standing at the gate vs. someone in the background,
# then set the threshold roughly halfway between the two.
YOLO_MIN_PERSON_HEIGHT_RATIO = float(os.environ.get("YOLO_MIN_PERSON_HEIGHT_RATIO", "0.6"))

# Set YOLO_DEBUG_RATIOS=1 to print each detected person's box-height/frame-height
# ratio (and whether it passed the distance filter) for every frame. Use this to
# figure out the right YOLO_MIN_PERSON_HEIGHT_RATIO for your camera: stand at the
# gate and note your ratio, then have someone walk past in the background and note
# theirs, and set the threshold roughly halfway between.
YOLO_DEBUG_RATIOS = os.environ.get("YOLO_DEBUG_RATIOS", "0") == "1"

# --- Region of interest -------------------------------------------------
# The height-ratio filter above only rejects people who are too *small*
# in-frame (i.e. too far away). It does nothing about someone who is
# close to the camera but standing off to the side — e.g. a waiting
# area, a second doorway, or a hallway visible in the same shot next to
# the gate lane. ROI_POINTS lets you additionally restrict counting to
# a specific polygon within the frame.
#
# Format: comma-separated x,y pairs, as fractions (0.0-1.0) of frame
# width/height, in point order. Using fractions instead of pixels means
# the ROI keeps working if the camera resolution changes.
#
# IMPORTANT - use a TRAPEZOID, not a rectangle, for most gate cameras.
# ROI_POINTS is a shape in the 2D video frame, not a shape on the real
# floor - those are only the same thing if the camera looks straight
# down. A typical gate camera is mounted at an angle looking along the
# lane, so perspective makes a constant-width lane on the floor project
# as NARROW near the top of the frame (far from the camera) and WIDE
# near the bottom (close to the camera). A plain rectangle
# (e.g. 0.35,0.0,0.65,0.0,0.65,1.0,0.35,1.0) fixes the same width at
# every distance, so it ends up too narrow to catch people near the
# bottom of the frame and too wide near the top. A trapezoid that
# narrows toward the top matches the actual lane shape:
#
#   (0.42,0.3)   (0.58,0.3)     <- far edge of the lane, narrow
#        \           /
#         \         /
#   (0.15,1.0)   (0.85,1.0)     <- near edge (bottom of frame), wide
#
#   ROI_POINTS=0.42,0.3,0.58,0.3,0.85,1.0,0.15,1.0
#
# These numbers are only an illustration - the real shape depends on
# your camera's mounting height/angle. Calibrate it by testing: set
# ROI_DEBUG=1 (see below), have someone stand at the far end of the
# lane and note their foot-point coordinates, then have them stand at
# the near end and note those too, and use the coordinates on either
# side of the lane at each distance as your four points. This doesn't
# need to be exact - a hand-calibrated trapezoid is enough for "did
# someone walk into the gate area", and is much simpler than a full
# camera calibration / homography, which isn't necessary for this.
#
# Leave unset (default) to disable ROI filtering - every detection that
# passes the distance filter is counted, same as before this existed.
_ROI_POINTS_RAW = os.environ.get("ROI_POINTS", "").strip()


def _parse_roi_points(raw):
    if not raw:
        return None
    try:
        vals = [float(v) for v in raw.split(",")]
        if len(vals) < 6 or len(vals) % 2 != 0:  # need >=3 points
            raise ValueError("ROI_POINTS needs at least 3 x,y pairs")
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    except ValueError as e:
        print(f"[person_detector] Ignoring invalid ROI_POINTS ({e})")
        return None


YOLO_ROI_POINTS = _parse_roi_points(_ROI_POINTS_RAW)  # None disables ROI filtering

# Set ROI_DEBUG_RATIOS=1 to print each detected person's foot point and
# whether it fell inside the ROI polygon, for every frame. Use this the
# same way as YOLO_DEBUG_RATIOS: stand in the gate lane vs. outside it
# and watch which points pass, to dial in ROI_POINTS.
ROI_DEBUG = os.environ.get("ROI_DEBUG", "0") == "1"


def _point_in_roi(x, y, frame_w, frame_h, roi_points):
    """True if the pixel point (x, y) falls inside `roi_points` (a list
    of (fx, fy) fractional-coordinate polygon vertices). Uses OpenCV's
    pointPolygonTest when available (cv2 is already a dependency of
    ultralytics, so this doesn't add a new one); falls back to a plain
    ray-casting test if cv2 isn't importable for some reason."""
    poly_px = [(fx * frame_w, fy * frame_h) for fx, fy in roi_points]
    try:
        import cv2
        import numpy as np
        contour = np.array(poly_px, dtype="float32")
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
    except ImportError:
        # Pure-Python ray-casting fallback (even-odd rule).
        inside = False
        n = len(poly_px)
        px, py = x, y
        x1, y1 = poly_px[-1]
        for x2, y2 in poly_px:
            if ((y1 > py) != (y2 > py)) and \
                    (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1):
                inside = not inside
            x1, y1 = x2, y2
        return inside


_yolo_model = None
_yolo_lock = threading.Lock()
_yolo_unavailable = False  # set True after a failed load so we don't retry every frame


def _get_yolo():
    """Lazily loads a YOLOv8 nano model for person detection. If the
    `ultralytics` package (or its weights) isn't available, this fails
    once, prints a warning, and every caller falls back to face-count
    instead — the gate keeps working, just with a slightly weaker
    multi-person check."""
    global _yolo_model, _yolo_unavailable
    if _yolo_model is None and not _yolo_unavailable:
        with _yolo_lock:
            if _yolo_model is None and not _yolo_unavailable:
                try:
                    from ultralytics import YOLO
                    _yolo_model = YOLO(YOLO_MODEL_PATH)
                    print(f"[person_detector] YOLO person-detector loaded ({YOLO_MODEL_PATH})")
                except Exception as e:
                    _yolo_unavailable = True
                    print(f"[person_detector] YOLO unavailable ({e}) — multi-person check will "
                          f"fall back to face-count. Install with: pip install ultralytics")
    return _yolo_model


def is_available():
    """True if the YOLO model is loaded (or loadable) right now."""
    return _get_yolo() is not None

def detect_persons_scored(frame, min_height_ratio=None, roi_points=None):
    """Returns a list of dicts, one per person detection that survives
    the SAME distance-ratio + ROI filtering that count_persons() applies,
    each shaped as:
 
        {"confidence": float, "box": [x1, y1, x2, y2]}
 
    box coordinates are integer pixel coordinates in `frame`'s own
    width/height. Returns None if the YOLO model isn't available (no
    face-count fallback here, since a face detector has no meaningful
    "confidence"/"box" pair in this same sense).
 
    This exists as its own function (rather than being inlined only
    inside count_persons()) so that evaluation code can pull the exact
    same filtered detections — with their confidence scores — that the
    live gate uses, for Precision/Recall/F1/mAP. count_persons() below
    is just a thin wrapper around this.
    """
    model = _get_yolo()
    if model is None:
        return None
 
    ratio = YOLO_MIN_PERSON_HEIGHT_RATIO if min_height_ratio is None else min_height_ratio
    roi = YOLO_ROI_POINTS if roi_points is None else roi_points
    frame_h, frame_w = frame.shape[0], frame.shape[1]
 
    results = model.predict(frame, verbose=False, conf=YOLO_PERSON_CONF,
                             iou=YOLO_PERSON_IOU, classes=[0])
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
 
            box_ratio = (y2 - y1) / frame_h
            passed_distance = box_ratio >= ratio
            if YOLO_DEBUG_RATIOS:
                print(f"[person_detector] box height ratio={box_ratio:.3f} "
                      f"(threshold={ratio:.3f}) -> "
                      f"{'COUNTED' if passed_distance else 'ignored (too far)'}")
            if not passed_distance:
                continue  # too small in-frame -> too far from the camera to be "at the gate"
 
            if roi:
                foot_x, foot_y = (x1 + x2) / 2, y2  # bottom-center = where they're standing
                passed_roi = _point_in_roi(foot_x, foot_y, frame_w, frame_h, roi)
                if ROI_DEBUG:
                    print(f"[person_detector] foot point=({foot_x:.0f},{foot_y:.0f}) -> "
                          f"{'COUNTED' if passed_roi else 'ignored (outside ROI)'}")
                if not passed_roi:
                    continue  # standing outside the defined gate area
 
            detections.append({
                "confidence": confidence,
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })
    return detections

def count_persons(frame, fallback_faces=None, min_height_ratio=None, roi_points=None):
    """Returns (count, boxes) — the number of *people* (full bodies, not
    just faces) visible in `frame` AND close enough to the camera to
    plausibly be at the gate, via YOLOv8 (COCO class 0 = person).

    Detections whose box height is below `min_height_ratio` (as a
    fraction of the frame's height) are discarded before counting —
    this is what stops someone merely visible in the background from
    triggering a "multiple people" false positive. Defaults to
    YOLO_MIN_PERSON_HEIGHT_RATIO (see above) if not given explicitly.

    Detections whose foot point (bottom-center of the box) falls outside
    `roi_points` (a list of (fx, fy) fractional polygon vertices) are
    also discarded — this restricts counting to a specific area of the
    frame, e.g. the gate lane itself rather than an adjacent walkway.
    For an angled camera this should be a trapezoid, not a rectangle —
    see the YOLO_ROI_POINTS comment above for why. Defaults to
    YOLO_ROI_POINTS (see above) if not given explicitly; None/empty
    disables ROI filtering entirely.

    If ultralytics isn't installed, falls back to `fallback_faces` — a
    zero-arg callable (or a plain list) of already-detected faces (e.g.
    from face_engine.detect_faces) supplied by the caller, so this
    module never needs to import face_engine itself. Note: neither the
    distance filter nor the ROI filter apply to this fallback path,
    since InsightFace's face detector doesn't reliably pick up
    small/far-away faces to begin with — it's already implicitly biased
    toward close-up faces.

    Used by the main gate to refuse access when more than one person is
    presenting to the camera at once (anti-tailgating).
    """
    model = _get_yolo()
    if model is None:
        faces = fallback_faces() if callable(fallback_faces) else (fallback_faces or [])
        return len(faces), [[int(v) for v in f.bbox] for f in faces]

    ratio = YOLO_MIN_PERSON_HEIGHT_RATIO if min_height_ratio is None else min_height_ratio
    roi = YOLO_ROI_POINTS if roi_points is None else roi_points
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    min_box_h = frame_h * ratio

    results = model.predict(frame, verbose=False, conf=YOLO_PERSON_CONF,
                             iou=YOLO_PERSON_IOU, classes=[0])
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_ratio = (y2 - y1) / frame_h
            passed_distance = box_ratio >= ratio
            if YOLO_DEBUG_RATIOS:
                print(f"[person_detector] box height ratio={box_ratio:.3f} "
                      f"(threshold={ratio:.3f}) -> "
                      f"{'COUNTED' if passed_distance else 'ignored (too far)'}")
            if not passed_distance:
                continue  # too small in-frame -> too far from the camera to be "at the gate"

            if roi:
                foot_x, foot_y = (x1 + x2) / 2, y2  # bottom-center = where they're standing
                passed_roi = _point_in_roi(foot_x, foot_y, frame_w, frame_h, roi)
                if ROI_DEBUG:
                    print(f"[person_detector] foot point=({foot_x:.0f},{foot_y:.0f}) -> "
                          f"{'COUNTED' if passed_roi else 'ignored (outside ROI)'}")
                if not passed_roi:
                    continue  # standing outside the defined gate area

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return len(boxes), boxes