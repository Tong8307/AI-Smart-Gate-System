"""
AI Smart Gate System
Flask backend

Serves the dashboard (templates/index.html) and exposes the API it calls:
  POST /api/login            { username, password }
  POST /api/logout
  GET  /api/session          -> whether the current browser is logged in

  POST /api/register         multipart form: photos (x3), name, role, mode=live|upload
                              (mode tells the server which extra checks
                              to apply — see register() below)
  GET  /api/registrations    recently registered users
  POST /api/register/quality-check
                              multipart form: photo, [slot=normal|far|close]
                              -> validates one uploaded candidate photo
                              (face count, detectability, blur, size,
                              color vs black & white, front-facing,
                              facial-feature coverage, and — if slot is
                              given — distance guidance) without storing
                              anything. Used by the Upload Image
                              registration path. Photos reused across
                              more than one of the 3 slots are caught
                              separately, at /api/register time, once all
                              3 are available to compare.
  GET  /video_feed_register  live MJPEG preview for the Live Capture panel
                              — reuses the same shared camera as the gate
                              feeds below; a browser can't also open that
                              device itself via getUserMedia once this
                              process already holds it.
  GET  /api/register/live/check?slot=normal|far|close
                              -> same 4 checks + distance guidance, but
                              read from the CURRENT shared-camera frame
                              instead of an upload. Polled continuously by
                              the Live Capture guide.
  POST /api/register/live/capture
                              form: slot -> re-validates the current
                              shared-camera frame and, if it passes,
                              returns it as a raw JPEG (not JSON) to be
                              held as that slot's captured photo.

  GET  /video_feed_entry     live MJPEG stream from the Entry Gate camera
  GET  /api/scan/latest      latest automatic detection/match result (poll this)
    GET  /api/gate/state
  POST /api/gate/unlock
  POST /api/gate/lock
  GET  /api/settings         current auto-lock settings
  POST /api/settings         { auto_lock_enabled?, auto_lock_delay? }

  POST /api/cctv/scan        force an immediate multi-face check (same camera)
                              (CCTV also logs continuously in the background —
                              this just forces an extra check right now)
  GET  /api/logs?source=gate|cctv
  GET  /api/logs/profiles    gate+CCTV events grouped by person, with photos

  Anti-tailgating: every gate scan (auto AND manual) also runs a YOLO
  person-count check (see face_engine.count_persons). If 2+ full people
  are in frame, access is refused for everyone that pass, flagged with
  "multi_person": true and logged as a warning — regardless of whether a
  face matched.

Each physical camera is opened in its own background thread (face_engine.CameraManager)
and shared by the live video feed and the automatic face-matching loop, so
the dashboard shows real video *and* scans continuously without needing the
"Scan now" button to be clicked.
"""

import os
import functools
import threading
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session, Response

import face_engine as fe

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-me")

# --- Hardcoded admin account -------------------------------------------
# For real deployments, set these via environment variables instead of
# leaving the defaults in place.
ADMIN_USERNAME = os.environ.get("GATE_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("GATE_ADMIN_PASSWORD", "admin123")

AUTO_LOCK_DELAY = float(os.environ.get("GATE_AUTO_LOCK_DELAY", "8"))

# --- Runtime-editable gate settings ---------------------------------------
# Backed by the Automation card on the Gate Control page (see
# /api/settings below). Starts from the env-var defaults above but can be
# changed live from the dashboard without a restart.
gate_settings = {
    "auto_lock_enabled": True,
    "auto_lock_delay": AUTO_LOCK_DELAY,
}
gate_settings_lock = threading.Lock()

# --- Camera / location labels -------------------------------------------
# Purely cosmetic names shown in the dashboard tabs, the CCTV camera
# dropdown, and log entries. They do NOT change which physical camera is
# used — that's still controlled by CAMERA_INDEX_ENTRY / CAMERA_INDEX_EXIT
# in face_engine.py, and by ensure_camera_running("exit")'s fallback to
# the shared camera when only one webcam is plugged in (see below).
# Rename these via env vars to match your actual setup, e.g. if your
# single camera is mounted at a corner overlooking both directions:
#   GATE_LABEL_ENTRY="Corner cam - entry side"
#   GATE_LABEL_EXIT="Corner cam - exit side"
CAMERA_LABEL_ENTRY = os.environ.get("GATE_LABEL_ENTRY", "Entry Gate")
CAMERA_LABEL_EXIT = os.environ.get("GATE_LABEL_EXIT", "Exit Gate")

# --- In-memory state (gate_events/cctv_events reset on restart; swap for a
#     DB if you need that history to survive a restart too. Registrations
#     are different: the actual face data lives in face_database.pkl on
#     disk via face_engine, so it already survives restarts — we just
#     rebuild this display list from that file on startup below.) --------
gate_state = {"unlocked": False}          # outside (entry) gate
gate_state_inside = {"unlocked": False}   # inside (exit) gate
gate_events = []   # each: {time, name, granted, confidence, camera, [anomaly, note, snapshot]}
cctv_events = []   # each: {time, name, known, camera, location, [snapshot]}
cctv_location = {"key": "corner", "label": "Corner"}

# --- Occupancy / anti-passback ------------------------------------------
# Keyed by the registered user's id. A person is added here the moment
# they're granted entry, and only removed when an admin marks them as
# exited (see /api/occupancy/<id>/exit) — this one camera can't tell
# "coming in" from "going out" on its own, so exits are a manual step.
# While someone is marked "inside", a second granted match for that same
# id does NOT re-open the gate automatically; it's logged as an anomaly
# instead, since it usually means either tailgating/passback abuse or
# someone who just forgot to mark themselves exited.
occupancy = {}   # user_id -> {id, name, since}


def _load_recent_registrations_from_db():
    """Rebuilds the dashboard's registrations list from the persisted
    face database, so registered users still show up after a restart."""
    try:
        records = fe.load_database()
    except Exception:
        return []
    items = [
        {
            "id": r.get("id"),
            "name": r.get("name", "Unknown"),
            "role": r.get("role", "Resident"),
            "when": r.get("registered_at", ""),
            "active": r.get("active", True),
        }
        for r in records
    ]
    # Most recently registered first.
    items.sort(key=lambda r: r["when"], reverse=True)
    return items


recent_registrations = _load_recent_registrations_from_db()

# Reference embedding for the Live Capture registration flow (see
# register_live_capture below). Slot "normal" is always shot first and
# (re)sets this; the "far"/"close" shots that follow are compared
# against it via fe.same_person so someone else stepping into frame
# mid-flow does not get silently merged into the same registration.
# Single global (not per-session) on purpose, matching the rest of
# this prototype: one shared camera, one registration in progress at
# a time.
_live_capture_ref = {"embedding": None}
_live_capture_ref_lock = threading.Lock()

# --- Live camera ----------------------------------------------------------
# The prototype uses ONE physical webcam. Entry, Exit and CCTV are logical
# modes of that same camera. Only the currently selected page changes what
# the automatic recognition result means.
camera_mode = "entry"
camera_mode_lock = threading.Lock()
shared_camera = fe.CameraManager(camera_index=fe.CAMERA_INDEX)


def set_camera_mode(mode):
    global camera_mode
    if mode not in ("entry", "exit", "cctv"):
        raise ValueError("Invalid camera mode")
    with camera_mode_lock:
        camera_mode = mode
    return mode


def get_camera_mode():
    with camera_mode_lock:
        return camera_mode

def _auto_relock_after_delay(state=None):
    import time
    state = state if state is not None else gate_state
    with gate_settings_lock:
        enabled = gate_settings["auto_lock_enabled"]
        delay = gate_settings["auto_lock_delay"]
    if not enabled:
        # Auto-lock is switched off — leave the gate as-is; it now only
        # re-locks via the manual "Lock gate" button.
        return
    time.sleep(delay)
    state["unlocked"] = False


def handle_entry_event(user_id, name, granted, confidence, frame=None, bbox=None, emotion=None):
    """Called from the camera's background thread when it auto-recognizes
    someone with active access. Blocks a silent re-open if that same
    person is already marked as inside (anti-passback).

    `emotion` is a best-effort read from emotion_engine.py (or None if
    that model isn't set up) — logging only, it never affects the access
    decision.

    Returns a dict describing what actually happened — {"opened": bool,
    "anomaly": bool, "note": str|None} — so the caller can correct the
    live status: a face matching the database does NOT always mean the
    gate opened (e.g. anti-passback), and the live banner needs to know
    that, not just the raw match result."""
    if user_id and user_id in occupancy:
        snapshot = fe.save_face_snapshot(frame, bbox, prefix="reentry", emotion=emotion)
        note = ("Already marked as inside — gate was NOT auto-opened. "
                "Mark them exited first if this is a false alarm.")
        gate_events.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name,
            "granted": False,
            "confidence": confidence,
            "camera": "CAM 01 (Entry)",
            "anomaly": True,
            "note": note,
            "snapshot": snapshot,
            "emotion": emotion,
        })
        return {"opened": False, "anomaly": True, "note": note}

    # Unknown/unmatched people are always denied and recorded with a face snapshot.
    if not granted or not user_id:
        snapshot = fe.save_face_snapshot(frame, bbox, prefix="denied_unknown", emotion=emotion)
        note = "Unknown or unmatched person — access denied. Face recorded in access log."
        gate_events.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name or "Unknown Visitor",
            "granted": False,
            "confidence": confidence,
            "camera": "CAM 01 (Entry)",
            "anomaly": False,
            "note": note,
            "snapshot": snapshot,
            "emotion": emotion,
        })
        gate_state["unlocked"] = False
        return {"opened": False, "anomaly": False, "note": note}

    gate_state["unlocked"] = True
    occupancy[user_id] = {
        "id": user_id,
        "name": name,
        "since": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    gate_events.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": name,
        "granted": True,
        "confidence": confidence,
        "camera": "CAM 01 (Entry)",
        "snapshot": fe.save_face_snapshot(frame, bbox, prefix="granted", emotion=emotion),
        "emotion": emotion,
    })
    threading.Thread(target=_auto_relock_after_delay, daemon=True).start()
    return {"opened": True, "anomaly": False, "note": None}


def handle_exit_event(user_id, name, granted, confidence, frame=None, bbox=None, emotion=None):
    """Called from the inside camera's background thread when it
    auto-recognizes someone. This gate means EXIT, not entry: a match
    unlocks the inside gate and clears the person from occupancy (if they
    were marked inside). Someone matched here who wasn't marked inside is
    still let through (nothing to anti-passback against on the way out),
    but the event is flagged so it's easy to notice on the log.

    Returns {"opened": bool, "anomaly": bool, "note": str|None} — see
    handle_entry_event for why this matters for the live status."""
    # Unknown/unmatched people are denied at the exit gate too.
    if not granted or not user_id:
        snapshot = fe.save_face_snapshot(frame, bbox, prefix="exit_denied_unknown", emotion=emotion)
        note = "Unknown or unmatched person — exit access denied. Face recorded in access log."
        gate_events.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name or "Unknown Visitor",
            "granted": False,
            "confidence": confidence,
            "camera": "CAM 02 (Exit)",
            "anomaly": False,
            "note": note,
            "snapshot": snapshot,
            "emotion": emotion,
        })
        gate_state_inside["unlocked"] = False
        return {"opened": False, "anomaly": False, "note": note}

    gate_state_inside["unlocked"] = True
    was_inside = occupancy.pop(user_id, None) is not None

    # Decide the snapshot prefix up front instead of saving once with
    # prefix="exit" and then, for the anomaly case, saving AGAIN with
    # prefix="exit_unmarked" and overwriting event["snapshot"] with the
    # new path. That second save doesn't replace the first file — it's a
    # brand-new image on disk — so the "exit_..." snapshot from that same
    # instant becomes an orphaned file nothing ever references again,
    # plus a wasted image encode/write on every exit anomaly.
    note = None
    anomaly = not was_inside
    snapshot_prefix = "exit_unmarked" if anomaly else "exit"
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": name,
        "granted": True,
        "confidence": confidence,
        "camera": "CAM 02 (Exit)",
        "snapshot": fe.save_face_snapshot(frame, bbox, prefix=snapshot_prefix, emotion=emotion),
        "emotion": emotion,
    }
    if anomaly:
        note = "Exited but wasn't marked as inside — check occupancy records."
        event["anomaly"] = True
        event["note"] = note
    gate_events.insert(0, event)

    threading.Thread(target=_auto_relock_after_delay, args=(gate_state_inside,), daemon=True).start()
    # Note: exit anomalies (not was_inside) still open the gate — flagged
    # for review, but not blocked like entry's anti-passback case.
    return {"opened": True, "anomaly": not was_inside, "note": note}


def handle_cctv_event(frame, detection, camera_label):
    """Called continuously from a camera's background thread (not just on
    a manual 'Detect now' click) whenever a face is seen and its
    per-name cooldown has elapsed. Always keeps a snapshot — known or
    unknown — so the CCTV log and person-profile view always have a
    picture of the moment someone passed through."""
    snapshot = fe.save_face_snapshot(
        frame, detection.get("bbox"),
        prefix="cctv_known" if detection["known"] else "cctv_unknown",
        emotion=detection.get("emotion"),
    )
    cctv_events.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": detection["name"],
        "known": detection["known"],
        "camera": camera_label,
        "location": cctv_location["label"],
        "snapshot": snapshot,
        "emotion": detection.get("emotion"),
    })


def handle_multi_person_warning(frame, count, camera_label):
    """Called when 2+ full people are seen at a gate camera at once.
    Nobody is granted access this pass — logged as a warning so it shows
    up clearly in the gate log instead of silently doing nothing."""
    snapshot = fe.save_face_snapshot(frame, None, prefix="multi_person")
    gate_events.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": "Multiple people detected",
        "granted": False,
        "confidence": None,
        "camera": camera_label,
        "anomaly": True,
        "note": f"{count} people were in frame at once — access denied for everyone. "
                "Only one person may present to the camera at a time.",
        "snapshot": snapshot,
    })


def handle_camera_gate_event(user_id, name, granted, confidence, frame=None, bbox=None, emotion=None):
    mode = get_camera_mode()
    if mode == "entry":
        return handle_entry_event(user_id, name, granted, confidence, frame, bbox, emotion)
    elif mode == "exit":
        return handle_exit_event(user_id, name, granted, confidence, frame, bbox, emotion)
    return None


def handle_camera_cctv_event(frame, detection):
    # No mode check here on purpose: shared_camera only calls this at all
    # while shared_camera.cctv_enabled is True (see face_engine.py), which
    # is turned on/off explicitly by /api/cctv/enter and /api/cctv/leave
    # below — NOT by camera_mode, which flips any time a video feed / poll
    # route fires and previously caused CCTV to log faces even while you
    # were on the Entry or Exit tab.
    handle_cctv_event(frame, detection, "CCTV")


def handle_camera_multi_person(frame, count):
    mode = get_camera_mode()
    if mode == "entry":
        handle_multi_person_warning(frame, count, "Entry Gate")
    elif mode == "exit":
        handle_multi_person_warning(frame, count, "Exit Gate")


shared_camera.on_gate_event = handle_camera_gate_event
shared_camera.on_cctv_event = handle_camera_cctv_event
shared_camera.on_multi_person = handle_camera_multi_person


def ensure_camera_running(mode=None):
    if mode:
        set_camera_mode(mode)
    if not shared_camera.is_running():
        shared_camera.start()


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        return view(*args, **kwargs)
    return wrapped


# -----------------------------
# Pages
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Auth
# -----------------------------

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["logged_in"] = True
        session["username"] = username
        return jsonify({"ok": True, "username": username})

    return jsonify({"ok": False, "error": "Invalid username or password"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/session")
def session_check():
    return jsonify({"ok": True, "logged_in": bool(session.get("logged_in")),
                     "username": session.get("username")})


# -----------------------------
# Registration
# -----------------------------

def _build_quality_response(result, slot):
    """Shared response shape for both quality-check paths (uploaded file,
    or a live frame pulled from the shared camera) — see fe.assess_photo_quality
    for what `result` contains. `slot` is "" for the plain Upload Image
    check, or one of fe.CAPTURE_SLOTS for the Live Capture distance guide."""
    response = {
        "ok": result["ok"],
        "reasons": list(result["reasons"]),
        "face_count": result["face_count"],
        "face_ratio": result["face_ratio"],
        "sharpness": result["sharpness"],
    }
    if slot in fe.CAPTURE_SLOTS:
        if result["face_count"] == 1 and result["face_ratio"] is not None:
            distance_status = fe.classify_capture_distance(result["face_ratio"], slot)
            response["distance_status"] = distance_status
            if distance_status == "too_far":
                response["ok"] = False
                response["reasons"].append("Move slightly closer for this shot.")
            elif distance_status == "too_close":
                response["ok"] = False
                response["reasons"].append("Move slightly farther away for this shot.")
        else:
            response["distance_status"] = "no_face"
    return response


@app.route("/api/register/quality-check", methods=["POST"])
@login_required
def register_quality_check():
    """Validates a single candidate registration photo *uploaded from
    disk* without storing anything — used by the Upload Image path.
    Runs fe.assess_photo_quality with extra_checks=True: the usual 4
    checks (exactly one face, face detectable, image clear enough, face
    large enough in frame) PLUS color/black&white and front-facing
    checks that only make sense for files picked from disk — Live
    Capture doesn't run those (it already has its own real-time framing
    guide and a same_person cross-check instead). The facial-feature-
    visibility check (masked/covered nose or mouth) is the exception:
    it's part of extra_checks here too, but Live Capture also requires
    it separately via check_feature_visibility — see register_live_check
    / register_live_capture below.
    """
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"ok": False, "error": "No image provided."}), 400
    image = fe.decode_image_from_bytes(photo.read())
    if image is None:
        return jsonify({"ok": False, "error": "Could not read that image."}), 400

    result = fe.assess_photo_quality(image, extra_checks=True)
    slot = (request.form.get("slot") or "").strip().lower()
    return jsonify(_build_quality_response(result, slot))


@app.route("/video_feed_register")
@login_required
def video_feed_register():
    """MJPEG preview for the Live Capture panel on the Register page.

    There's only one physical webcam in this prototype, already owned
    by shared_camera (see the module docstring). A browser can't also
    open that device directly via getUserMedia — the OS/driver won't
    allow two exclusive claims on one camera — so Live Capture instead
    just watches this same server-side stream, exactly like the Entry/
    Exit/CCTV tabs already do.

    Deliberately does NOT call set_camera_mode() the way the gate feeds
    do: registering someone shouldn't silently flip the dashboard's
    Entry/Exit/CCTV mode out from under whoever's watching those tabs.
    It only makes sure the shared camera thread is running.
    """
    try:
        if not shared_camera.is_running():
            shared_camera.start()
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    # annotate=False: this is just a framing preview for the person
    # registering — no gate-style bounding box / name / confidence label,
    # and no "Persons detected: N" box. The Live Capture panel already
    # draws its own guide outline + slot label in the browser (see
    # index.html), so the server-side overlay would land right on top of
    # those and read as garbled double text.
    return Response(fe.mjpeg_generator(shared_camera, annotate=False),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/register/live/check")
@login_required
def register_live_check():
    """Polled continuously (no image upload — see video_feed_register)
    while the user positions themselves for a Live Capture slot. Grabs
    whatever frame the shared camera currently has and checks: exactly
    one face in frame, and the facial-feature-visibility check (rejects
    a masked/covered nose or mouth) — no blur or min-face-size check
    here, since (given ?slot=normal|far|close) the per-slot distance
    verdict below is what actually drives the guide's red/yellow/green
    state and framing requirement."""
    frame = shared_camera.get_latest_frame()
    if frame is None:
        return jsonify({"ok": False, "reasons": ["Camera isn't ready yet."],
                         "face_count": 0, "face_ratio": None, "sharpness": None,
                         "distance_status": "no_face"})
    result = fe.assess_photo_quality(frame, min_face_ratio=0, min_sharpness=0, check_feature_visibility=True)
    slot = (request.args.get("slot") or "").strip().lower()
    return jsonify(_build_quality_response(result, slot))


@app.route("/api/register/live/capture", methods=["POST"])
@login_required
def register_live_capture():
    """Grabs the CURRENT frame from the shared camera, re-validates it
    (defense in depth — the frontend only enables the Capture button
    once /api/register/live/check last reported "correct", but frames
    can change in the moment between that poll and the click), and on
    success returns the frame itself as a raw JPEG image (not JSON) so
    the frontend can hold onto it exactly like a captured photo file —
    it's later resent as one of the 3 "photos" to /api/register."""
    frame = shared_camera.get_latest_frame()
    if frame is None:
        return jsonify({"ok": False, "error": "Camera isn't ready yet."}), 503

    result = fe.assess_photo_quality(frame, min_face_ratio=0, min_sharpness=0, check_feature_visibility=True)
    slot = (request.form.get("slot") or "").strip().lower()
    response = _build_quality_response(result, slot)
    if not response["ok"]:
        return jsonify(response), 422

    # Live Capture takes 3 shots of (supposedly) the same person at three
    # distances. The first shot ("normal") sets the reference embedding
    # for this attempt; the later shots are compared against it (see
    # fe.same_person) rather than just trusted, so someone else stepping
    # into frame mid-flow does not get silently enrolled alongside them.
    embedding = result["face"].embedding
    if slot == "normal" or _live_capture_ref["embedding"] is None:
        with _live_capture_ref_lock:
            _live_capture_ref["embedding"] = embedding
    else:
        with _live_capture_ref_lock:
            reference = _live_capture_ref["embedding"]
        is_same, similarity = fe.same_person(embedding, reference)
        response["same_person"] = is_same
        response["same_person_similarity"] = round(similarity, 3)
        if not is_same:
            response["ok"] = False
            response["reasons"].append(
                "Doesn't look like the same person as your first shot -- "
                "make sure it's still you in frame, then try again."
            )
            return jsonify(response), 422

    jpeg_bytes = fe.encode_jpeg(frame)
    if jpeg_bytes is None:
        return jsonify({"ok": False, "error": "Could not encode the captured frame."}), 500

    resp = Response(jpeg_bytes, mimetype="image/jpeg")
    resp.headers["X-Face-Ratio"] = str(response.get("face_ratio"))
    return resp


MIN_REGISTRATION_PHOTOS = 3


@app.route("/api/register", methods=["POST"])
@login_required
def register():
    """Registers a person from exactly 3 photos (field name "photos", sent as
    multiple files under the same key). Those 3 photos can come from either
    registration path in index.html:
      - Live Capture: 3 frames grabbed from the browser's own webcam once
        each slot (normal / far / close) reported "correct" distance.
      - Upload Image: 3 files the user picked, each already validated
        against /api/register/quality-check client-side.

    `mode` (form field, "live" or "upload" — sent by index.html so this
    endpoint knows which path these photos came from) controls whether
    the extra Upload-Image-only checks run here too:
      - mode="upload": fe.assess_photo_quality(..., extra_checks=True)
        (adds color/black&white, front-facing, and facial-feature-
        visibility checks), PLUS a duplicate-photo check across all 3
        accepted photos (catches the same file being reused for more
        than one of the 3 slots).
      - mode="live" (or missing): fe.assess_photo_quality(..., min_face_ratio=0,
        min_sharpness=0, check_feature_visibility=True) — this mirrors
        register_live_check / register_live_capture's relaxed checks
        exactly (no blur or min-face-size floor, only "exactly one
        face" + facial-feature-visibility with the looser Live
        threshold), since this is a re-check of photos that already
        passed those same checks during capture — using the stricter
        Upload-style defaults here would just re-reject good Live
        Capture photos.

    What happens to each photo either way, in order:
      1. Decoded from bytes into an image (fe.decode_image_from_bytes).
      2. Run through fe.assess_photo_quality as above. This is a
         server-side re-check — defense in depth — not a re-trust of
         whatever the browser already validated.
      3. That face's 512-d ArcFace embedding (face.embedding) — a
         numeric "fingerprint" of the face's geometry, not the pixels
         themselves — is extracted and kept; the photo itself is
         discarded (not saved to disk).
      4. Repeat for every uploaded photo. Photos that fail any check are
         skipped with a warning (not treated as a hard failure) so one
         bad photo doesn't block registration if the others are fine.

    All embeddings that were successfully extracted are stored together
    against this one person (fe.add_user) — see match_embedding for why
    several stored embeddings per person matches more reliably than one.
    At least one usable photo is required.
    """
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "Resident").strip()
    photos = request.files.getlist("photos") or ([request.files["photo"]] if request.files.get("photo") else [])
    mode = (request.form.get("mode") or "live").strip().lower()
    is_upload = mode == "upload"

    if not name:
        return jsonify({"ok": False, "error": "Name is required."}), 400
    if not photos:
        return jsonify({"ok": False, "error": "No photo uploaded."}), 400
    if len(photos) != MIN_REGISTRATION_PHOTOS:
        return jsonify({"ok": False, "error": "Exactly 3 photos are required: normal, far, and close distance."}), 400

    embeddings = []
    embedding_photo_nums = []  # which original photo (1-based) each embedding came from
    image_hashes = []          # fe.compute_image_hash() per accepted photo, same order/index — upload mode only
    warnings = []
    for i, photo in enumerate(photos, start=1):
        image_bytes = photo.read()
        image = fe.decode_image_from_bytes(image_bytes)
        if image is None:
            warnings.append(f"Photo {i}: could not read that image file — skipped.")
            continue
        if is_upload:
            result = fe.assess_photo_quality(image, extra_checks=True)
        else:
            # Mirror register_live_check / register_live_capture's relaxed
            # checks exactly — this is a re-check of photos that already
            # passed those, so using the stricter blur/size defaults here
            # would just re-reject good Live Capture photos.
            result = fe.assess_photo_quality(image, min_face_ratio=0, min_sharpness=0,
                                              check_feature_visibility=True)
        if not result["ok"]:
            warnings.append(f"Photo {i}: " + " ".join(result["reasons"]) + " — skipped.")
            continue
        embeddings.append(result["face"].embedding)
        embedding_photo_nums.append(i)
        if is_upload:
            image_hashes.append(result.get("image_hash"))

    # Duplicate-photo check — Upload Image mode only. Catches the same
    # picture being uploaded for more than one of the 3 slots (e.g. the
    # same file picked twice). Every accepted photo is compared against
    # every other one so any repeated pair is caught, not just adjacent
    # ones. Live Capture doesn't need this: its frames come straight from
    # the camera, one per slot, so there's nothing to accidentally reuse.
    if is_upload:
        duplicate_pairs = [
            (embedding_photo_nums[a], embedding_photo_nums[b])
            for a in range(len(image_hashes))
            for b in range(a + 1, len(image_hashes))
            if fe.images_are_duplicate(image_hashes[a], image_hashes[b])
        ]
        if duplicate_pairs:
            nums = ", ".join(f"{a} & {b}" for a, b in duplicate_pairs)
            return jsonify({
                "ok": False,
                "error": (
                    f"Photo {nums} look like the same picture -- please provide "
                    "3 different photos."
                ),
                "warnings": warnings,
            }), 422

    if not embeddings:
        return jsonify({
            "ok": False,
            "error": "No usable face was found in any of the photos.",
            "warnings": warnings,
        }), 422

    # Same-person cross-check across whichever photos actually made it
    # through the quality check above. This covers the Upload Image path
    # (which has no live per-shot check the way Live Capture does — see
    # register_live_capture) and also re-checks Live Capture's own 3
    # shots server-side as defense in depth. Every embedding is compared
    # against the first good one; if any photo looks like a different
    # person, the whole registration is rejected rather than quietly
    # blending two people's faces into one identity.
    if len(embeddings) >= 2:
        reference = embeddings[0]
        mismatched_photo_nums = []
        for emb, photo_num in zip(embeddings[1:], embedding_photo_nums[1:]):
            is_same, _similarity = fe.same_person(emb, reference)
            if not is_same:
                mismatched_photo_nums.append(photo_num)
        if mismatched_photo_nums:
            nums = ", ".join(str(n) for n in mismatched_photo_nums)
            return jsonify({
                "ok": False,
                "error": (
                    f"Photo {nums} doesn't look like the same person as the "
                    "others — make sure all 3 photos are of the same person "
                    "and try again."
                ),
                "warnings": warnings,
            }), 422

    record = fe.add_user(name, role, embeddings)

    recent_registrations.insert(0, {
        "id": record["id"],
        "name": name,
        "role": role,
        "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active": True,
        "photo_count": len(embeddings),
    })

    return jsonify({
        "ok": True, "id": record["id"], "name": name, "role": role,
        "photo_count": len(embeddings), "warnings": warnings,
    })


@app.route("/api/registrations")
@login_required
def registrations():
    return jsonify({"ok": True, "items": recent_registrations[:20]})


# -----------------------------
# People / access control
# -----------------------------

@app.route("/api/users")
@login_required
def users_list():
    items = fe.list_users()
    for u in items:
        u["inside"] = u["id"] in occupancy
    items.sort(key=lambda r: r["registered_at"], reverse=True)
    return jsonify({"ok": True, "items": items})


@app.route("/api/users/<user_id>/access", methods=["POST"])
@login_required
def users_set_access(user_id):
    """Enable/revoke this person's access. Revoking does NOT delete their
    face data — it just excludes them from matching (see face_engine),
    so they immediately stop being able to open the gate or being
    recognized on CCTV, without losing their registration."""
    data = request.get_json(silent=True) or {}
    active = bool(data.get("active", True))
    updated = fe.set_user_active(user_id, active)
    if not updated:
        return jsonify({"ok": False, "error": "User not found."}), 404

    for r in recent_registrations:
        if r.get("id") == user_id:
            r["active"] = active

    if not active:
        # A revoked person shouldn't still be tracked as "inside" either.
        occupancy.pop(user_id, None)

    return jsonify({"ok": True, "id": user_id, "active": active})


@app.route("/api/users/<user_id>", methods=["DELETE"])
@login_required
def users_delete(user_id):
    """Permanently removes this person and their face data."""
    deleted = fe.delete_user(user_id)
    if not deleted:
        return jsonify({"ok": False, "error": "User not found."}), 404

    occupancy.pop(user_id, None)
    global recent_registrations
    recent_registrations = [r for r in recent_registrations if r.get("id") != user_id]

    return jsonify({"ok": True, "id": user_id})


# -----------------------------
# Occupancy (who's currently inside)
# -----------------------------

@app.route("/api/occupancy")
@login_required
def occupancy_list():
    return jsonify({"ok": True, "items": list(occupancy.values())})


@app.route("/api/occupancy/<user_id>/exit", methods=["POST"])
@login_required
def occupancy_exit(user_id):
    """Manual 'they've left' step — this single camera can't tell entry
    from exit direction on its own, so clearing occupancy is an explicit
    admin action rather than automatic."""
    person = occupancy.pop(user_id, None)
    if person:
        gate_events.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": person["name"],
            "granted": True,
            "confidence": None,
            "camera": "CAM 01",
            "note": "Marked as exited by admin.",
        })
    return jsonify({"ok": True, "items": list(occupancy.values())})


# -----------------------------
# Live video + automatic scanning
# -----------------------------

@app.route("/api/camera/mode", methods=["POST"])
@login_required
def camera_mode_view():
    mode = request.form.get("mode") or (request.get_json(silent=True) or {}).get("mode")
    if mode not in ("entry", "exit", "cctv"):
        return jsonify({"ok": False, "error": "Mode must be entry, exit or cctv."}), 400
    try:
        ensure_camera_running(mode)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return jsonify({"ok": True, "mode": mode, "camera_index": fe.CAMERA_INDEX})

@app.route("/video_feed_entry")
@login_required
def video_feed_entry():
    try:
        ensure_camera_running("entry")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return Response(fe.mjpeg_generator(shared_camera),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/entry/latest")
@login_required
def entry_latest():
    try:
        ensure_camera_running("entry")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return jsonify({"ok": True, "result": shared_camera.get_latest_result(),
                     "gate_unlocked": gate_state["unlocked"]})




@app.route("/api/gate/state")
@login_required
def gate_state_view():
    return jsonify({"ok": True, "unlocked": gate_state["unlocked"]})


@app.route("/api/gate/unlock", methods=["POST"])
@login_required
def gate_unlock():
    gate_state["unlocked"] = True
    gate_events.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": "Manual override",
        "granted": True,
        "confidence": None,
        "camera": "CAM 01",
    })
    return jsonify({"ok": True, "unlocked": True})


@app.route("/api/gate/lock", methods=["POST"])
@login_required
def gate_lock():
    gate_state["unlocked"] = False
    return jsonify({"ok": True, "unlocked": False})


# -----------------------------
# Automation settings (Gate Control page)
# -----------------------------

@app.route("/api/settings", methods=["GET", "POST"])
@login_required
def gate_settings_view():
    if request.method == "GET":
        with gate_settings_lock:
            return jsonify({"ok": True, **gate_settings})

    data = request.get_json(silent=True) or {}

    with gate_settings_lock:
        if "auto_lock_enabled" in data:
            if not isinstance(data["auto_lock_enabled"], bool):
                return jsonify({"ok": False, "error": "auto_lock_enabled must be true/false."}), 400
            gate_settings["auto_lock_enabled"] = data["auto_lock_enabled"]

        if "auto_lock_delay" in data:
            try:
                delay = float(data["auto_lock_delay"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "auto_lock_delay must be a number."}), 400
            if not (3 <= delay <= 30):
                return jsonify({"ok": False, "error": "auto_lock_delay must be between 3 and 30 seconds."}), 400
            gate_settings["auto_lock_delay"] = delay

        return jsonify({"ok": True, **gate_settings})


# -----------------------------
# Live video + automatic scanning — INSIDE (exit) gate
# -----------------------------

@app.route("/video_feed_exit")
@login_required
def video_feed_exit():
    try:
        ensure_camera_running("exit")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return Response(fe.mjpeg_generator(shared_camera),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/exit/latest")
@login_required
def exit_latest():
    try:
        ensure_camera_running("exit")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return jsonify({"ok": True, "result": shared_camera.get_latest_result(),
                     "gate_unlocked": gate_state_inside["unlocked"]})


@app.route("/api/exit/state")
@login_required
def exit_state_view():
    return jsonify({"ok": True, "unlocked": gate_state_inside["unlocked"]})


@app.route("/api/exit/unlock", methods=["POST"])
@login_required
def exit_unlock():
    gate_state_inside["unlocked"] = True
    gate_events.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "name": "Manual override",
        "granted": True,
        "confidence": None,
        "camera": "CAM 02",
    })
    return jsonify({"ok": True, "unlocked": True})


@app.route("/api/exit/lock", methods=["POST"])
@login_required
def exit_lock():
    gate_state_inside["unlocked"] = False
    return jsonify({"ok": True, "unlocked": False})


# -----------------------------
# CCTV (many people expected, no unlock) — shares whichever gate camera
# is selected in the dropdown
# -----------------------------

@app.route("/api/cctv/enter", methods=["POST"])
@login_required
def cctv_enter():
    """Call this the moment the user actually opens/clicks into the CCTV
    section in the dashboard (not just when a feed/poll route happens to
    fire). Starts the shared camera if needed and turns CCTV face logging
    on. This is the ONLY thing that should turn logging on."""
    try:
        ensure_camera_running("cctv")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    shared_camera.enable_cctv()
    return jsonify({"ok": True, "cctv_enabled": True})


@app.route("/api/cctv/leave", methods=["POST"])
@login_required
def cctv_leave():
    """Call this when the user navigates away from / closes the CCTV
    section. Turns CCTV face logging back off; the shared camera keeps
    running for Entry/Exit, it just stops logging to the CCTV log."""
    shared_camera.disable_cctv()
    return jsonify({"ok": True, "cctv_enabled": False})


@app.route("/video_feed_cctv")
@login_required
def video_feed_cctv():
    try:
        ensure_camera_running("cctv")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return Response(fe.mjpeg_generator(shared_camera),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cctv/latest")
@login_required
def cctv_latest():
    return jsonify({"ok": True, "result": shared_camera.get_latest_result(), "camera": "CCTV"})


@app.route("/api/cctv/location", methods=["POST"])
@login_required
def cctv_location_view():
    global cctv_location
    data = request.get_json(silent=True) or {}
    key = data.get("location")
    labels = {"corner": "Corner", "corridor-b": "Corridor B"}
    if key not in labels:
        return jsonify({"ok": False, "error": "Location must be corner or corridor-b."}), 400
    cctv_location = {"key": key, "label": labels[key]}
    return jsonify({"ok": True, "location": cctv_location})


@app.route("/api/cctv/scan", methods=["POST"])
@login_required
def cctv_scan():
    # Compatibility endpoint. CCTV itself is live/automatic; no button is
    # required. This simply returns the latest result from the dedicated CCTV camera.
    try:
        ensure_camera_running("cctv")
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 503
    return jsonify({"ok": True, "result": shared_camera.get_latest_result(), "camera": "CCTV"})


# -----------------------------
# Debug / diagnostics
# -----------------------------

@app.route("/api/cameras")
@login_required
def cameras_view():
    return jsonify({
        "ok": True,
        "entry": {"label": CAMERA_LABEL_ENTRY, "index": fe.CAMERA_INDEX},
        "exit": {"label": CAMERA_LABEL_EXIT, "index": fe.CAMERA_INDEX},
        "cctv": {"label": "CCTV", "index": fe.CAMERA_INDEX},
        "mode": get_camera_mode(),
        "cctv_enabled": shared_camera.cctv_enabled,
        "shared_camera": True,
    })


@app.route("/api/gpu-status")
@login_required
def gpu_status():
    """Reports whether the face engine is actually running on GPU or CPU.
    Loads the model on first call if it hasn't been loaded yet."""
    fe.get_app()  # ensure the model (and its startup log line) has been loaded
    return jsonify({"ok": True, **fe.get_gpu_status()})


# -----------------------------
# Logs
# -----------------------------

@app.route("/api/logs")
@login_required
def logs():
    source = request.args.get("source", "gate")
    if source == "cctv":
        return jsonify({"ok": True, "items": cctv_events[:200]})
    return jsonify({"ok": True, "items": gate_events[:200]})


@app.route("/api/logs/profiles")
@login_required
def logs_profiles():
    """Groups gate + CCTV events by person instead of one flat timeline,
    so each registered person shows up once with their own scan history
    and photos attached — success and failure alike. Unrecognized faces
    are grouped too, but by CLUSTER rather than dumped into one shared
    'Unknown Visitor' bucket (see fe.match_unknown_cluster): each
    distinct stranger gets their own 'Unknown #N' profile, so a person
    who walks past the camera five times shows up as one profile with
    five scans, not five unrelated "Unknown" entries mixed in with
    everyone else's."""
    profiles = {}

    for u in fe.list_users():
        profiles[u["name"]] = {
            "id": u["id"], "name": u["name"], "role": u["role"],
            "active": u["active"], "inside": u["id"] in occupancy,
            "scans": [],
        }

    def bucket_for(name):
        if name not in profiles:
            profiles[name] = {"id": None, "name": name, "role": None,
                               "active": None, "inside": False, "scans": []}
        return profiles[name]

    for e in gate_events[:300]:
        bucket_for(e["name"])["scans"].append({
            "time": e["time"], "source": "gate", "camera": e.get("camera"),
            "granted": e.get("granted"), "confidence": e.get("confidence"),
            "anomaly": e.get("anomaly", False), "note": e.get("note"),
            "snapshot": e.get("snapshot"),
        })
    for e in cctv_events[:300]:
        bucket_for(e["name"])["scans"].append({
            "time": e["time"], "source": "cctv", "camera": e.get("camera"),
            "granted": e.get("known"), "confidence": None,
            "anomaly": False, "note": None, "location": e.get("location"), "snapshot": e.get("snapshot"),
        })

    items = list(profiles.values())
    for p in items:
        p["scans"].sort(key=lambda s: s["time"], reverse=True)
        p["scan_count"] = len(p["scans"])
        # Every captured photo for this person, most recent first — not
        # just the latest one — so the profile card can show all of them
        # (e.g. laid out around/under the circular avatar) instead of
        # only ever showing their single most recent snapshot.
        p["snapshots"] = [s["snapshot"] for s in p["scans"] if s.get("snapshot")][:12]
        p["last_snapshot"] = p["snapshots"][0] if p["snapshots"] else None
    items.sort(key=lambda p: p["scan_count"], reverse=True)

    return jsonify({"ok": True, "items": items})


if __name__ == "__main__":
    # threaded=True is required: the /video_feed connection stays open
    # streaming frames, so other requests need their own threads to still
    # get served at the same time.
    app.run(debug=True, threaded=True)