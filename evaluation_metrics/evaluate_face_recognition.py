"""
AI Smart Gate System — Face Recognition Evaluation

Computes the metrics in Table 3.7 (Accuracy, Precision, Recall, F1, FAR,
FRR, Processing Time, FPS) by running face_engine.detect_faces() +
face_engine.match_embedding() against a labeled test set you provide.

Drop this file in the SAME folder as app.py / face_engine.py /
person_detector.py / emotion_engine.py (it imports face_engine directly,
same as app.py does), and make sure the face database you registered
your test subjects into (face_database.pkl) is present and reachable via
FACE_DB_PATH (or the default location next to face_engine.py).

--------------------------------------------------------------------------
EXPECTED TEST DATA STRUCTURE
--------------------------------------------------------------------------
Both of the following are accepted (and can be mixed):

test_data/
  registered/                 <- genuine attempts: people who ARE in the
                                  face database, used to measure FRR
    Alice/                    <- folder name must match that person's
      photo1.jpg                 "name" (or "id") in face_database.pkl,
      photo2.jpg                 case-insensitive
    Bob.jpg                   <- OR just drop a photo straight in, no
                                  subfolder needed; the filename (minus
                                  extension) is used as the label

  imposter/                   <- unauthorized attempts: people who are
                                  NOT in the face database, used to
                                  measure FAR
    stranger_1/                  folder name (or filename, if flat) is
      photo1.jpg                 just a label for your own bookkeeping,
    person2.jpg                  it isn't matched against anything

Tips for building this set:
  - Use photos/frames the model hasn't seen during registration (don't
    reuse the exact 3 registration photos) — otherwise FRR will look
    artificially low.
  - Vary angle/distance/lighting the way CAPTURE_SLOTS does (normal/far/
    close), and ideally pull some frames from an actual gate-camera clip
    rather than posed photos, since that's what the system sees in
    production.
  - "impostors" should be real people who were never registered — family/
    coworkers who agree to test, or a public face dataset — NOT sample
    photos of your registered users under a different name.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python evaluate_face_recognition.py --test-dir test_data
    python evaluate_face_recognition.py --test-dir test_data --threshold 0.6
    python evaluate_face_recognition.py --test-dir test_data --db-path /path/to/face_database.pkl

Outputs a results table to the console and a JSON report next to the
script (face_recognition_eval_report.json) with per-image predictions,
so you can inspect individual misses for your report/appendix.
"""

import argparse
import json
import os
import sys
import time

import cv2

def _add_face_engine_to_path():
    """face_engine.py normally lives in the main project folder
    (alongside app.py). This script can be run either from that same
    folder, or from a subfolder like evaluation_metrics/ — so check the
    script's own folder first, then walk up parent folders looking for
    face_engine.py, and add whichever one has it to sys.path."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(4):  # look up a few levels, no further
        if os.path.exists(os.path.join(candidate, "face_engine.py")):
            sys.path.insert(0, candidate)
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    # Not found anywhere — fall back to the script's own folder so the
    # import error below is at least informative about where it looked.
    sys.path.insert(0, here)
    return None


_found_at = _add_face_engine_to_path()
if _found_at is None:
    print("WARNING: couldn't find face_engine.py in this folder or any parent folder. "
          "Move this script into the same folder as face_engine.py, or place a copy "
          "of face_engine.py next to it.")
import face_engine as fe  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def _resolve_set_dir(test_dir, folder_name):
    """Case-insensitive lookup for a single expected folder name under
    test_dir (e.g. 'registered' or 'imposter'). Returns None if it
    doesn't exist."""
    if not os.path.isdir(test_dir):
        return None
    lower_map = {name.lower(): name for name in os.listdir(test_dir)}
    if folder_name.lower() in lower_map:
        return os.path.join(test_dir, lower_map[folder_name.lower()])
    return None


def _iter_people(set_dir):
    """Yields (person_label, [image_paths]) pairs from a set folder
    (e.g. registered/ or imposter/), supporting TWO layouts so you
    don't have to restructure whatever you already have:

      1. Per-person subfolders (recommended, supports multiple photos
         per person):
             imposter/person1/photo1.jpg
             imposter/person1/photo2.jpg

      2. Flat image files directly inside the set folder (one photo =
         one "person", filename used as the label):
             imposter/person1.jpg
             imposter/person2.jpg

    Both layouts can be mixed in the same folder.
    """
    if not os.path.isdir(set_dir):
        return
    for entry in sorted(os.listdir(set_dir)):
        full_path = os.path.join(set_dir, entry)
        if os.path.isdir(full_path):
            images = _list_images(full_path)
            if images:
                yield entry, images
        elif os.path.splitext(entry)[1].lower() in IMAGE_EXTS:
            label = os.path.splitext(entry)[0]
            yield label, [full_path]


def _record_matches_label(record, label):
    if record is None:
        return False
    label = label.strip().lower()
    return (
        str(record.get("name", "")).strip().lower() == label
        or str(record.get("id", "")).strip().lower() == label
    )


def run_probe(image_path, records, threshold):
    """Runs one probe image through detection + matching, timing it.
    Returns a dict describing what happened, or None if the image
    couldn't be read at all."""
    frame = cv2.imread(image_path) #read image
    if frame is None:
        print(f"  [skip] couldn't read image: {image_path}")
        return None

    t0 = time.perf_counter() #start counting the time
    faces = fe.detect_faces(frame) #detect all faces in the picture, returns a list
    face = fe.largest_face(faces) if faces else None # if have many face just pick the larger face
    record, similarity = (None, 0.0)
    if face is not None:
        # record = compare the person / similarity = check the rate of the similarity
        record, similarity = fe.match_embedding(face.embedding, records, threshold) #Use the picture to compare with the embedding data set
    t1 = time.perf_counter() # end counting time

    return {
        "image": image_path,
        "face_detected": face is not None,
        "matched_name": record.get("name") if record else None, # The system recognize this is who ?
        "matched_id": record.get("id") if record else None,
        "similarity": round(similarity, 4),
        "processing_time_ms": round((t1 - t0) * 1000, 2), #One picture use how many ms
    }


# Use picture to categorized the TP/FN/FP/TN
def evaluate(test_dir, db_path, threshold):
    records = fe.load_database(db_path)
    if not records:
        print(f"WARNING: no registered users found at {db_path}. "
              f"FRR/TP numbers will be meaningless until you register "
              f"your test subjects.")

    registered_dir = _resolve_set_dir(test_dir, "registered")
    impostors_dir = _resolve_set_dir(test_dir, "imposter")

    tp = fn = fp = tn = 0
    misidentifications = []  # genuine probe matched to the WRONG registered person
    all_results = []
    times_ms = []

    # --- Genuine attempts (registered folder) -> measures FRR / TP / FN ---
    if registered_dir:
        for person_label, images in _iter_people(registered_dir): #based on file naming person label
            print(f"[registered/{person_label}] {len(images)} image(s)")
            for img_path in images:
                result = run_probe(img_path, records, threshold)
                if result is None:
                    continue
                times_ms.append(result["processing_time_ms"])
                result["set"] = "registered"
                result["true_label"] = person_label 

                correct_match = ( #system identify the picture person name == the naming of the folder
                    result["face_detected"]
                    and result["matched_name"] is not None
                    and _record_matches_label(
                        {"name": result["matched_name"], "id": result["matched_id"]},
                        person_label,
                    )
                )
                if correct_match:
                    tp += 1
                    result["outcome"] = "TP" #guess correct true positive
                else:
                    fn += 1
                    result["outcome"] = "FN" #guess incorrect false negative
                    if result["matched_name"] is not None:
                        misidentifications.append(result)
                all_results.append(result)
    else:
        print(f"NOTE: no 'registered' folder found under {test_dir} — "
              f"skipping genuine-attempt testing.")

    # --- Impostor attempts (impostors folder) -> measures FAR / FP / TN ---
    if impostors_dir:
        for person_label, images in _iter_people(impostors_dir):
            print(f"[imposter/{person_label}] {len(images)} image(s)")
            for img_path in images:
                result = run_probe(img_path, records, threshold)
                if result is None:
                    continue
                times_ms.append(result["processing_time_ms"])
                result["set"] = "impostor"
                result["true_label"] = person_label

                #if the system got give any specific name then should be incorrect due to all is stranger
                falsely_accepted = result["matched_name"] is not None 
                if falsely_accepted:
                    fp += 1
                    result["outcome"] = "FP" #False Positive - The person accept the person should be false
                else:
                    tn += 1
                    result["outcome"] = "TN" #True Negative - the system reject the person should be true
                all_results.append(result)
    else:
        print(f"NOTE: no 'imposter' folder found under {test_dir} — "
              f"skipping impostor testing.")

    total = tp + tn + fp + fn
    genuine_attempts = tp + fn
    unauthorized_attempts = fp + tn

    def safe_div(a, b):
        return a / b if b else 0.0

    #Formula 
    accuracy = safe_div(tp + tn, total) #Accuracy of the percentage
    precision = safe_div(tp, tp + fp) #When the system said it is someone, the answer is correct
    recall = safe_div(tp, tp + fn) #The registered user have been recognized correctly
    f1 = safe_div(2 * precision * recall, precision + recall) #precision and recall balance
    far = safe_div(fp, unauthorized_attempts) #In stranger got how many person being access
    frr = safe_div(fn, genuine_attempts) #In the registered user got how many is accidentally to reject
    avg_time_ms = safe_div(sum(times_ms), len(times_ms)) #Avarage processisng time which is the total time of processing image / the total no.of image
    fps = safe_div(1000.0, avg_time_ms) #How many images/frames can be processed per second

    report = {
        "threshold": threshold,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "FAR": round(far, 4),
            "FRR": round(frr, 4),
            "avg_processing_time_ms": round(avg_time_ms, 2),
            "fps": round(fps, 2),
        },
        "counts": {
            "genuine_attempts": genuine_attempts,
            "unauthorized_attempts": unauthorized_attempts,
            "total_probes": total,
        },
        "misidentifications": misidentifications,
        "per_image_results": all_results,
    }
    return report

# Print out the result
def print_report(report):
    m = report["metrics"]
    cm = report["confusion_matrix"]
    print("\n" + "=" * 60)
    print("FACE RECOGNITION EVALUATION — Table 3.7 results")
    print("=" * 60)
    print(f"Threshold used         : {report['threshold']}")
    print(f"Genuine attempts        : {report['counts']['genuine_attempts']}")
    print(f"Unauthorized attempts   : {report['counts']['unauthorized_attempts']}")
    print(f"Confusion matrix         TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
    print("-" * 60)
    print(f"Accuracy                : {m['accuracy']*100:.2f}%")
    print(f"Precision                : {m['precision']*100:.2f}%")
    print(f"Recall                   : {m['recall']*100:.2f}%")
    print(f"F1 Score                 : {m['f1_score']*100:.2f}%")
    print(f"FAR                      : {m['FAR']*100:.2f}%")
    print(f"FRR                      : {m['FRR']*100:.2f}%")
    print(f"Avg Processing Time      : {m['avg_processing_time_ms']} ms")
    print(f"FPS                      : {m['fps']}")
    print("=" * 60)
    if report["misidentifications"]:
        print(f"\n{len(report['misidentifications'])} genuine probe(s) matched to the WRONG "
              f"person (counted as FN above, listed here for your appendix):")
        for r in report["misidentifications"]:
            print(f"  {r['image']}  true={r['true_label']}  matched={r['matched_name']}  "
                  f"sim={r['similarity']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition accuracy/FAR/FRR/speed.")
    parser.add_argument("--test-dir", default="test_data",
                         help="Folder containing registered/ and imposter/ subfolders (default: test_data)")
    parser.add_argument("--db-path", default=fe.DB_PATH,
                         help="Path to face_database.pkl (default: face_engine.DB_PATH)")
    parser.add_argument("--threshold", type=float, default=fe.DEFAULT_THRESHOLD,
                         help=f"Match threshold (default: {fe.DEFAULT_THRESHOLD}, same as the live gate)")
    parser.add_argument("--out", default="face_recognition_eval_report.json",
                         help="Where to save the detailed JSON report")
    args = parser.parse_args()

    report = evaluate(args.test_dir, args.db_path, args.threshold)
    print_report(report)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull per-image results saved to: {out_path}")


if __name__ == "__main__":
    main()