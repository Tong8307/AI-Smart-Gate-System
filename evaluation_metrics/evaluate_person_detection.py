"""
AI Smart Gate System — Person Detection Evaluation (counting accuracy)

Computes the metrics that don't require bounding-box
ground truth: Person Counting Accuracy, Processing Time, and FPS. It
runs person_detector.count_persons() against a labeled CSV of test
images/frames where you already know the true number of people who
should be counted "at the gate" (i.e. after the confidence, distance,
and ROI filters already baked into count_persons()).

Precision / Recall / F1 / mAP@0.5 / mAP@0.5:0.95 need per-box ground
truth (a box drawn around each real person, not just a headcount).
They ARE computed here, but only for images that have a matching
YOLO-format label file (see LABEL FILE FORMAT below) — for any image
in the CSV that has no label file, this script still runs its
counting-accuracy/timing/FPS check on that image, it just can't use it
for the box metrics. If NO image has a label file, the box metrics are
skipped entirely and only counting accuracy/timing/FPS are reported,
same as before.

--------------------------------------------------------------------------
LABEL FILE FORMAT (for Precision/Recall/F1/mAP)
--------------------------------------------------------------------------
Standard YOLO-format .txt, one per image, one line per real person
who should be detected AFTER distance+ROI filtering (i.e. the same
"at the gate" people counted in expected_count — not everyone visible
in the shot):

    <class_id> <x_center> <y_center> <width> <height>

all four numbers normalized 0-1 relative to the image's width/height,
class_id 0 = person (lines with any other class_id are ignored). This
is the format produced by LabelImg, CVAT, Roboflow, etc. when you
export "YOLO" annotations, and the format ultralytics itself expects
for model.val() (see the note near print_report()).

The evaluator looks for each image's labels at, in order:
  1. --labels-dir/<image_stem>.txt, if --labels-dir was given
  2. the same folder as the image, <image_stem>.txt
  3. the image's own path with an "images" path segment swapped for
     "labels" (the ultralytics convention: .../images/x.jpg ->
     .../labels/x.txt)
An image with none of these found is treated as "no boxes available"
and only contributes to counting accuracy/timing/FPS, not box metrics.

--------------------------------------------------------------------------
EXPECTED TEST DATA STRUCTURE
--------------------------------------------------------------------------
A CSV file (default name: ground_truth.csv) with two columns:

    image_path,expected_count
    test_frames/frame_0001.jpg,1
    test_frames/frame_0002.jpg,2
    test_frames/frame_0003.jpg,0
    test_frames/frame_0004.jpg,1

  - image_path: relative (to the CSV's own folder) or absolute path to
    a still frame. Grab these as individual frames from an actual gate
    camera recording (e.g. via `ffmpeg -i clip.mp4 -vf fps=1 frame_%04d.jpg`)
    rather than posed photos, since count_persons() depends heavily on
    the camera's real angle/distance for its distance + ROI filters.
  - expected_count: the TRUE number of people standing in the gate
    lane in that frame, i.e. what a human reviewing the frame would
    count as "at the gate" — NOT the total number of people visible
    anywhere in the shot (someone in the far background who SHOULD be
    filtered out by YOLO_MIN_PERSON_HEIGHT_RATIO/ROI counts as 0).

  NOTE: the exact file extension in image_path (.jpg vs .jpeg vs .png,
  and case) doesn't have to match the actual file on disk exactly —
  see resolve_image_path() below, which will try common alternates
  automatically and warn you if it had to guess.

Cover a mix of: 0 people, 1 person at various distances, 2+ people
side-by-side (the tailgating case this whole module exists for), and
someone visible only in the background (to confirm the distance/ROI
filters correctly exclude them).

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python evaluate_person_detection.py --csv ground_truth.csv

    # test against a specific ROI/height-ratio configuration instead of
    # whatever's currently in your environment variables:
    python evaluate_person_detection.py --csv ground_truth.csv \
        --min-height-ratio 0.6 --roi-points 0.42,0.3,0.58,0.3,0.85,1.0,0.15,1.0
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import person_detector as pd  # noqa: E402

# Extensions we'll try as fallbacks if the exact path in the CSV
# doesn't exist on disk (handles .jpg vs .jpeg vs .png vs case typos).
_FALLBACK_EXTENSIONS = [".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"]


def _parse_roi_arg(raw):
    if not raw:
        return None
    return pd._parse_roi_points(raw)


def resolve_image_path(img_path):
    """
    Return an existing file path for img_path, trying:
      1. the exact path as given
      2. case-insensitive match against files in the same folder
      3. same filename stem with a different common image extension
    Returns (resolved_path_or_None, note_string_or_None).
    """
    if os.path.isfile(img_path):
        return img_path, None

    folder = os.path.dirname(img_path) or "."
    stem = os.path.splitext(os.path.basename(img_path))[0]

    if not os.path.isdir(folder):
        return None, None

    # Case-insensitive / different-extension match against real files.
    candidates = glob.glob(os.path.join(folder, stem + ".*"))
    if candidates:
        # Prefer an exact case-insensitive stem match, first one found.
        chosen = candidates[0]
        return chosen, f"used '{os.path.basename(chosen)}' instead of '{os.path.basename(img_path)}'"

    # Last resort: try swapping in each fallback extension explicitly.
    for ext in _FALLBACK_EXTENSIONS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate, f"used '{os.path.basename(candidate)}' instead of '{os.path.basename(img_path)}'"

    return None, None


def find_label_path(image_path, labels_dir):
    """Return a YOLO-format .txt label path for image_path if one exists,
    trying --labels-dir, the image's own folder, then the ultralytics
    images/->labels/ convention. Returns None if none of those exist."""
    stem = os.path.splitext(os.path.basename(image_path))[0]

    if labels_dir:
        candidate = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(candidate):
            return candidate

    same_dir = os.path.join(os.path.dirname(image_path), stem + ".txt")
    if os.path.isfile(same_dir):
        return same_dir

    parts = image_path.replace("\\", "/").split("/")
    if "images" in parts:
        idx = parts.index("images")
        swapped = parts[:idx] + ["labels"] + parts[idx + 1:]
        swapped[-1] = os.path.splitext(swapped[-1])[0] + ".txt"
        candidate = "/".join(swapped)
        if os.path.isfile(candidate):
            return candidate

    return None


def load_yolo_labels(label_path, frame_w, frame_h, person_class_id=1):
    """Parse a YOLO-format label file into a list of pixel-space
    [x1, y1, x2, y2] boxes for the person class, ignoring any other
    class_id that might be present in the file."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = parts
            if int(float(cls_id)) != person_class_id:
                continue
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = (cx - w / 2) * frame_w
            y1 = (cy - h / 2) * frame_h
            x2 = (cx + w / 2) * frame_w
            y2 = (cy + h / 2) * frame_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_at_threshold(dets_sorted, gts_by_image, iou_threshold):
    """Greedy IoU matching, highest-confidence detection first, each
    ground-truth box usable at most once. dets_sorted is a list of
    (confidence, image_key, box) already sorted by confidence descending.
    Returns parallel tp/fp lists (1/0) aligned with dets_sorted."""
    matched = {img: [False] * len(boxes) for img, boxes in gts_by_image.items()}
    tp, fp = [], []
    for _, img_key, box in dets_sorted:
        gt_boxes = gts_by_image.get(img_key, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[img_key][j]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched[img_key][best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def compute_ap(dets_sorted, gts_by_image, iou_threshold, total_gt):
    """Average Precision at a single IoU threshold, via all-point
    (precision-envelope) interpolation of the precision/recall curve —
    the same continuous-interpolation approach used by VOC2012/COCO-style
    evaluators. Returns 0.0 if there is nothing to detect."""
    if total_gt == 0:
        return 0.0
    tp, fp = match_at_threshold(dets_sorted, gts_by_image, iou_threshold)

    precisions, recalls = [], []
    cum_tp = cum_fp = 0
    for t, f in zip(tp, fp):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_box_metrics(all_dets, gts_by_image, total_gt, pr_conf_threshold, pr_iou_threshold):
    """all_dets: list of (confidence, image_key, box) across every image
    that had a label file, in any order. gts_by_image: image_key -> list
    of ground-truth boxes. total_gt: total real people across those
    images. pr_conf_threshold/pr_iou_threshold: the single operating
    point used for Precision/Recall/F1 (mAP sweeps confidence itself so
    it doesn't need these)."""
    dets_sorted = sorted(all_dets, key=lambda d: d[0], reverse=True)

    # Precision / Recall / F1 at one fixed confidence + IoU operating point
    # (the production YOLO_PERSON_CONF by default, IoU 0.5 by default) —
    # this is "how good are detections at the settings the gate actually
    # runs with", as distinct from mAP's confidence-swept curve.
    op_dets = [d for d in dets_sorted if d[0] >= pr_conf_threshold]
    tp, fp = match_at_threshold(op_dets, gts_by_image, pr_iou_threshold)
    tp_count, fp_count = sum(tp), sum(fp)
    fn_count = total_gt - tp_count
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
    recall = tp_count / total_gt if total_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    map50 = compute_ap(dets_sorted, gts_by_image, 0.5, total_gt)
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
    map_5095 = sum(compute_ap(dets_sorted, gts_by_image, t, total_gt) for t in iou_thresholds) / len(iou_thresholds)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "map_0.5": map50,
        "map_0.5:0.95": map_5095,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "pr_operating_point": {"confidence": pr_conf_threshold, "iou": pr_iou_threshold},
    }


def load_ground_truth(csv_path):
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"].strip()
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, img_path)
            rows.append({
                "image_path": img_path,
                "expected_count": int(row["expected_count"]),
            })
    return rows


def evaluate(csv_path, min_height_ratio, roi_points, labels_dir=None,
             pr_conf_threshold=None, pr_iou_threshold=0.5):
    rows = load_ground_truth(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return None

    if not pd.is_available():
        print("WARNING: YOLO/ultralytics isn't available in this environment — "
              "count_persons() will fall back to face-count, which does NOT apply "
              "the distance/ROI filters and will not reflect production behavior. "
              "Install with: pip install ultralytics")

    if pr_conf_threshold is None:
        pr_conf_threshold = pd.YOLO_PERSON_CONF

    correct = 0
    times_ms = []
    per_image = []

    all_dets = []       # (confidence, image_key, box) across every labeled image
    gts_by_image = {}   # image_key -> [gt boxes]
    total_gt = 0
    labeled_image_count = 0

    for row in rows:
        resolved_path, note = resolve_image_path(row["image_path"])
        if resolved_path is None:
            print(f"  [skip] couldn't find image: {row['image_path']}")
            continue
        if note:
            print(f"  [note] {note}")

        frame = cv2.imread(resolved_path)
        if frame is None:
            print(f"  [skip] couldn't read image (unsupported/corrupt): {resolved_path}")
            continue

        t0 = time.perf_counter()
        count, boxes = pd.count_persons(
            frame,
            min_height_ratio=min_height_ratio,
            roi_points=roi_points,
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times_ms.append(elapsed_ms)

        is_correct = count == row["expected_count"]
        correct += int(is_correct)

        # --- box-based metrics (only for images with a label file) -----
        label_path = find_label_path(resolved_path, labels_dir)
        has_boxes = False
        if label_path:
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            gt_boxes = load_yolo_labels(label_path, frame_w, frame_h)
            gts_by_image[resolved_path] = gt_boxes
            total_gt += len(gt_boxes)
            labeled_image_count += 1
            has_boxes = True

            scored = pd.detect_persons_scored(
                frame, min_height_ratio=min_height_ratio, roi_points=roi_points,
            )
            if scored is not None:
                for det in scored:
                    all_dets.append((det["confidence"], resolved_path, det["box"]))

        result = {
            "image": resolved_path,
            "expected_count": row["expected_count"],
            "predicted_count": count,
            "correct": is_correct,
            "boxes": boxes,
            "processing_time_ms": round(elapsed_ms, 2),
            "has_box_ground_truth": has_boxes,
        }
        per_image.append(result)
        flag = "OK" if is_correct else "MISS"
        print(f"  [{flag}] {resolved_path}  expected={row['expected_count']} "
              f"predicted={count}  ({elapsed_ms:.1f} ms)")

    total = len(per_image)
    accuracy_pct = (correct / total * 100) if total else 0.0
    avg_time_ms = (sum(times_ms) / len(times_ms)) if times_ms else 0.0
    fps = (1000.0 / avg_time_ms) if avg_time_ms else 0.0

    metrics = {
        "person_counting_accuracy_pct": round(accuracy_pct, 2),
        "avg_processing_time_ms": round(avg_time_ms, 2),
        "fps": round(fps, 2),
    }

    box_metrics = None
    if labeled_image_count == 0:
        print("\nNo YOLO-format label files found for any test image — "
              "Precision/Recall/F1/mAP@0.5/mAP@0.5:0.95 skipped. See the "
              "LABEL FILE FORMAT note at the top of this script to add them.")
    elif not pd.is_available():
        print("\nYOLO unavailable — Precision/Recall/F1/mAP@0.5/mAP@0.5:0.95 "
              "skipped (they need scored YOLO detections, not the face-count fallback).")
    else:
        box_metrics = compute_box_metrics(
            all_dets, gts_by_image, total_gt, pr_conf_threshold, pr_iou_threshold,
        )
        metrics.update({
            "precision": round(box_metrics["precision"], 4),
            "recall": round(box_metrics["recall"], 4),
            "f1_score": round(box_metrics["f1_score"], 4),
            "map_0.5": round(box_metrics["map_0.5"], 4),
            "map_0.5:0.95": round(box_metrics["map_0.5:0.95"], 4),
        })

    report = {
        "config": {
            "min_height_ratio": min_height_ratio if min_height_ratio is not None else pd.YOLO_MIN_PERSON_HEIGHT_RATIO,
            "roi_points": roi_points if roi_points is not None else pd.YOLO_ROI_POINTS,
            "yolo_conf": pd.YOLO_PERSON_CONF,
            "yolo_iou": pd.YOLO_PERSON_IOU,
            "pr_operating_point": {"confidence": pr_conf_threshold, "iou": pr_iou_threshold},
        },
        "metrics": metrics,
        "counts": {
            "total_test_images": total,
            "correct_counts": correct,
            "incorrect_counts": total - correct,
            "labeled_images_with_boxes": labeled_image_count,
            "total_ground_truth_boxes": total_gt,
        },
        "box_metrics_detail": box_metrics,
        "per_image_results": per_image,
    }
    return report


def print_report(report):
    m = report["metrics"]
    c = report["counts"]
    box_detail = report.get("box_metrics_detail")

    print("\n" + "=" * 60)
    print("PERSON DETECTION EVALUATION")
    print("=" * 60)
    print(f"Total test images        : {c['total_test_images']}")
    print(f"Correct counts           : {c['correct_counts']}")
    print(f"Incorrect counts         : {c['incorrect_counts']}")
    print("-" * 60)
    print(f"Person Counting Accuracy : {m['person_counting_accuracy_pct']:.2f}%")
    print(f"Avg Processing Time      : {m['avg_processing_time_ms']} ms")
    print(f"FPS                      : {m['fps']}")

    if box_detail:
        op = box_detail["pr_operating_point"]
        print("-" * 60)
        print(f"Images with box ground truth : {c['labeled_images_with_boxes']} "
              f"({c['total_ground_truth_boxes']} ground-truth people)")
        print(f"Precision  (conf>={op['confidence']:.2f}, IoU>={op['iou']:.2f}) : "
              f"{m['precision']:.4f}  (TP={box_detail['true_positives']}, "
              f"FP={box_detail['false_positives']})")
        print(f"Recall     (conf>={op['confidence']:.2f}, IoU>={op['iou']:.2f}) : "
              f"{m['recall']:.4f}  (FN={box_detail['false_negatives']})")
        print(f"F1 Score                                    : {m['f1_score']:.4f}")
        print(f"mAP@0.5                                     : {m['map_0.5']:.4f}")
        print(f"mAP@0.5:0.95                                 : {m['map_0.5:0.95']:.4f}")
    print("=" * 60)

    if not box_detail:
        print("\nNOTE: Precision, Recall, F1, mAP@0.5 and mAP@0.5:0.95 were not "
              "computed this run — they require per-box ground truth "
              "(bounding boxes, not just a headcount). Add YOLO-format label "
              "files (see the LABEL FILE FORMAT note at the top of this "
              "script) and re-run. Alternatively, once you have such labels "
              "you can instead get mAP via ultralytics' own validator:\n\n"
              "    from ultralytics import YOLO\n"
              "    model = YOLO(person_detector.YOLO_MODEL_PATH)\n"
              "    metrics = model.val(data='your_dataset.yaml')\n"
              "    print(metrics.box.map50, metrics.box.map)\n\n"
              "though note that path evaluates YOLO in isolation and does not "
              "apply this system's distance/ROI filters the way this script's "
              "own Precision/Recall/F1/mAP above do.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate person-counting accuracy/speed.")
    parser.add_argument("--csv", default="ground_truth.csv",
                         help="CSV with image_path,expected_count columns (default: ground_truth.csv)")
    parser.add_argument("--min-height-ratio", type=float, default=None,
                         help="Override YOLO_MIN_PERSON_HEIGHT_RATIO for this run")
    parser.add_argument("--roi-points", type=str, default=None,
                         help="Override ROI_POINTS for this run, e.g. '0.42,0.3,0.58,0.3,0.85,1.0,0.15,1.0'")
    parser.add_argument("--out", default="person_detection_eval_report.json",
                         help="Where to save the detailed JSON report")
    parser.add_argument("--labels-dir", default=None,
                         help="Folder of YOLO-format <image_stem>.txt box labels for "
                              "Precision/Recall/F1/mAP. If omitted, the script also looks "
                              "next to each image and via the images/->labels/ convention.")
    parser.add_argument("--pr-conf-threshold", type=float, default=None,
                         help="Confidence operating point for Precision/Recall/F1 "
                              "(default: YOLO_PERSON_CONF, i.e. the production setting)")
    parser.add_argument("--pr-iou-threshold", type=float, default=0.5,
                         help="IoU match threshold for Precision/Recall/F1 (default: 0.5)")
    args = parser.parse_args()

    roi_points = _parse_roi_arg(args.roi_points)
    report = evaluate(
        args.csv, args.min_height_ratio, roi_points,
        labels_dir=args.labels_dir,
        pr_conf_threshold=args.pr_conf_threshold,
        pr_iou_threshold=args.pr_iou_threshold,
    )
    if report is None:
        return
    print_report(report)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full per-image results saved to: {out_path}")


if __name__ == "__main__":
    main()