"""YOLO person / prohibited-object detection (flow.txt §A.8, §A.9, §A.9b)."""

from video import models

PERSON_CLASS = 0


def _base_conf(obj_cfg):
    """YOLO's own cut-off: low enough for the weak detections of OB2 (0.25 = v1 behaviour)."""
    weak = obj_cfg.get("weak_conf")
    return min(0.25, weak) if weak else 0.25


def _boxes_from(result, model, dx=0.0, dy=0.0):
    out = []
    for r in result:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"cls": cls, "name": model.names[cls], "conf": float(b.conf[0]),
                        "xyxy": [x1 + dx, y1 + dy, x2 + dx, y2 + dy]})
    return out


def run_yolo(frame_bgr, obj_cfg):
    """One YOLO pass. Returns raw boxes [{"cls", "name", "conf", "xyxy"}] for persons + prohibited classes."""
    model = models.yolo(obj_cfg["weights"])
    wanted = [PERSON_CLASS] + list(obj_cfg["classes"].keys())
    res = model(frame_bgr, verbose=False, classes=wanted, conf=_base_conf(obj_cfg), imgsz=obj_cfg["imgsz"])
    return _boxes_from(res, model)


def roi_for_face(face_bbox, frame_shape, obj_cfg):
    """OB2 hands / chest / desk region around the face (x: -1.5w..+2.5w, y: -0.5h..+4h), or None."""
    if face_bbox is None:
        return None
    x, y, w, h = face_bbox
    fh, fw = frame_shape[:2]
    x1, y1 = max(0, int(x - 1.5 * w)), max(0, int(y - 0.5 * h))
    x2, y2 = min(fw, int(x + 2.5 * w)), min(fh, int(y + 4.0 * h))
    if x2 - x1 < 32 or y2 - y1 < 32 or (x2 - x1) * (y2 - y1) > obj_cfg.get("roi_max_frac", 0.6) * fw * fh:
        return None
    return x1, y1, x2, y2


def run_yolo_roi(frame_bgr, face_bbox, obj_cfg):
    """OB2 second pass on the ROI (prohibited classes only), boxes mapped back to frame coordinates."""
    roi = roi_for_face(face_bbox, frame_bgr.shape, obj_cfg)
    if roi is None:
        return []
    x1, y1, x2, y2 = roi
    model = models.yolo(obj_cfg["weights"])
    res = model(frame_bgr[y1:y2, x1:x2], verbose=False, classes=list(obj_cfg["classes"].keys()),
                conf=_base_conf(obj_cfg), imgsz=obj_cfg["imgsz"])
    return _boxes_from(res, model, dx=x1, dy=y1)


def merge_boxes(boxes, extra, iou_thr=0.5):
    """Per-class merge of two box lists: overlapping boxes (IoU >= iou_thr) keep the higher confidence."""
    out = list(boxes)
    for e in extra:
        dup = None
        for i, b in enumerate(out):
            if b["cls"] != e["cls"]:
                continue
            inter = _inter(b["xyxy"], e["xyxy"])
            union = _area(b["xyxy"]) + _area(e["xyxy"]) - inter
            if union > 0 and inter / union >= iou_thr:
                dup = i
                break
        if dup is None:
            out.append(e)
        elif e["conf"] > out[dup]["conf"]:
            out[dup] = e
    return out


def verify_class_ids(obj_cfg):
    """Fail loudly if the configured class IDs do not match the loaded model's names."""
    names = models.yolo(obj_cfg["weights"]).names
    expected = obj_cfg.get("model_names") or {PERSON_CLASS: "person", 63: "laptop", 67: "cell phone", 73: "book"}
    for cls in [PERSON_CLASS] + list(obj_cfg["classes"]):
        if cls not in names:
            raise ValueError(f"YOLO class {cls} does not exist in {obj_cfg['weights']}")
        if cls in expected and names[cls] != expected[cls]:
            raise ValueError(f"YOLO class {cls} is '{names.get(cls)}' in {obj_cfg['weights']}, "
                             f"expected '{expected[cls]}'")
    return {cls: names[cls] for cls in [PERSON_CLASS] + list(obj_cfg["classes"])}


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _inside_ratio(inner, outer):
    """Fraction of `inner` box area that lies inside `outer` (both xyxy)."""
    a = _area(inner)
    return _inter(inner, outer) / a if a > 0 else 0.0


def _contains_centre(box, f):
    x, y, w, h = f["bbox"]
    cx, cy = x + w / 2, y + h / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def dedupe_persons(boxes, iou_thr=0.6, contain_thr=0.8):
    """Greedy merge (highest confidence first) of person boxes that describe the same person."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b["conf"]):
        dup = False
        for k in kept:
            inter = _inter(b["xyxy"], k["xyxy"])
            union = _area(b["xyxy"]) + _area(k["xyxy"]) - inter
            small = min(_area(b["xyxy"]), _area(k["xyxy"]))
            if (union > 0 and inter / union >= iou_thr) or (small > 0 and inter / small >= contain_thr):
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def count_persons(boxes, faces, frame_shape, pc):
    """
    Returns (persons, yolo_n).
      v1 ("yolo"): YOLO count only.
      v2 ("fused"): max(deduped YOLO count, face count). `faces` may be an int (face count) or the face list;
      with the list, MP2 applies: static faces (f["static"]) are ignored, YOLO boxes between yolo_conf_low and
      yolo_conf count only with a live face inside, and boxes whose only face is static are dropped.
    """
    if pc["method"] == "yolo":
        n = count_persons_legacy(boxes, pc["yolo_conf"])
        return n, n
    fa = frame_shape[0] * frame_shape[1]
    big = [b for b in boxes if b["cls"] == PERSON_CLASS and _area(b["xyxy"]) >= pc["min_area_frac"] * fa]
    if isinstance(faces, int):
        persons = [b for b in big if b["conf"] >= pc["yolo_conf"]]
        yolo_n = len(dedupe_persons(persons, pc["dedupe_iou"]))
        return max(yolo_n, faces), yolo_n

    live = [f for f in faces if not f.get("static")]
    static = [f for f in faces if f.get("static")]
    low = pc.get("yolo_conf_low") or pc["yolo_conf"]
    persons = [b for b in big if b["conf"] >= pc["yolo_conf"]
               or (b["conf"] >= low and any(_contains_centre(b["xyxy"], f) for f in live))]
    persons = [b for b in persons
               if not (any(_contains_centre(b["xyxy"], f) for f in static)
                       and not any(_contains_centre(b["xyxy"], f) for f in live))]
    yolo_n = len(dedupe_persons(persons, pc["dedupe_iou"]))
    return max(yolo_n, len(live)), yolo_n


def objects_v2(boxes, obj_cfg, face_bbox=None):
    """
    Per-class confidence thresholds.
    Returns ({name: max confidence}, [{label, conf, xyxy}] for evidence, weak: bool).
    weak = a prohibited detection >= weak_conf but below its class threshold (OB2).
    Boxes lying >= ignore_inside_face inside the subject's face box are ignored (OB2).
    """
    found, kept, weak = {}, [], False
    face = None if face_bbox is None else [face_bbox[0], face_bbox[1],
                                           face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]]
    inside_thr = obj_cfg.get("ignore_inside_face")
    weak_conf = obj_cfg.get("weak_conf")
    for b in boxes:
        cls = b["cls"]
        if cls not in obj_cfg["classes"]:
            continue
        if face is not None and inside_thr and _inside_ratio(b["xyxy"], face) >= inside_thr:
            continue
        if b["conf"] >= obj_cfg["conf"][cls]:
            name = obj_cfg["classes"][cls]
            found[name] = max(found.get(name, 0.0), round(b["conf"], 3))
            kept.append({"label": name, "conf": round(b["conf"], 3), "xyxy": [round(v, 1) for v in b["xyxy"]]})
        elif weak_conf and b["conf"] >= weak_conf:
            weak = True
    return found, kept, weak


def count_persons_legacy(boxes, conf=0.5):
    return sum(1 for b in boxes if b["cls"] == PERSON_CLASS and b["conf"] > conf)


def objects_legacy(boxes, obj_cfg):
    """v1: any prohibited class above one shared confidence. Returns {name: count}."""
    found = {}
    for b in boxes:
        if b["cls"] in obj_cfg["classes"] and b["conf"] > obj_cfg["legacy_conf"]:
            name = obj_cfg["classes"][b["cls"]]
            found[name] = found.get(name, 0) + 1
    return found
