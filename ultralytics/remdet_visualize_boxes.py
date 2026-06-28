from __future__ import annotations

import glob
import sys
from pathlib import Path


# =========================
# PyCharm config area
# =========================

# REMDET/MMYOLO config file.
CONFIG_FILE = r"C:\Users\10443\Desktop\access\Comparative\visdrone\remdet_s_visdrone\remdet_s-300e_visdrone.py"

# For paper figures, prefer the best checkpoint. Change to epoch_300.pth if you want the final epoch.
CONFIG_CHECKPOINT = r"C:\Users\10443\Desktop\access\Comparative\visdrone\remdet_s_visdrone\best_coco_bbox_mAP_epoch_190.pth"

# Image file, image folder, or glob pattern.
CONFIG_SOURCE = r"assets\1.jpg"

# If you have the REMDET source project, set it here, for example:
# CONFIG_PROJECT_DIR = r"C:\Users\10443\Desktop\RemDet-main"
# Leave None if REMDET/MMYOLO is already installed in your Python environment.
CONFIG_PROJECT_DIR = None

CONFIG_OUT_DIR = r"paper_vis\remdet_boxes"
CONFIG_MODEL_NAME = "REMDET-S"
CONFIG_DEVICE = "cuda:0"  # use "cpu" if you do not have CUDA in this environment
CONFIG_SCORE_THR = 0.25
CONFIG_LINE_WIDTH = 1
CONFIG_SHOW_LABELS = False
CONFIG_SHOW_SCORE = False
CONFIG_EXT = ".jpg"

CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    for base in (Path.cwd(), REPO_ROOT, SCRIPT_DIR):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def add_project_to_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if CONFIG_PROJECT_DIR:
        project_dir = resolve_path(CONFIG_PROJECT_DIR)
        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))


def load_runtime_deps():
    add_project_to_path()
    try:
        import cv2
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmdet.apis import inference_detector, init_detector
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Run this script in the REMDET/MMDetection environment. "
            "This .pth checkpoint cannot be loaded by ultralytics.YOLO directly."
        ) from exc

    try:
        from mmyolo.utils import register_all_modules

        register_all_modules()
    except Exception:
        pass

    return cv2, np, torch, Config, init_detector, inference_detector


def collect_images(source: str) -> list[Path]:
    source_path = resolve_path(source)
    source_str = str(source)

    if any(ch in source_str for ch in "*?[]"):
        pattern = source_str if Path(source_str).is_absolute() else str(resolve_path(source_str))
        candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
    elif source_path.is_dir():
        candidates = [p for p in source_path.rglob("*") if p.is_file()]
    else:
        candidates = [source_path]

    images = sorted({p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTS})
    if not images:
        raise FileNotFoundError(f"No images found from CONFIG_SOURCE: {source}")
    return images


def read_image(cv2, np, path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def save_image(cv2, path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def clean_test_pipeline(cfg) -> None:
    pipeline = None
    if hasattr(cfg, "test_pipeline"):
        pipeline = cfg.test_pipeline
    elif hasattr(cfg, "test_dataloader"):
        pipeline = cfg.test_dataloader.dataset.pipeline
    if pipeline is None:
        return

    cleaned = []
    for transform in pipeline:
        transform_type = str(transform.get("type", ""))
        if "LoadAnnotations" in transform_type or "LoadYOLOAnnotations" in transform_type:
            continue
        cleaned.append(transform)

    if hasattr(cfg, "test_pipeline"):
        cfg.test_pipeline = cleaned
    if hasattr(cfg, "test_dataloader"):
        cfg.test_dataloader.dataset.pipeline = cleaned


def build_model(Config, init_detector):
    cfg = Config.fromfile(str(resolve_path(CONFIG_FILE)))
    clean_test_pipeline(cfg)
    checkpoint = str(resolve_path(CONFIG_CHECKPOINT))
    return init_detector(cfg, checkpoint, device=CONFIG_DEVICE)


def extract_detections(result, np):
    if hasattr(result, "pred_instances"):
        pred = result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy().astype(int)
        return bboxes, scores, labels

    if isinstance(result, tuple):
        result = result[0]

    boxes = []
    scores = []
    labels = []
    if isinstance(result, list):
        for class_id, class_result in enumerate(result):
            if class_result is None or len(class_result) == 0:
                continue
            arr = np.asarray(class_result)
            boxes.append(arr[:, :4])
            scores.append(arr[:, 4])
            labels.append(np.full((arr.shape[0],), class_id, dtype=int))

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(labels)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    )
    return palette[class_id % len(palette)]


def draw_detections(cv2, image, bboxes, scores, labels):
    annotated = image.copy()
    h, w = annotated.shape[:2]
    line_width = max(1, CONFIG_LINE_WIDTH)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.8, (h + w) / 2400))
    font_thickness = max(1, line_width)

    keep = scores >= CONFIG_SCORE_THR
    for box, score, label in zip(bboxes[keep], scores[keep], labels[keep]):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_for_class(int(label))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        if CONFIG_SHOW_LABELS:
            cls_name = CLASSES[int(label)] if int(label) < len(CLASSES) else str(int(label))
            text = f"{cls_name} {score:.2f}" if CONFIG_SHOW_SCORE else cls_name
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            ty = max(y1, th + baseline + 3)
            cv2.rectangle(annotated, (x1, ty - th - baseline - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 2, ty - baseline),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return annotated


def main() -> None:
    cv2, np, _, Config, init_detector, inference_detector = load_runtime_deps()
    images = collect_images(CONFIG_SOURCE)
    model = build_model(Config, init_detector)

    out_dir = resolve_path(CONFIG_OUT_DIR) / CONFIG_MODEL_NAME
    for image_path in images:
        result = inference_detector(model, str(image_path))
        bboxes, scores, labels = extract_detections(result, np)
        image = read_image(cv2, np, image_path)
        annotated = draw_detections(cv2, image, bboxes, scores, labels)
        save_image(cv2, out_dir / f"{image_path.stem}{CONFIG_EXT}", annotated)

    print(f"Saved {len(images)} REMDET visualization image(s) to: {out_dir}")


if __name__ == "__main__":
    main()
