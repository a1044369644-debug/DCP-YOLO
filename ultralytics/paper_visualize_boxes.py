from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
cv2 = None
np = None
YOLO = None
colors = None


# Edit this block when running directly from PyCharm.
# When no command-line arguments are provided, these values are used.
USE_CONFIG_WHEN_NO_ARGS = True
CONFIG_WEIGHTS = [
    r"C:\Users\10443\Desktop\access\Comparative\UAVDT\yolo11s-uavdt\weights\best.pt",
]
CONFIG_SOURCE = [
    r"C:\Users\10443\Desktop\img\UAVDT\original",
]
CONFIG_MODEL_NAMES = None
CONFIG_OUT_DIR = r"paper_vis\detection_boxes"
CONFIG_IMGSZ = 640
CONFIG_CONF = 0.25
CONFIG_IOU = 0.70
CONFIG_MAX_DET = 300
CONFIG_DEVICE = None  # e.g. "0" for GPU 0, or "cpu"
CONFIG_LABEL_DIR = None  # optional YOLO txt label folder for a Ground Truth panel
CONFIG_PANEL_WIDTH = 640
CONFIG_COLS = 0
CONFIG_LINE_WIDTH = 1
CONFIG_FONT_SIZE = None
CONFIG_SHOW_CONF = False
CONFIG_SHOW_LABELS = False
CONFIG_INCLUDE_ORIGINAL = True
CONFIG_SAVE_SINGLE = True
CONFIG_SAVE_GRID = True
CONFIG_EXT = ".jpg"


def load_runtime_deps() -> None:
    global YOLO, colors, cv2, np

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from ultralytics.utils.plotting import colors as _colors
    except ModuleNotFoundError as exc:
        missing = exc.name
        install_hint = "opencv-python" if missing == "cv2" else missing
        raise SystemExit(
            f"Missing dependency: {missing}. Activate your Ultralytics training environment, "
            f"or install the missing package, for example: pip install {install_hint}"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    colors = _colors


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 1 and USE_CONFIG_WHEN_NO_ARGS:
        return argparse.Namespace(
            weights=CONFIG_WEIGHTS,
            source=CONFIG_SOURCE,
            model_names=CONFIG_MODEL_NAMES,
            out_dir=CONFIG_OUT_DIR,
            imgsz=CONFIG_IMGSZ,
            conf=CONFIG_CONF,
            iou=CONFIG_IOU,
            max_det=CONFIG_MAX_DET,
            device=CONFIG_DEVICE,
            line_width=CONFIG_LINE_WIDTH,
            font_size=CONFIG_FONT_SIZE,
            hide_conf=not CONFIG_SHOW_CONF,
            hide_labels=not CONFIG_SHOW_LABELS,
            no_original=not CONFIG_INCLUDE_ORIGINAL,
            label_dir=CONFIG_LABEL_DIR,
            panel_width=CONFIG_PANEL_WIDTH,
            cols=CONFIG_COLS,
            pad=8,
            no_single=not CONFIG_SAVE_SINGLE,
            no_grid=not CONFIG_SAVE_GRID,
            ext=CONFIG_EXT,
        )

    parser = argparse.ArgumentParser(description="Create paper-style detection box visualizations.")
    parser.add_argument("--weights", nargs="+", required=True, help="One or more .pt model weights.")
    parser.add_argument("--source", nargs="+", required=True, help="Image file, directory, or glob pattern.")
    parser.add_argument("--model-names", nargs="+", default=None, help="Display names for each weight.")
    parser.add_argument("--out-dir", default="paper_vis/detection_boxes", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0, cpu, 0,1.")
    parser.add_argument("--line-width", type=int, default=None, help="Box line width. Auto if omitted.")
    parser.add_argument("--font-size", type=int, default=None, help="Label font size. Auto if omitted.")
    parser.add_argument("--hide-conf", action="store_true", help="Do not draw confidence values.")
    parser.add_argument("--hide-labels", action="store_true", help="Do not draw class labels.")
    parser.add_argument("--no-original", action="store_true", help="Do not include original image in comparison grid.")
    parser.add_argument("--label-dir", default=None, help="Optional YOLO txt label folder for a Ground Truth panel.")
    parser.add_argument("--panel-width", type=int, default=640, help="Panel width in grid. Use 0 to keep original size.")
    parser.add_argument("--cols", type=int, default=0, help="Grid columns. Default uses one row.")
    parser.add_argument("--pad", type=int, default=8, help="Grid padding in pixels.")
    parser.add_argument("--no-single", action="store_true", help="Do not save individual model images.")
    parser.add_argument("--no-grid", action="store_true", help="Do not save comparison grid images.")
    parser.add_argument("--ext", default=".jpg", choices=[".jpg", ".png"], help="Output image extension.")
    return parser.parse_args()


def infer_model_names(weights: list[str], names: list[str] | None) -> list[str]:
    if names is not None:
        if len(names) != len(weights):
            raise ValueError("--model-names must have the same length as --weights.")
        return names

    inferred = []
    for weight in weights:
        path = Path(weight)
        inferred.append(path.parent.parent.name if path.parent.name == "weights" else path.stem)
    return inferred


def collect_images(sources: list[str]) -> list[Path]:
    images: list[Path] = []
    for source in sources:
        if any(ch in source for ch in "*?[]"):
            candidates = [Path(p) for p in glob.glob(source, recursive=True)]
        else:
            path = Path(source)
            if path.is_dir():
                candidates = [p for p in path.rglob("*") if p.is_file()]
            else:
                candidates = [path]

        for path in candidates:
            if path.is_file() and path.suffix.lower() in IMG_EXTS:
                images.append(path)

    images = sorted(dict.fromkeys(images))
    if not images:
        raise FileNotFoundError(f"No images found from: {sources}")
    return images


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, round(image.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    title_h = max(36, round(image.shape[0] * 0.055))
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:, :] = image

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.2, image.shape[1] / 900))
    thickness = max(1, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(8, (image.shape[1] - tw) // 2)
    y = (title_h + th) // 2
    cv2.putText(canvas, title, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return canvas


def make_grid(panels: list[np.ndarray], cols: int, pad: int) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to compose.")
    cols = cols or len(panels)
    cols = max(1, min(cols, len(panels)))
    rows = math.ceil(len(panels) / cols)
    tile_h = max(panel.shape[0] for panel in panels)
    tile_w = max(panel.shape[1] for panel in panels)
    grid_h = rows * tile_h + (rows - 1) * pad
    grid_w = cols * tile_w + (cols - 1) * pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y0 = r * (tile_h + pad) + (tile_h - panel.shape[0]) // 2
        x0 = c * (tile_w + pad) + (tile_w - panel.shape[1]) // 2
        grid[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    return grid


def class_name(names: dict[int, str], cls: int) -> str:
    if isinstance(names, dict):
        return names.get(cls, str(cls))
    if isinstance(names, (list, tuple)) and 0 <= cls < len(names):
        return str(names[cls])
    return str(cls)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    # Fixed BGR palette for paper figures. Do not depend on each YOLO fork's default palette.
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


def draw_gt_boxes(image: np.ndarray, label_file: Path, names: dict[int, str]) -> np.ndarray:
    annotated = image.copy()
    if not label_file.is_file():
        return annotated

    h, w = annotated.shape[:2]
    line_width = max(round((h + w) * 0.0015), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(line_width / 3, 0.45)
    font_thickness = max(line_width - 1, 1)

    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = int(round((xc - bw / 2) * w))
        y1 = int(round((yc - bh / 2) * h))
        x2 = int(round((xc + bw / 2) * w))
        y2 = int(round((yc + bh / 2) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_for_class(cls)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        label = class_name(names, cls)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        ty = max(y1, th + baseline + 3)
        cv2.rectangle(annotated, (x1, ty - th - baseline - 3), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 2, ty - baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return annotated


def plot_result(result, args: argparse.Namespace) -> np.ndarray:
    annotated = result.orig_img.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return annotated

    h, w = annotated.shape[:2]
    line_width = args.line_width if args.line_width is not None else max(round((h + w) * 0.0015), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = args.font_size if args.font_size is not None else max(line_width / 3, 0.45)
    font_thickness = max(line_width - 1, 1)
    names = getattr(result, "names", {})

    xyxy = boxes.xyxy.detach().cpu().numpy()
    labels = boxes.cls.detach().cpu().numpy().astype(int)
    scores = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else [None] * len(labels)

    for box, cls, score in zip(xyxy, labels, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_for_class(int(cls))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        text_parts = []
        if not args.hide_labels:
            text_parts.append(class_name(names, int(cls)))
        if not args.hide_conf and score is not None:
            text_parts.append(f"{float(score):.2f}")
        if not text_parts:
            continue

        text = " ".join(text_parts)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        ty = max(y1, th + baseline + 3)
        cv2.rectangle(annotated, (x1, ty - th - baseline - 3), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(annotated, text, (x1 + 2, ty - baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return annotated


def main() -> None:
    args = parse_args()
    load_runtime_deps()
    weights = [str(Path(w)) for w in args.weights]
    model_names = infer_model_names(weights, args.model_names)
    images = collect_images(args.source)
    out_dir = Path(args.out_dir)

    models = [(name, YOLO(weight)) for name, weight in zip(model_names, weights)]
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "verbose": False,
        "save": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    for image_path in images:
        original = read_image(image_path)
        panels: list[np.ndarray] = []

        if not args.no_original:
            panels.append(add_panel_title(resize_to_width(original, args.panel_width), "Original"))

        first_names = models[0][1].names if models else {}
        if args.label_dir:
            gt_file = Path(args.label_dir) / f"{image_path.stem}.txt"
            gt_panel = draw_gt_boxes(original, gt_file, first_names)
            panels.append(add_panel_title(resize_to_width(gt_panel, args.panel_width), "Ground Truth"))

        for model_name, model in models:
            result = model.predict(source=str(image_path), **predict_kwargs)[0]
            annotated = plot_result(result, args)

            if not args.no_single:
                save_image(out_dir / "single" / model_name / f"{image_path.stem}{args.ext}", annotated)

            panels.append(add_panel_title(resize_to_width(annotated, args.panel_width), model_name))

        if not args.no_grid:
            grid = make_grid(panels, args.cols, args.pad)
            save_image(out_dir / "grid" / f"{image_path.stem}_compare{args.ext}", grid)

    print(f"Saved visualizations for {len(images)} image(s) to {out_dir}")


if __name__ == "__main__":
    main()
