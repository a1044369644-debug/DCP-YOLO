from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO


CONFIG_IMAGE = r"H:\data_f\Visdrone_DET\VisDrone2019-DET\VisDrone2019-DET-yolo\images\val2017\0000129_02411_d_0000138.jpg"
CONFIG_OUT_DIR = r"paper_vis\heatmap_response_0000129"
CONFIG_WEIGHTS = [
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\yolo11s\weights\best.pt",
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\AFRD-LCC-APFF\weights\best.pt",
]
CONFIG_NAMES = ["YOLO11s", "AFRD-LCC-APFF (Ours)"]
CONFIG_IMGSZ = 640
CONFIG_CONF = 0.25
CONFIG_IOU = 0.70
CONFIG_MAX_DET = 300
CONFIG_DEVICE = "0"
CONFIG_ALPHA = 0.36
CONFIG_CLIP_PERCENTILE = 99.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready YOLO detection-response heatmap comparison.")
    parser.add_argument("--image", default=CONFIG_IMAGE, help="Input image path.")
    parser.add_argument("--weights", nargs="+", default=CONFIG_WEIGHTS, help="YOLO .pt weights.")
    parser.add_argument("--names", nargs="+", default=CONFIG_NAMES, help="Panel names for weights.")
    parser.add_argument("--out-dir", default=CONFIG_OUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=CONFIG_IMGSZ, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=CONFIG_CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=CONFIG_IOU, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=CONFIG_MAX_DET, help="Maximum detections.")
    parser.add_argument("--device", default=CONFIG_DEVICE, help="CUDA device id, or cpu.")
    parser.add_argument("--alpha", type=float, default=CONFIG_ALPHA, help="Heatmap overlay alpha.")
    parser.add_argument(
        "--overlay-mode",
        choices=["constant", "response"],
        default="response",
        help="constant reproduces the earlier full-color overlay; response keeps zero-response areas closer to the original image.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=CONFIG_CLIP_PERCENTILE,
        help="Joint percentile used as the shared color scale upper bound.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata for paper images.")
    return parser.parse_args()


def imread(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def imwrite(path: str | Path, image: np.ndarray, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image).save(path, dpi=(dpi, dpi))
    else:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, dpi=(dpi, dpi))


def add_gaussian(heatmap: np.ndarray, xyxy: np.ndarray, weight: float) -> None:
    x1, y1, x2, y2 = xyxy.astype(float)
    h, w = heatmap.shape[:2]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    sigma_x = min(max(bw / 3.0, 4.0), 80.0)
    sigma_y = min(max(bh / 3.0, 4.0), 80.0)
    radius_x = int(math.ceil(3.0 * sigma_x))
    radius_y = int(math.ceil(3.0 * sigma_y))

    xa = max(0, int(math.floor(cx - radius_x)))
    xb = min(w, int(math.ceil(cx + radius_x + 1)))
    ya = max(0, int(math.floor(cy - radius_y)))
    yb = min(h, int(math.ceil(cy + radius_y + 1)))
    if xa >= xb or ya >= yb:
        return

    xs = np.arange(xa, xb, dtype=np.float32)
    ys = np.arange(ya, yb, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    kernel = np.exp(-0.5 * (((grid_x - cx) / sigma_x) ** 2 + ((grid_y - cy) / sigma_y) ** 2))
    heatmap[ya:yb, xa:xb] += float(weight) * kernel.astype(np.float32)


def detection_heatmap(
    weight: str,
    image_path: Path,
    shape: tuple[int, int],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> tuple[np.ndarray, dict[str, float]]:
    model = YOLO(weight)
    result = model.predict(
        str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]

    heatmap = np.zeros(shape, dtype=np.float32)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return heatmap, {"detections": 0.0, "mean_conf": 0.0, "max_conf": 0.0}

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    for box, score in zip(xyxy, confs):
        add_gaussian(heatmap, box, float(score))

    return heatmap, {
        "detections": float(len(boxes)),
        "mean_conf": float(confs.mean()),
        "max_conf": float(confs.max()),
    }


def normalize_shared(heatmaps: list[np.ndarray], clip_percentile: float) -> list[np.ndarray]:
    values = np.concatenate([hm[hm > 0].reshape(-1) for hm in heatmaps if np.any(hm > 0)])
    if values.size == 0:
        return [hm.copy() for hm in heatmaps]
    upper = float(np.percentile(values, clip_percentile))
    upper = max(upper, 1e-7)
    return [np.clip(hm / upper, 0.0, 1.0).astype(np.float32) for hm in heatmaps]


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    if mode == "constant":
        return cv2.addWeighted(color, alpha, image_bgr, 1.0 - alpha, 0)
    alpha_map = (alpha * np.power(np.clip(heatmap, 0.0, 1.0), 0.65))[..., None]
    overlay = color.astype(np.float32) * alpha_map + image_bgr.astype(np.float32) * (1.0 - alpha_map)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    h, w = image.shape[:2]
    title_h = max(52, round(h * 0.065))
    canvas = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    canvas[title_h:, :] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(1.45, w / 1000))
    thickness = max(2, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = (title_h + th) // 2
    cv2.putText(canvas, title, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return canvas


def make_grid(panels: list[np.ndarray], pad: int) -> np.ndarray:
    tile_h = max(panel.shape[0] for panel in panels)
    tile_w = max(panel.shape[1] for panel in panels)
    grid = np.full((tile_h, len(panels) * tile_w + (len(panels) - 1) * pad, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        x0 = i * (tile_w + pad) + (tile_w - panel.shape[1]) // 2
        y0 = (tile_h - panel.shape[0]) // 2
        grid[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    return grid


def write_note(
    out_dir: Path,
    image_path: Path,
    weights: list[str],
    names: list[str],
    args: argparse.Namespace,
    stats: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# 检测响应热力图生成说明",
        "",
        "## 图像与模型",
        f"- 原始图像：`{image_path}`",
    ]
    for name, weight in zip(names, weights):
        lines.append(f"- {name}：`{weight}`")

    lines += [
        "",
        "## 生成方式",
        f"1. 使用相同推理参数运行两个模型：`imgsz={args.imgsz}`，`conf={args.conf}`，`iou={args.iou}`，`max_det={args.max_det}`。",
        "2. 对每个模型的 NMS 后检测框，以检测框中心为均值、框宽高的 1/3 为标准差生成二维高斯响应；每个高斯响应由该检测框置信度加权。",
        "3. 将同一模型全部检测框的高斯响应累加，得到该模型在原图坐标系下的检测响应热力图。",
        f"4. 两个模型热力图使用联合色阶归一化：取所有非零响应的第 `{args.clip_percentile}` 百分位作为共同上限，避免单个模型被单独归一化后造成视觉差异被放大。",
        f"5. 使用 JET colormap 叠加到原图，叠加模式为 `{args.overlay_mode}`，叠加强度 `alpha={args.alpha}`。红色/黄色表示检测响应更强，蓝色表示响应较弱。",
        "",
        "## 论文图注示例",
        "图 X 展示了 YOLO11s 与本文 AFRD-LCC-APFF 模型在 VisDrone 场景中的检测响应热力图。热力图由模型真实预测框的位置与置信度生成，并采用联合色阶归一化以保证两种模型具有可比的视觉尺度。相比 YOLO11s，AFRD-LCC-APFF 在道路车辆与人群密集区域保持更连续的响应，表明其对小尺度目标和复杂背景下的关键区域具有更稳定的检测关注。",
        "",
        "## 运行统计",
    ]
    for name, item in stats.items():
        lines.append(f"- {name}: " + ", ".join(f"{k}={v:.6g}" for k, v in item.items()))
    (out_dir / "method_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    if len(args.names) != len(args.weights):
        raise ValueError("--names must have the same count as --weights.")

    image_path = Path(args.image)
    image = imread(image_path)
    h, w = image.shape[:2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_heatmaps = []
    stats: dict[str, dict[str, float]] = {}
    for name, weight in zip(args.names, args.weights):
        heatmap, model_stats = detection_heatmap(
            weight,
            image_path,
            (h, w),
            args.imgsz,
            args.conf,
            args.iou,
            args.max_det,
            args.device,
        )
        raw_heatmaps.append(heatmap)
        stats[name] = model_stats

    heatmaps = normalize_shared(raw_heatmaps, args.clip_percentile)
    panels = [add_title(image, "Original")]
    imwrite(out_dir / f"{image_path.stem}_original.png", image, args.dpi)

    for name, heatmap in zip(args.names, heatmaps):
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        overlay = overlay_heatmap(image, heatmap, args.alpha, args.overlay_mode)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_heatmap.png", np.uint8(255 * heatmap), args.dpi)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_overlay.png", overlay, args.dpi)
        panels.append(add_title(overlay, name))

    grid = make_grid(panels, pad=max(12, math.ceil(w * 0.01)))
    imwrite(out_dir / f"{image_path.stem}_comparison.png", grid, args.dpi)
    write_note(out_dir, image_path, args.weights, args.names, args, stats)

    print(f"Saved detection-response heatmap comparison to: {out_dir.resolve()}")
    print(f"Main figure: {(out_dir / f'{image_path.stem}_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
