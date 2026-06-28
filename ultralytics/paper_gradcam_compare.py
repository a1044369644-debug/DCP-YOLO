from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


CONFIG_IMAGE = r"H:\data_f\Visdrone_DET\VisDrone2019-DET\VisDrone2019-DET-yolo\images\val2017\0000295_02000_d_0000031.jpg"
CONFIG_OUT_DIR = r"paper_vis\heatmap_compare"
CONFIG_WEIGHTS = [
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\yolo11s\weights\best.pt",
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\AFRD-LCC-APFF\weights\best.pt",
]
CONFIG_NAMES = ["YOLO11s", "AFRD-LCC-APFF (Ours)"]
CONFIG_TARGET_LAYERS = [
    [16, 19, 22],
    [12, 14, 16],
]
CONFIG_IMGSZ = 640
CONFIG_TOPK = 100
CONFIG_ALPHA = 0.45
CONFIG_DEVICE = "0"
CONFIG_COLORMAP = cv2.COLORMAP_JET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready YOLO Grad-CAM heatmap comparison.")
    parser.add_argument("--image", default=CONFIG_IMAGE, help="Input image path.")
    parser.add_argument("--weights", nargs="+", default=CONFIG_WEIGHTS, help="Two or more YOLO .pt weights.")
    parser.add_argument("--names", nargs="+", default=CONFIG_NAMES, help="Panel names for weights.")
    parser.add_argument(
        "--target-layers",
        nargs="+",
        default=["16,19,22", "12,14,16"],
        help="Comma-separated layer ids for each weight, e.g. 16,19,22 12,14,16.",
    )
    parser.add_argument("--out-dir", default=CONFIG_OUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=CONFIG_IMGSZ, help="Letterboxed inference size.")
    parser.add_argument("--topk", type=int, default=CONFIG_TOPK, help="Top class-score anchors used as CAM target.")
    parser.add_argument("--alpha", type=float, default=CONFIG_ALPHA, help="Heatmap overlay alpha.")
    parser.add_argument("--device", default=CONFIG_DEVICE, help="CUDA device id, or cpu.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata for paper images.")
    return parser.parse_args()


def parse_target_layers(values: list[str], n: int) -> list[list[int]]:
    layers = [[int(x) for x in item.split(",") if x.strip()] for item in values]
    if len(layers) != n:
        raise ValueError("--target-layers must have the same count as --weights.")
    return layers


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
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


def letterbox_tensor(image_bgr: np.ndarray, imgsz: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h0, w0 = image_bgr.shape[:2]
    boxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image_bgr)
    h1, w1 = boxed.shape[:2]

    scale = min(imgsz / h0, imgsz / w0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    pad_x = max((w1 - new_w) // 2, 0)
    pad_y = max((h1 - new_h) // 2, 0)

    image_rgb = boxed[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).float().unsqueeze(0) / 255.0
    return tensor.to(device), (pad_x, pad_y, new_w, new_h)


def normalize01(cam: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= float(cam.min())
    denom = float(cam.max()) + eps
    return cam / denom


def class_scores_from_output(output: torch.Tensor | tuple | dict) -> torch.Tensor:
    """Return class scores that remain connected to the target feature layers."""
    if isinstance(output, (tuple, list)) and len(output) > 1 and isinstance(output[1], dict):
        preds = output[1]
        if "one2many" in preds:
            return preds["one2many"]["scores"].sigmoid()
        if "scores" in preds:
            return preds["scores"].sigmoid()

    pred = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if pred.ndim != 3 or pred.shape[1] <= 4:
        raise RuntimeError(f"Unexpected YOLO prediction shape: {tuple(pred.shape)}")
    return pred[:, 4:, :]


def gradcam_for_model(
    weight: str,
    image_bgr: np.ndarray,
    target_layer_ids: list[int],
    imgsz: int,
    topk: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    yolo = YOLO(weight)
    model = yolo.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Layer {layer_id} output is not a tensor.")
            activations[layer_id] = output
            output.retain_grad()

        return hook

    for layer_id in target_layer_ids:
        handles.append(model.model[layer_id].register_forward_hook(make_hook(layer_id)))

    tensor, (pad_x, pad_y, new_w, new_h) = letterbox_tensor(image_bgr, imgsz, device)
    try:
        output = model(tensor)
        class_scores = class_scores_from_output(output)
        anchor_scores = class_scores.max(dim=1).values.flatten()
        k = min(topk, int(anchor_scores.numel()))
        top_values, _ = torch.topk(anchor_scores, k=k)
        target = top_values.mean()
        target.backward()

        cams = []
        layer_stats = {}
        for layer_id, activation in activations.items():
            grad = activation.grad
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
            cam_np = cam[0, 0].detach().cpu().numpy()
            cam_np = normalize01(cam_np)
            cams.append(cam_np)
            layer_stats[f"layer_{layer_id}_max"] = float(cam_np.max())

        if not cams:
            raise RuntimeError(f"No CAM maps were generated for {weight}.")

        cam = normalize01(np.mean(np.stack(cams, axis=0), axis=0))
        cam = cam[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
        cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        cam = normalize01(cam)
        stats = {
            "target_score": float(target.detach().cpu()),
            "topk": float(k),
            "num_prediction_points": float(anchor_scores.numel()),
            **layer_stats,
        }
        return cam, stats
    finally:
        for handle in handles:
            handle.remove()


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * normalize01(cam)), CONFIG_COLORMAP)
    overlay = cv2.addWeighted(heat, alpha, image_bgr, 1.0 - alpha, 0)
    return overlay


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


def make_grid(panels: list[np.ndarray], pad: int = 12) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to compose.")
    tile_h = max(x.shape[0] for x in panels)
    tile_w = max(x.shape[1] for x in panels)
    grid_h = tile_h
    grid_w = len(panels) * tile_w + (len(panels) - 1) * pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        x0 = i * (tile_w + pad) + (tile_w - panel.shape[1]) // 2
        y0 = (tile_h - panel.shape[0]) // 2
        grid[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    return grid


def write_method_note(
    out_dir: Path,
    image_path: Path,
    weights: list[str],
    names: list[str],
    target_layers: list[list[int]],
    stats: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Grad-CAM 热力图生成说明",
        "",
        "## 图像与模型",
        f"- 原始图像：`{image_path}`",
    ]
    for name, weight, layers in zip(names, weights, target_layers):
        lines.append(f"- {name}：`{weight}`，目标层：`{layers}`")
    lines += [
        "",
        "## 生成方式",
        "1. 将输入图像按 YOLO 推理流程等比例 letterbox 到 640 x 640，并记录缩放与填充区域。",
        "2. 对每个模型，选择检测头之前的三个多尺度特征层作为 Grad-CAM 目标层。YOLO11s 使用 `[16, 19, 22]`，AFRD-LCC-APFF 使用 `[12, 14, 16]`，分别对应三个检测分支输入特征。",
        "3. 前向传播得到未经过 NMS 的检测输出，取所有预测点的最大类别置信度，并选取 Top-K 置信度作为可解释目标。本文脚本默认 `K=100`，目标函数为这些 Top-K 类别置信度的均值。",
        "4. 对该目标函数反向传播，计算每个目标层的梯度全局平均池化权重，并按 Grad-CAM 公式得到层级热力图：`ReLU(sum_k alpha_k A_k)`，其中 `A_k` 为第 `k` 个通道特征图，`alpha_k` 为对应梯度权重。",
        "5. 将三个尺度的热力图分别归一化并上采样到 640 x 640 后求平均，再去除 letterbox 填充并恢复到原图尺寸。",
        "6. 使用 JET 颜色映射叠加到原图：红色/黄色表示对检测置信度贡献较高的区域，蓝色表示贡献较低区域。",
        "",
        "## 论文图注示例",
        "图 X 展示了 YOLO11s 与本文 AFRD-LCC-APFF 模型在 VisDrone 场景中的多尺度 Grad-CAM 可视化结果。热力图由检测头前的三个尺度特征层共同生成，并以 Top-K 预测类别置信度为反向传播目标。相比 YOLO11s，AFRD-LCC-APFF 在密集小目标区域呈现更连续、更集中的响应，说明模型能够更充分地关注远距离车辆与行人等关键目标区域。",
        "",
        "## 运行统计",
    ]
    for name, item in stats.items():
        lines.append(f"- {name}: " + ", ".join(f"{k}={v:.6g}" for k, v in item.items()))
    (out_dir / "method_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.names) != len(args.weights):
        raise ValueError("--names must have the same count as --weights.")
    target_layers = parse_target_layers(args.target_layers, len(args.weights))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if str(args.device).lower() == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    image_path = Path(args.image)
    image_bgr = imread(image_path)

    panels = [add_title(image_bgr, "Original")]
    stats: dict[str, dict[str, float]] = {}
    imwrite(out_dir / f"{image_path.stem}_original.png", image_bgr, args.dpi)

    for name, weight, layers in zip(args.names, args.weights, target_layers):
        cam, model_stats = gradcam_for_model(weight, image_bgr, layers, args.imgsz, args.topk, device)
        overlay = overlay_cam(image_bgr, cam, args.alpha)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_cam.png", np.uint8(255 * cam), args.dpi)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_overlay.png", overlay, args.dpi)
        panels.append(add_title(overlay, name))
        stats[name] = model_stats

    grid = make_grid(panels, pad=max(12, math.ceil(image_bgr.shape[1] * 0.01)))
    imwrite(out_dir / f"{image_path.stem}_comparison.png", grid, args.dpi)
    write_method_note(out_dir, image_path, args.weights, args.names, target_layers, stats)

    print(f"Saved heatmap comparison to: {out_dir.resolve()}")
    print(f"Main figure: {(out_dir / f'{image_path.stem}_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
