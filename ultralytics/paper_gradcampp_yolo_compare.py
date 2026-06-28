from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


CONFIG_IMAGE = r"C:\Users\10443\Desktop\access\img\01.jpg"
CONFIG_OUT_DIR = r"paper_vis\gradcampp_01_dcp"
CONFIG_WEIGHTS = [
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\yolo11s\weights\best.pt",
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\AFRD-LCC-APFF\weights\best.pt",
]
CONFIG_NAMES = ["YOLO11s", "DCP-YOLO"]
CONFIG_TARGET_LAYERS = ["16,19,22", "12,14,16"]
CONFIG_IMGSZ = 640
CONFIG_TOPK = 50
CONFIG_ALPHA = 0.45
CONFIG_DEVICE = "0"


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
        raise RuntimeError(f"Unexpected YOLO prediction shape for Grad-CAM++: {tuple(pred.shape)}")
    return pred[:, 4:, :]


class YoloGradCAMWrapper(torch.nn.Module):
    """Return YOLO raw predictions as a tensor for pytorch-grad-cam targets."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return class_scores_from_output(output)


class YoloTopKClassTarget:
    """Use the mean of top-K raw class scores as the detection explanation target."""

    def __init__(self, topk: int):
        self.topk = topk

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # pytorch-grad-cam iterates over the batch dimension, so this receives [C, N].
        if model_output.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO output shape for Grad-CAM++ target: {tuple(model_output.shape)}")
        scores = model_output.max(dim=0).values
        k = min(self.topk, int(scores.numel()))
        return torch.topk(scores, k=k).values.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO Grad-CAM++ comparison figure for papers.")
    parser.add_argument("--image", default=CONFIG_IMAGE, help="Input image path.")
    parser.add_argument("--weights", nargs="+", default=CONFIG_WEIGHTS, help="YOLO .pt weights.")
    parser.add_argument("--names", nargs="+", default=CONFIG_NAMES, help="Panel names for weights.")
    parser.add_argument("--target-layers", nargs="+", default=CONFIG_TARGET_LAYERS, help="Comma-separated layer ids per model.")
    parser.add_argument("--out-dir", default=CONFIG_OUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=CONFIG_IMGSZ, help="Letterboxed input size.")
    parser.add_argument("--topk", type=int, default=CONFIG_TOPK, help="Top-K class scores used as the target.")
    parser.add_argument("--alpha", type=float, default=CONFIG_ALPHA, help="Heatmap overlay alpha.")
    parser.add_argument("--device", default=CONFIG_DEVICE, help="CUDA device id, or cpu.")
    parser.add_argument("--title-scale", type=float, default=1.0, help="Multiplier for panel title font size.")
    parser.add_argument("--title-height-scale", type=float, default=1.0, help="Multiplier for panel title band height.")
    parser.add_argument("--aug-smooth", action="store_true", help="Use test-time augmentation smoothing in pytorch-grad-cam.")
    parser.add_argument("--eigen-smooth", action="store_true", help="Use PCA smoothing in pytorch-grad-cam.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI metadata for saved images.")
    return parser.parse_args()


def parse_layers(values: list[str], n: int) -> list[list[int]]:
    layers = [[int(x.strip()) for x in value.split(",") if x.strip()] for value in values]
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
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, dpi=(dpi, dpi))


def letterbox_tensor(image_bgr: np.ndarray, imgsz: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h0, w0 = image_bgr.shape[:2]
    boxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image_bgr)
    scale = min(imgsz / h0, imgsz / w0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    pad_x = max((boxed.shape[1] - new_w) // 2, 0)
    pad_y = max((boxed.shape[0] - new_h) // 2, 0)

    rgb_chw = boxed[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_chw)).float().unsqueeze(0) / 255.0
    return tensor.to(device), (pad_x, pad_y, new_w, new_h)


def crop_letterbox_and_resize(cam: np.ndarray, pad_info: tuple[int, int, int, int], out_shape: tuple[int, int]) -> np.ndarray:
    pad_x, pad_y, new_w, new_h = pad_info
    cam = cam[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    cam = cv2.resize(cam, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
    cam = cam.astype(np.float32)
    cam -= float(cam.min())
    denom = float(cam.max()) + 1e-7
    return cam / denom


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, image_bgr, 1.0 - alpha, 0)


def add_title(image: np.ndarray, title: str, title_scale: float = 1.0, title_height_scale: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    title_h = max(52, round(h * 0.065 * title_height_scale))
    canvas = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    canvas[title_h:, :] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(1.45, w / 1000)) * title_scale
    thickness = max(2, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = max(th + 4, (title_h + th) // 2)
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


def gradcampp_for_model(
    weight: str,
    image_bgr: np.ndarray,
    layer_ids: list[int],
    imgsz: int,
    topk: int,
    device: torch.device,
    aug_smooth: bool,
    eigen_smooth: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    yolo = YOLO(weight)
    model = yolo.model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    wrapper = YoloGradCAMWrapper(model).to(device).eval()
    input_tensor, pad_info = letterbox_tensor(image_bgr, imgsz, device)
    target_layers = [model.model[i] for i in layer_ids]
    targets = [YoloTopKClassTarget(topk)]

    with GradCAMPlusPlus(model=wrapper, target_layers=target_layers, use_cuda=device.type == "cuda") as cam_runner:
        grayscale_cam = cam_runner(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )[0]

    cam = crop_letterbox_and_resize(grayscale_cam, pad_info, image_bgr.shape[:2])

    with torch.no_grad():
        pred = wrapper(input_tensor)[0]
        scores = pred.max(dim=0).values
        k = min(topk, int(scores.numel()))
        top_values = torch.topk(scores, k=k).values
    return cam, {
        "topk": float(k),
        "target_mean_score": float(top_values.mean().detach().cpu()),
        "target_max_score": float(top_values.max().detach().cpu()),
        "target_layers": float(len(layer_ids)),
    }


def write_note(
    out_dir: Path,
    image_path: Path,
    weights: list[str],
    names: list[str],
    layers: list[list[int]],
    args: argparse.Namespace,
    stats: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Grad-CAM++ 热力图生成说明",
        "",
        "## 方法来源",
        "- Grad-CAM++：Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian, “Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks”, arXiv:1710.11063, https://arxiv.org/abs/1710.11063",
        "- Grad-CAM 基础方法：Ramprasaath R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”, ICCV 2017 / arXiv:1610.02391, https://arxiv.org/abs/1610.02391",
        "- 代码实现：`pytorch-grad-cam` 包中的 `GradCAMPlusPlus`。",
        "",
        "## 图像与模型",
        f"- 原始图像：`{image_path}`",
    ]
    for name, weight, layer_ids in zip(names, weights, layers):
        lines.append(f"- {name}：`{weight}`，目标层：`{layer_ids}`")

    lines += [
        "",
        "## 生成方式",
        f"1. 将原图按 YOLO 推理流程 letterbox 到 `{args.imgsz} x {args.imgsz}`，完成 Grad-CAM++ 后再去除填充并恢复到原图尺寸。",
        "2. 对检测模型，采用检测头前的三个多尺度特征层作为目标层；这三个层分别对应不同尺度检测分支的输入特征。",
        f"3. YOLO 检测没有单一分类 logit，因此目标函数定义为未 NMS 输出中 Top-{args.topk} 个最大类别分数的均值；两个模型使用相同目标定义。",
        "4. 使用 Grad-CAM++ 权重计算方式生成类判别热力图，并将多个目标层的结果平均融合。",
        f"5. 使用 JET colormap 叠加到原图，叠加强度 `alpha={args.alpha}`。红色/黄色表示对检测目标函数贡献更高的区域，蓝色表示贡献较低区域。",
        "",
        "## 论文图注示例",
        "图 X 展示了 YOLO11s 与 DCP-YOLO 在 VisDrone 场景中的 Grad-CAM++ 可视化结果。热力图基于检测头前多尺度特征层生成，并以未 NMS 预测中 Top-K 类别分数均值作为反向传播目标。高响应区域表示模型对检测置信度贡献较大的图像区域。",
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
    layers = parse_layers(args.target_layers, len(args.weights))
    device = torch.device("cpu" if str(args.device).lower() == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")

    image_path = Path(args.image)
    image = imread(image_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [add_title(image, "Original", args.title_scale, args.title_height_scale)]
    stats: dict[str, dict[str, float]] = {}
    imwrite(out_dir / f"{image_path.stem}_original.png", image, args.dpi)

    for name, weight, layer_ids in zip(args.names, args.weights, layers):
        cam, model_stats = gradcampp_for_model(
            weight,
            image,
            layer_ids,
            args.imgsz,
            args.topk,
            device,
            args.aug_smooth,
            args.eigen_smooth,
        )
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        overlay = overlay_cam(image, cam, args.alpha)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_gradcampp.png", np.uint8(255 * cam), args.dpi)
        imwrite(out_dir / f"{image_path.stem}_{safe_name}_overlay.png", overlay, args.dpi)
        panels.append(add_title(overlay, name, args.title_scale, args.title_height_scale))
        stats[name] = model_stats

    grid = make_grid(panels, pad=max(12, math.ceil(image.shape[1] * 0.01)))
    imwrite(out_dir / f"{image_path.stem}_gradcampp_comparison.png", grid, args.dpi)
    write_note(out_dir, image_path, args.weights, args.names, layers, args, stats)
    print(f"Saved Grad-CAM++ comparison to: {out_dir.resolve()}")
    print(f"Main figure: {(out_dir / f'{image_path.stem}_gradcampp_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
