from __future__ import annotations

import csv
import glob
import math
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter


# =========================
# PyCharm config area
# =========================

# Fill in one or more model weights.
CONFIG_WEIGHTS = [
    r"C:\Users\10443\Desktop\access\Comparative\visdrone\AFRD-LCC-APFF\weights\best.pt",
]

# Fill in an image, image folder, glob pattern, or dataset yaml.
# Examples:
#   r"H:\data\VisDrone\images\val"
#   r"H:\data\VisDrone\images\val\000001.jpg"
#   r"H:\data\VisDrone\images\val\*.jpg"
#   r"H:\data\VisDrone\data.yaml"
CONFIG_SOURCE = r"H:\data_f\Visdrone_DET\VisDrone2019-DET\VisDrone2019-DET-yolo\data.yaml"
CONFIG_DATA_SPLIT = "train"  # used only when CONFIG_SOURCE is a dataset yaml

# Optional display names. Keep None to infer from weight paths.
CONFIG_MODEL_NAMES = None  # e.g. ["YOLO11", "Ours"]

CONFIG_OUT_DIR = r"paper_vis\speed_benchmark"
CONFIG_IMGSZ = 640
CONFIG_CONF = 0.25
CONFIG_IOU = 0.70
CONFIG_MAX_DET = 300
CONFIG_DEVICE = None  # None = auto, "0" = GPU 0, "cpu" = CPU
CONFIG_HALF = False  # set True for FP16 on CUDA
CONFIG_FUSE = True

# Speed test settings.
CONFIG_BATCH_SIZE = 1
CONFIG_WARMUP = 10
CONFIG_REPEAT = 1
CONFIG_MAX_IMAGES = 0  # 0 means use all images
CONFIG_CUDNN_BENCHMARK = True


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def add_repo_to_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: pyyaml. Install it with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {path}")
    return data


def resolve_dataset_yaml(yaml_path: Path, split: str) -> list[str]:
    yaml_path = resolve_path(yaml_path)
    data = load_yaml(yaml_path)
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    value = data.get(split)
    if value is None:
        fallback = "test" if split == "val" else "val"
        value = data.get(fallback)
    if value is None:
        raise ValueError(f"No '{split}' or fallback split found in dataset yaml: {yaml_path}")

    values = value if isinstance(value, list) else [value]
    sources = []
    for item in values:
        p = Path(str(item))
        sources.append(str(p if p.is_absolute() else root / p))
    return sources


def collect_images(source: str | list[str], split: str) -> list[Path]:
    sources = source if isinstance(source, list) else [source]
    images: list[Path] = []

    for item in sources:
        path = resolve_path(item)
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            images.extend(collect_images(resolve_dataset_yaml(path, split), split))
            continue

        item_str = str(item)
        if any(ch in item_str for ch in "*?[]"):
            pattern = item_str if Path(item_str).is_absolute() else str(resolve_path(item_str))
            candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
        elif path.is_dir():
            candidates = [p for p in path.rglob("*") if p.is_file()]
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                images.append(candidate.resolve())

    images = sorted(dict.fromkeys(images))
    if CONFIG_MAX_IMAGES and CONFIG_MAX_IMAGES > 0:
        images = images[:CONFIG_MAX_IMAGES]
    if not images:
        raise FileNotFoundError(f"No images found from source: {source}")
    return images


def infer_model_names(weights: list[str], names: list[str] | None) -> list[str]:
    if names is not None:
        if len(names) != len(weights):
            raise ValueError("CONFIG_MODEL_NAMES must have the same length as CONFIG_WEIGHTS.")
        return names

    inferred = []
    for weight in weights:
        path = Path(weight)
        inferred.append(path.parent.parent.name if path.parent.name == "weights" else path.stem)
    return inferred


def chunks(items: list[Path], size: int):
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def configure_torch() -> None:
    try:
        import torch

        if CONFIG_CUDNN_BENCHMARK and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def safe_float(value) -> float:
    return float(value) if value is not None else 0.0


def result_speed(result) -> tuple[float, float, float, float]:
    speed = result.speed or {}
    pre = safe_float(speed.get("preprocess"))
    infer = safe_float(speed.get("inference"))
    post = safe_float(speed.get("postprocess"))
    return pre, infer, post, pre + infer + post


def timed_predict(model, batch_paths: list[Path]) -> tuple[list, float]:
    source = [str(p) for p in batch_paths]
    source_arg = source[0] if len(source) == 1 else source

    predict_kwargs = {
        "source": source_arg,
        "imgsz": CONFIG_IMGSZ,
        "conf": CONFIG_CONF,
        "iou": CONFIG_IOU,
        "max_det": CONFIG_MAX_DET,
        "save": False,
        "verbose": False,
        "stream": False,
        "batch": len(batch_paths),
        "half": CONFIG_HALF,
    }
    if CONFIG_DEVICE is not None:
        predict_kwargs["device"] = CONFIG_DEVICE

    sync_cuda()
    start = perf_counter()
    results = model.predict(**predict_kwargs)
    sync_cuda()
    elapsed_ms = (perf_counter() - start) * 1000
    return list(results), elapsed_ms / max(1, len(batch_paths))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": mean(values) if values else 0.0,
        f"{prefix}_median_ms": median(values) if values else 0.0,
        f"{prefix}_p90_ms": percentile(values, 90),
        f"{prefix}_p95_ms": percentile(values, 95),
        f"{prefix}_min_ms": min(values) if values else 0.0,
        f"{prefix}_max_ms": max(values) if values else 0.0,
    }


def benchmark_one_model(model_name: str, weight: str, images: list[Path], out_dir: Path) -> dict:
    from ultralytics import YOLO

    print(f"\nLoading model: {model_name}")
    model = YOLO(weight)
    if CONFIG_FUSE:
        try:
            model.fuse()
        except Exception as exc:
            print(f"Fuse skipped: {exc}")

    warmup_batch = images[: max(1, min(CONFIG_BATCH_SIZE, len(images)))]
    print(f"Warmup: {CONFIG_WARMUP} iteration(s)")
    for _ in range(max(0, CONFIG_WARMUP)):
        timed_predict(model, warmup_batch)

    rows = []
    e2e_values = []
    pre_values = []
    infer_values = []
    post_values = []
    total_values = []

    print(f"Benchmarking {len(images)} image(s), repeat={CONFIG_REPEAT}, batch={CONFIG_BATCH_SIZE}")
    sample_index = 0
    for repeat_idx in range(max(1, CONFIG_REPEAT)):
        for batch_paths in chunks(images, CONFIG_BATCH_SIZE):
            results, e2e_ms = timed_predict(model, batch_paths)
            for image_path, result in zip(batch_paths, results):
                pre, infer, post, total = result_speed(result)
                sample_index += 1
                e2e_values.append(e2e_ms)
                pre_values.append(pre)
                infer_values.append(infer)
                post_values.append(post)
                total_values.append(total)
                rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "index": sample_index,
                        "image": str(image_path),
                        "e2e_ms": f"{e2e_ms:.4f}",
                        "preprocess_ms": f"{pre:.4f}",
                        "inference_ms": f"{infer:.4f}",
                        "postprocess_ms": f"{post:.4f}",
                        "ultralytics_total_ms": f"{total:.4f}",
                        "detections": len(result),
                    }
                )

    detail_csv = out_dir / f"{model_name}_per_image.csv"
    write_csv(detail_csv, rows)

    summary = {
        "model": model_name,
        "weight": weight,
        "images": len(images),
        "samples": len(e2e_values),
        "batch_size": CONFIG_BATCH_SIZE,
        "imgsz": CONFIG_IMGSZ,
        "device": "auto" if CONFIG_DEVICE is None else CONFIG_DEVICE,
        "half": CONFIG_HALF,
        "fuse": CONFIG_FUSE,
        **summarize(e2e_values, "e2e"),
        **summarize(pre_values, "preprocess"),
        **summarize(infer_values, "inference"),
        **summarize(post_values, "postprocess"),
        **summarize(total_values, "ultralytics_total"),
    }
    summary["e2e_fps"] = 1000.0 / summary["e2e_mean_ms"] if summary["e2e_mean_ms"] > 0 else 0.0
    summary["inference_fps"] = 1000.0 / summary["inference_mean_ms"] if summary["inference_mean_ms"] > 0 else 0.0
    summary["ultralytics_total_fps"] = (
        1000.0 / summary["ultralytics_total_mean_ms"] if summary["ultralytics_total_mean_ms"] > 0 else 0.0
    )

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Model: {summary['model']}")
    print(f"Images/Samples: {summary['images']} / {summary['samples']}")
    print(f"Batch/imgsz/device/half: {summary['batch_size']} / {summary['imgsz']} / {summary['device']} / {summary['half']}")
    print("-" * 72)
    print(f"End-to-end latency: {summary['e2e_mean_ms']:.2f} ms, FPS: {summary['e2e_fps']:.2f}")
    print(f"Inference only:     {summary['inference_mean_ms']:.2f} ms, FPS: {summary['inference_fps']:.2f}")
    print(f"YOLO total speed:   {summary['ultralytics_total_mean_ms']:.2f} ms, FPS: {summary['ultralytics_total_fps']:.2f}")
    print(f"P50/P90/P95 e2e:    {summary['e2e_median_ms']:.2f} / {summary['e2e_p90_ms']:.2f} / {summary['e2e_p95_ms']:.2f} ms")
    print(f"Pre/Infer/Post:     {summary['preprocess_mean_ms']:.2f} / {summary['inference_mean_ms']:.2f} / {summary['postprocess_mean_ms']:.2f} ms")
    print("=" * 72)


def main() -> None:
    add_repo_to_path()
    configure_torch()

    weights = [str(resolve_path(w)) for w in CONFIG_WEIGHTS]
    model_names = infer_model_names(weights, CONFIG_MODEL_NAMES)
    images = collect_images(CONFIG_SOURCE, CONFIG_DATA_SPLIT)
    out_dir = Path(CONFIG_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model_name, weight in zip(model_names, weights):
        summary = benchmark_one_model(model_name, weight, images, out_dir)
        summaries.append(summary)
        print_summary(summary)

    summary_csv = out_dir / "summary.csv"
    write_csv(summary_csv, summaries)
    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved per-image CSV files to: {out_dir}")


if __name__ == "__main__":
    main()
