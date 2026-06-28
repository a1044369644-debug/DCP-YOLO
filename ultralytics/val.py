from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


def model_structure_text(model):
    """Return a parse_model-like structure table for a loaded YOLO model."""
    layers = getattr(model, "model", model)
    lines = [f"\n{'':>3}{'from':>20}{'params':>10}  {'module':<45}"]
    for i, module in enumerate(layers):
        f = getattr(module, "f", "")
        params = getattr(module, "np", sum(x.numel() for x in module.parameters()))
        module_type = getattr(module, "type", f"{module.__class__.__module__}.{module.__class__.__name__}")
        lines.append(f"{i:>3}{str(f):>20}{params:10.0f}  {module_type:<45}")
    return "\n".join(lines)


if __name__ == "__main__":
    imgsz = 640
    model = YOLO(r"C:\Users\10443\Desktop\access\ablation\AFRD-3\weights\best.pt")

    structure = model_structure_text(model.model)
    print(structure)

    params = get_num_params(model.model)
    gflops = get_flops(model.model, imgsz=imgsz)
    print(f"Model parameters: {params:,}")
    print(f"Model GFLOPs@{imgsz}: {gflops:.3f}")

    metrics = model.val(
        # data=r"H:\data_f\tinyperson\dataset.yaml",
        # data=r"H:\data_f\UAVDT\dataset.yaml",
        data=r"H:\data_f\Visdrone_DET\VisDrone2019-DET\VisDrone2019-DET-yolo\data.yaml",
        split="val",
        save_json=True,
        imgsz=imgsz,
    )

    info_file = Path(metrics.save_dir) / "model_info.txt"
    info_file.write_text(
        f"parameters: {params}\nGFLOPs@{imgsz}: {gflops:.6f}\n",
        encoding="utf-8",
    )

    structure_file = Path(metrics.save_dir) / "model_structure.txt"
    structure_file.write_text(structure + "\n", encoding="utf-8")

    print(f"Model info saved to {info_file}")
    print(f"Model structure saved to {structure_file}")
