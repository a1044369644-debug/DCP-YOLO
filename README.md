# DCP-YOLO

Source code for **DCP-YOLO: A Detail Calibration Pyramid Network for Small Object Detection in UAV Imagery**.

DCP-YOLO is a YOLO11-based detector designed for small-object detection in UAV imagery. The model improves small-object representation through a high-resolution `P2-P4` detection structure, detail-preserving downsampling, local coordinate calibration, and adaptive multi-scale feature fusion.

This repository is prepared for code availability. Trained weights, datasets, validation outputs, manuscript files, and visualization assets are not included.

## Overview

Small objects in UAV images are often only a few pixels wide and are affected by dense distributions, large scale variation, motion blur, occlusion, and complex backgrounds. Directly applying general-purpose detectors may lose fine spatial details during repeated downsampling, while fixed feature-pyramid fusion can be insufficient for dense small targets.

DCP-YOLO addresses these issues with:

- **High-resolution detection structure**: removes the deepest `P5` branch and performs detection on `P2`, `P3`, and `P4`.
- **AFRD-style downsampling**: implemented in code as `DCED`, preserving local detail, contextual, and high-frequency responses before downsampling.
- **LCC-C3k2-style backbone calibration**: implemented as `DGCA_C3k2`, strengthening local feature extraction, channel recalibration, and coordinate-aware spatial modeling.
- **APFF-style pyramid fusion**: implemented as `DMoE`, directly aligning and adaptively fusing `P2`, `P3`, and `P4` features at each output level.

## Main Results

On the VisDrone2019 validation set, the manuscript reports:

| Model | Params | FLOPs | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s | 9.42M | 21.4G | 39.9 | 23.7 |
| Modified YOLO11s | 3.11M | 24.4G | 43.8 | 26.8 |
| DCP-YOLO | 2.76M | 28.7G | 47.0 | 29.1 |

Additional experiments in the manuscript evaluate generalization on UAVDT and TinyPerson.

## Repository Structure

```text
DCP-YOLO/
|-- ultralytics/
|   |-- cfg/                  # Base Ultralytics model and dataset configs
|   |-- mycfg/                # DCP-YOLO and ablation model configs
|   |-- nn/modules/           # Custom modules: DCED, DGCA_C3k2, DMoE, etc.
|   |-- train.py              # Local training entry
|   `-- val.py                # Local validation/model-info entry
|-- examples/                 # Example scripts from the Ultralytics framework
|-- tests/                    # Test files
|-- predict.py
|-- pyproject.toml
`-- README.md
```

## Key Files

- `ultralytics/mycfg/yolo11-dced-dgca-dmoe-123.yaml`: DCP-YOLO configuration.
- `ultralytics/mycfg/yolo11-123.yaml`: Modified YOLO11s `P2-P4` baseline.
- `ultralytics/nn/modules/conv.py`: downsampling modules, including `DCED` and `SFDDown`.
- `ultralytics/nn/modules/block.py`: backbone feature modules, including `DGCA_C3k2`.
- `ultralytics/nn/modules/moe.py`: adaptive multi-scale fusion module `DMoE`.
- `ultralytics/optim/muon.py`: MuSGD-related optimizer implementation.

## Installation

Create a Python environment and install the project in editable mode:

```bash
git clone https://github.com/a1044369644-debug/DCP-YOLO.git
cd DCP-YOLO
pip install -e .
```

Install a PyTorch version suitable for your CUDA environment before training or validation.

## Dataset Preparation

The datasets used in the manuscript are public:

- VisDrone2019-DET: <https://github.com/VisDrone/VisDrone-Dataset>
- UAVDT: <https://sites.google.com/view/grli-uavdt>
- TinyPerson/TinyBenchmark: <https://github.com/ucas-vg/TinyBenchmark>

After downloading a dataset, prepare a YOLO-format `data.yaml` file and update the paths according to your local machine.

## Training

Example command for training DCP-YOLO from scratch:

```bash
yolo detect train \
  model=ultralytics/mycfg/yolo11-dced-dgca-dmoe-123.yaml \
  data=/path/to/your/data.yaml \
  imgsz=640 \
  epochs=300 \
  batch=16 \
  optimizer=MuSGD
```

Adjust `data`, `epochs`, `batch`, and device settings according to your dataset and GPU memory.

## Validation

Example command for validating a trained checkpoint:

```bash
yolo detect val \
  model=/path/to/best.pt \
  data=/path/to/your/data.yaml \
  imgsz=640 \
  split=val
```

The repository does not include trained weights. Use your own trained `best.pt` checkpoint for validation or inference.

## Model Configurations

The `ultralytics/mycfg/` directory contains the main configuration and ablation variants used during development. The most relevant files are:

| Config | Description |
| --- | --- |
| `yolo11-dced-dgca-dmoe-123.yaml` | Full DCP-YOLO model |
| `yolo11-123.yaml` | Modified YOLO11s baseline with `P2-P4` detection |
| `yolo11-ced-123.yaml` | Downsampling-related ablation |
| `yolo11-cgwt-123.yaml` | Feature-calibration-related ablation |
| `yolo11-dmoe-123.yaml` | Fusion-related ablation |

## Notes

- This codebase is derived from the Ultralytics framework and keeps its project structure.
- The released repository intentionally excludes weights, datasets, prediction outputs, `runs/`, manuscript files, and visualization assets.
- Some local helper scripts may contain machine-specific paths from the experimental environment. For reproduction, use the general training and validation commands above with your own paths.

## License

This repository follows the license included in `LICENSE`.
