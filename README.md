# DCP-YOLO

> Official source release for **Adaptive Detail Calibration for Small-Object Detection in Aerial Images**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

DCP-YOLO is a lightweight detector for small objects in UAV imagery. Built on YOLO11, it preserves fine spatial detail during downsampling, recalibrates local features with channel and coordinate cues, and adaptively fuses the high-resolution `P2`, `P3`, and `P4` feature levels.

This repository contains the paper-aligned PyTorch modules, the Ultralytics-compatible implementation used by the released checkpoints, and the supporting framework code. Datasets, experiment outputs, and trained weights are distributed separately.

## Highlights

- **High-resolution detection head** — predicts directly from `P2`, `P3`, and `P4` instead of using the deepest `P5` level.
- **AFRD** — Adaptive Feature Retention Downsampling combines local-detail, contextual, and high-frequency branches before space-to-depth reduction.
- **LCC-C3k2** — Local Coordinate Calibration improves local representation through dual-pooling channel attention and cross-axis spatial calibration.
- **APFF** — Adaptive Pyramid Feature Fusion aligns all three pyramid levels and learns both scale-wise and expert-wise fusion weights.
- **Lightweight design** — the main DCP-YOLO model contains 2.76M parameters while improving small-object accuracy over the YOLO11s baseline in the reported experiments.

## Results

### VisDrone2019 Validation

| Model | Parameters | FLOPs | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s | 9.42M | 21.3G | 39.9 | 23.7 |
| Modified YOLO11s (`P2–P4`) | 3.11M | 24.4G | 43.8 | 26.8 |
| **DCP-YOLO** | **2.76M** | **28.7G** | **47.0** | **29.1** |

### VisDrone2019-DET Test-Dev

| Model | Parameters | FLOPs | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 9.42M | 21.3G | 44.0 | 34.7 | 32.7 | 18.8 |
| DCP-YOLO-n | 0.87M | 10.4G | 43.1 | 34.2 | 32.4 | 19.0 |
| **DCP-YOLO** | **2.76M** | **28.7G** | **49.4** | **39.0** | **37.9** | **22.3** |

The manuscript also reports cross-dataset experiments on UAVDT and TinyPerson.

## Released Weights

The checkpoints used for the principal, comparative, and ablation experiments are available from Baidu Netdisk:

- **Download:** [DCP-YOLO release weights](https://pan.baidu.com/s/1GPSdsTQWPnGg91H7InVm6w?pwd=cq27)
- **Extraction code:** `cq27`

The archive is organized as follows:

```text
DCP-YOLO_release_weights_20260803/
├── 01_core/                  # DCP-YOLO checkpoints for the three datasets
├── 03_ablation/              # Topology, module, and AFRD-placement ablations
├── 04_comparative_models/    # Checkpoints for the compared detectors
├── README.md
└── SHA256SUMS.txt
```

Verify downloads with `SHA256SUMS.txt`. DCP-YOLO checkpoints contain custom components and must be loaded with this repository's source code. Checkpoints from comparison methods should be evaluated with their corresponding implementations; REMDET uses the MMDetection `.pth` format.

## Datasets

- [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)
- [UAVDT](https://sites.google.com/view/grli-uavdt)
- [TinyPerson / TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark)

Convert each dataset to YOLO detection format and set the local paths in your dataset YAML. Dataset files and generated `runs/` directories are ignored by Git.

## Repository Layout

```text
DCP-YOLO/
├── modules.py                      # Paper-aligned AFRD, LCC-C3k2, and APFF
├── ultralytics/
│   ├── nn/modules/
│   │   ├── conv.py                 # DCED downsampling implementation
│   │   ├── lcc_c3k2.py             # LCC-C3k2 implementation
│   │   └── moe.py                  # DMoE feature-fusion implementation
│   ├── nn/tasks.py                 # Model parser and custom-module registration
│   ├── train.py                    # Local experiment example
│   └── val.py                      # Validation/model-information example
├── examples/                       # Ultralytics usage examples
├── tests/                          # Framework tests
├── predict.py                      # Local inference example
├── pyproject.toml
└── LICENSE
```

The local example scripts retain experiment-specific paths and are intended as references. Replace these paths with values for your environment before use.

## License

This project is distributed under the [GNU Affero General Public License v3.0](LICENSE), consistent with the included Ultralytics codebase. Review the license requirements before redistribution or deployment.

## Acknowledgements

DCP-YOLO is developed on top of the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. We thank the authors and maintainers of Ultralytics, VisDrone, UAVDT, and TinyBenchmark for making their work publicly available.
