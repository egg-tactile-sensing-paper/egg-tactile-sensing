# Licensing and third-party software

The root `LICENSE` file is **CC0 1.0 Universal**. It covers the material created for this
project: the tactile image dataset under `data/`, `config.yaml`, the training and evaluation
code under `cross_validation/`, and the `main_code.py` scripts.

It does **not** relicense anything owned by others, and it does not travel with the
third-party software this project depends on.

## Third-party software

The four baselines are trained with software released by others, which this repository
depends on but does not redistribute:

| component | upstream licence |
| --- | --- |
| Ultralytics (YOLOv8, YOLO11, and the training/evaluation framework) | AGPL-3.0 |
| YOLOv10 reference implementation | AGPL-3.0 |
| Original YOLOv9 implementation | GPL-3.0 |
| PyTorch, torchvision | BSD-3-Clause |

No source file from any of these projects is copied into this repository. `main_code.py` and
everything under `cross_validation/` import Ultralytics as an installed dependency, pinned in
`cross_validation/requirements.txt`.

COCO-pretrained starting checkpoints are **not** included; they are downloaded from
[`ultralytics/assets`](https://github.com/ultralytics/assets/releases/download/v8.3.0) under
their own terms. See `cross_validation/README.md`.

## Trained weights

`yolo*_code_and_weights/weights/best.pt` and `last.pt` are checkpoints we trained on this
dataset using Ultralytics software. Whether weights produced by AGPL-3.0-licensed training
code are themselves subject to that licence is not settled, and we make no claim either way:
the CC0 dedication above is offered for our own contributions, and anyone redistributing these
checkpoints should satisfy themselves about the upstream terms. This is the open item behind
the "No" answer to checklist question 12 in the paper.

## Dataset

The tactile images under `data/` were collected by the authors with GelSight Mini sensors and
annotated in Roboflow. They are dedicated to the public domain under CC0 1.0. Roboflow was
used as an annotation tool only; no third-party imagery is included.
