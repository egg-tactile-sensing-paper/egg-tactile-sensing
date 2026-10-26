"""Train and evaluate one (model, fold) pair and record the metrics as JSON.

Both the validation and the held-out test numbers are written, so there is never any doubt
about which partition a reported score came from.  Checkpoint selection uses the inner
validation episodes only; the test episodes are touched once, after training.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time

MODELS = ("yolov8m", "yolov9m", "yolov10m", "yolo11m")
CLASS_NAMES = {0: "normal_egg", 1: "broken_egg"}

HERE = os.path.dirname(os.path.abspath(__file__))
# Checkpoints and Ultralytics run directories are large and are not part of the release.
# They go wherever TACTILE_WORK points; the original runs used a scratch filesystem.
WORK = os.environ.get("TACTILE_WORK", os.path.join(HERE, "work"))


def metrics_of(result) -> dict:
    box = result.box
    per_class = {}
    for position, class_index in enumerate(box.ap_class_index):
        per_class[CLASS_NAMES[int(class_index)]] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "mAP50": float(box.ap50[position]),
            "mAP50_95": float(box.ap[position]),
        }
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": per_class,
    }


def dump_predictions(model, data_yaml: str, out_path: str, imgsz: int, conf: float = 0.001) -> None:
    """Store raw detections on the held-out images so folds can be pooled later."""
    import yaml

    with open(data_yaml) as handle:
        spec = yaml.safe_load(handle)
    image_dir = os.path.join(spec["path"], spec["test"])
    label_dir = image_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    records = []
    for name in sorted(os.listdir(image_dir)):
        prediction = model.predict(os.path.join(image_dir, name), conf=conf, imgsz=imgsz,
                                   verbose=False)[0]
        boxes = prediction.boxes
        label_file = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
        with open(label_file) as handle:
            truth = [[float(v) for v in line.split()] for line in handle if line.split()]
        records.append({
            "image": name,
            "width": int(prediction.orig_shape[1]),
            "height": int(prediction.orig_shape[0]),
            "pred_xyxy": boxes.xyxy.tolist(),
            "pred_conf": boxes.conf.tolist(),
            "pred_cls": [int(c) for c in boxes.cls.tolist()],
            "true_xywhn": [t[1:] for t in truth],
            "true_cls": [int(t[0]) for t in truth],
        })
    with open(out_path, "w") as handle:
        json.dump(records, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--data", required=True, help="path to the fold's data.yaml")
    parser.add_argument("--tag", required=True, help="name for this run, e.g. grouped-fold2")
    parser.add_argument("--init", default="pretrained", choices=("pretrained", "scratch"))
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights-dir", default=os.path.join(WORK, "pretrained"))
    parser.add_argument("--runs-dir", default=os.path.join(WORK, "runs"))
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--allow-download", action="store_true",
                        help="let ultralytics fetch checkpoints instead of requiring them"
                             " pre-staged in --weights-dir (the original runs were offline)")
    args = parser.parse_args()

    # Never let ultralytics reach the network from a compute node: every checkpoint it may
    # want, including the one its AMP check loads, is staged in --weights-dir beforehand.
    # That is how every run reported in the paper was produced; --allow-download exists only
    # so the script also works on a machine where the checkpoints have not been staged.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    if not args.allow_download:
        os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("WANDB_DISABLED", "true")

    run_name = f"{args.model}-{args.tag}-{args.init}"
    work_dir = os.path.join(args.runs_dir, run_name, "cwd")
    os.makedirs(work_dir, exist_ok=True)
    amp_probe = os.path.join(args.weights_dir, "yolo11n.pt")
    if os.path.exists(amp_probe) and not os.path.exists(os.path.join(work_dir, "yolo11n.pt")):
        os.symlink(amp_probe, os.path.join(work_dir, "yolo11n.pt"))
    os.chdir(work_dir)

    from ultralytics import YOLO

    if args.init == "pretrained":
        start_from = os.path.join(args.weights_dir, f"{args.model}.pt")
        if not os.path.exists(start_from):
            raise SystemExit(f"missing checkpoint {start_from}; run scripts/setup_env.sh first")
    else:
        start_from = f"{args.model}.yaml"

    started = time.time()
    model = YOLO(start_from)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        seed=args.seed, deterministic=True, device=args.device, workers=args.workers,
        project=args.runs_dir, name=run_name, exist_ok=True, plots=False, verbose=False,
    )
    best = str(model.trainer.best)

    evaluator = YOLO(best)
    scores = {}
    for split in ("val", "test"):
        result = evaluator.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                               device=args.device, project=args.runs_dir,
                               name=f"{run_name}-{split}", exist_ok=True, plots=False, verbose=False)
        scores[split] = metrics_of(result)

    os.makedirs(args.results_dir, exist_ok=True)
    if not args.no_predictions:
        dump_predictions(evaluator, args.data,
                         os.path.join(args.results_dir, f"{run_name}.predictions.json"), args.imgsz)

    with open(args.data) as handle:
        data_spec = handle.read()
    import yaml

    spec = yaml.safe_load(data_spec)
    counts = {
        split: len(os.listdir(os.path.join(spec["path"], spec[split])))
        for split in ("train", "val", "test")
    }
    record = {
        "model": args.model,
        "tag": args.tag,
        "init": args.init,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "data": os.path.abspath(args.data),
        "data_yaml": data_spec,
        "best_checkpoint": best,
        "minutes": round((time.time() - started) / 60, 1),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "test_images": counts["test"],
        "val": scores["val"],
        "test": scores["test"],
    }
    out_path = os.path.join(args.results_dir, f"{run_name}.json")
    with open(out_path, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"[train] {run_name}: test mAP50={scores['test']['mAP50']:.3f} "
          f"mAP50:95={scores['test']['mAP50_95']:.3f} ({record['minutes']} min) -> {out_path}")


if __name__ == "__main__":
    main()
