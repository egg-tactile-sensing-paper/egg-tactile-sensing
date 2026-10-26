"""Write the run matrix: one line per (model, protocol, initialisation) job.

Four arms, so the paper can separate three questions that the published table conflates:
the effect of the evaluation protocol, the effect of the initialisation, and the ranking
of the four detectors under each combination.
"""
from __future__ import annotations

import argparse
import json
import os

MODELS = ("yolov8m", "yolov9m", "yolov10m", "yolo11m")
ARMS = ("grouped-pretrained", "grouped-scratch", "published-pretrained", "published-scratch")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", default=os.path.join(here, "folds"))
    parser.add_argument("--out", default=os.path.join(here, "runs.tsv"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    args = parser.parse_args()

    with open(os.path.join(args.folds_dir, "folds.json")) as handle:
        manifest = json.load(handle)

    rows = []
    for arm in args.arms:
        protocol, init = arm.split("-")
        for model in args.models:
            if protocol == "grouped":
                for fold in manifest["folds"]:
                    rows.append((model, f"grouped-fold{fold['fold']}", fold["data_yaml"], init, arm))
            else:
                rows.append((model, "published", manifest["published_data_yaml"], init, arm))

    with open(args.out, "w") as handle:
        handle.write("# model\ttag\tdata\tinit\tarm\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    print(f"[runlist] {len(rows)} runs -> {args.out}")
    for arm in args.arms:
        print(f"  {arm}: {sum(1 for r in rows if r[4] == arm)}")


if __name__ == "__main__":
    main()
