"""Aggregate the finished runs into the tables the paper needs.

For the grouped protocol the headline numbers are the mean and standard deviation over the
five held-out egg sets.  Because each fold's test set is small, the detections are also
pooled across folds -- every one of the 19 eggs is held out exactly once, so the pooled
score is a single estimate computed on all 377 images, none of which was seen in training
by the model that scored it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics

METRICS = ("precision", "recall", "mAP50", "mAP50_95")
LABELS = {"precision": "Precision", "recall": "Recall", "mAP50": "mAP50", "mAP50_95": "mAP50:95"}
MODEL_LABELS = {"yolov8m": "YOLOv8m", "yolov9m": "YOLOv9m", "yolov10m": "YOLOv10m", "yolo11m": "YOLOv11m"}
ARM_LABELS = {
    "grouped-pretrained": "Grouped CV (pretrained)",
    "grouped-scratch": "Grouped CV (from scratch)",
    "published-pretrained": "Published split (pretrained)",
    "published-scratch": "Published split (from scratch)",
}
MAIN_ARMS = ("grouped-pretrained", "grouped-scratch", "published-pretrained", "published-scratch")
MODELS = ("yolov8m", "yolov9m", "yolov10m", "yolo11m")


def arm_of(record: dict) -> str:
    """The session-confound probe is its own arm; without this its runs would be counted as
    published-split runs, because their tag does not start with "grouped"."""
    if record["tag"].startswith("probe-"):
        return record["tag"]
    protocol = "grouped" if record["tag"].startswith("grouped") else "published"
    return f"{protocol}-{record['init']}"


def pooled_map(prediction_files: list[str]) -> dict | None:
    """COCO-style mAP over the union of the held-out images, if torchmetrics is present."""
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
    except Exception:
        return None
    metric = MeanAveragePrecision(box_format="xyxy")
    for path in prediction_files:
        with open(path) as handle:
            records = json.load(handle)
        for item in records:
            width, height = item["width"], item["height"]
            truth = []
            for cx, cy, w, h in item["true_xywhn"]:
                truth.append([(cx - w / 2) * width, (cy - h / 2) * height,
                              (cx + w / 2) * width, (cy + h / 2) * height])
            metric.update(
                [{"boxes": torch.tensor(item["pred_xyxy"]).reshape(-1, 4),
                  "scores": torch.tensor(item["pred_conf"]).reshape(-1),
                  "labels": torch.tensor(item["pred_cls"], dtype=torch.long).reshape(-1)}],
                [{"boxes": torch.tensor(truth).reshape(-1, 4),
                  "labels": torch.tensor(item["true_cls"], dtype=torch.long).reshape(-1)}],
            )
    computed = metric.compute()
    return {"mAP50": float(computed["map_50"]), "mAP50_95": float(computed["map"])}


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(here, "results"))
    parser.add_argument("--out", default=os.path.join(here, "analysis"))
    parser.add_argument("--no-pooled", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        if path.endswith(".predictions.json"):
            continue
        with open(path) as handle:
            records.append(json.load(handle))
    if not records:
        raise SystemExit(f"no results in {args.results_dir}")

    by_arm: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        by_arm[arm_of(record)][record["model"]].append(record)

    # The pooled scores are expensive enough to be worth computing once; make_paper_tex.py
    # reads them back out of pooled.json rather than recomputing or having them typed in.
    pooled_out: dict[str, dict[str, dict]] = {}

    lines = ["# Cross-validated detection results", "",
             f"{len(records)} finished runs in"
             f" `{os.path.relpath(args.results_dir, os.path.dirname(here))}`.", ""]
    latex: list[str] = []

    for arm in MAIN_ARMS:
        models = by_arm.get(arm)
        if not models:
            continue
        grouped = arm.startswith("grouped")
        lines += [f"## {ARM_LABELS[arm]}", ""]
        lines.append("| Model | " + " | ".join(LABELS[m] for m in METRICS) + " | folds |")
        lines.append("| --- " * (len(METRICS) + 2) + "|")

        latex += [f"% {ARM_LABELS[arm]}", r"\begin{tabular}{lcccc}", r"\toprule",
                  "Model & " + " & ".join(LABELS[m] for m in METRICS) + r" \\", r"\midrule"]

        for model in MODELS:
            runs = sorted(models.get(model, []), key=lambda r: r["tag"])
            if not runs:
                continue
            cells, latex_cells = [], []
            for metric in METRICS:
                values = [r["test"][metric] for r in runs]
                if grouped and len(values) > 1:
                    mean = statistics.mean(values)
                    sd = statistics.stdev(values)
                    cells.append(f"{mean:.3f} ± {sd:.3f}")
                    latex_cells.append(f"${mean:.3f} \\pm {sd:.3f}$")
                else:
                    cells.append(f"{values[0]:.3f}")
                    latex_cells.append(f"${values[0]:.3f}$")
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + f" | {len(runs)} |")
            latex.append(f"{MODEL_LABELS[model]} & " + " & ".join(latex_cells) + r" \\")

        latex += [r"\bottomrule", r"\end{tabular}", ""]
        lines.append("")

        if grouped:
            lines += ["Per fold (held-out eggs only):", "",
                      "| Model | fold | test images | " + " | ".join(LABELS[m] for m in METRICS) + " |",
                      "| --- " * (len(METRICS) + 3) + "|"]
            for model in MODELS:
                for run in sorted(models.get(model, []), key=lambda r: r["tag"]):
                    fold = run["tag"].replace("grouped-fold", "")
                    images = run.get("test_images", "")
                    scores = " | ".join(f"{run['test'][m]:.3f}" for m in METRICS)
                    lines.append(f"| {MODEL_LABELS[model]} | {fold} | {images} | {scores} |")
            lines.append("")

            if not args.no_pooled:
                lines += ["Pooled over the folds (all 19 eggs, each scored by a model that never saw it):",
                          "", "| Model | pooled mAP50 | pooled mAP50:95 |", "| --- | --- | --- |"]
                for model in MODELS:
                    runs = models.get(model, [])
                    files = [os.path.join(args.results_dir,
                                          f"{r['model']}-{r['tag']}-{r['init']}.predictions.json")
                             for r in runs]
                    files = [f for f in files if os.path.exists(f)]
                    if len(files) != len(runs) or not files:
                        continue
                    pooled = pooled_map(files)
                    if pooled is None:
                        lines.append("| _torchmetrics not installed; pooled scores skipped_ | | |")
                        break
                    pooled_out.setdefault(arm, {})[model] = pooled
                    lines.append(f"| {MODEL_LABELS[model]} | {pooled['mAP50']:.3f} | {pooled['mAP50_95']:.3f} |")
                lines.append("")

    # The headline comparison: what the protocol alone changes.
    if "grouped-pretrained" in by_arm and "published-pretrained" in by_arm:
        lines += ["## Effect of the evaluation protocol", "",
                  "| Model | published split mAP50 | grouped CV mAP50 | difference |",
                  "| --- | --- | --- | --- |"]
        for model in MODELS:
            published = by_arm["published-pretrained"].get(model)
            grouped = by_arm["grouped-pretrained"].get(model)
            if not published or not grouped:
                continue
            a = published[0]["test"]["mAP50"]
            b = statistics.mean(r["test"]["mAP50"] for r in grouped)
            lines.append(f"| {MODEL_LABELS[model]} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
        lines.append("")

    # The session-confound probe.  Both arms are cracked-only and share their training data,
    # so the broken-egg row is the whole measurement: "cross" is the one cracked egg recorded
    # in the intact session, "same" the cracked eggs recorded in the cracked session.
    if "probe-cross" in by_arm and "probe-same" in by_arm:
        lines += ["## Session-confound probe", "",
                  "Identical training data in both arms (intact session + cracked session eggs).",
                  "The test sets are cracked-only, so the broken-egg metrics are the measurement.",
                  "`cross` is episode 11, the only cracked egg recorded in the intact session;",
                  "`same` is episodes 18-19, cracked eggs from the cracked session.", "",
                  "| Model | cross recall | same recall | cross AP50 | same AP50 | AP50 difference |",
                  "| --- | --- | --- | --- | --- | --- |"]
        probe_latex = [r"% session-confound probe", r"\begin{tabular}{lcccc}", r"\toprule",
                       r"& \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{AP50} \\",
                       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                       r"Model & cross & same & cross & same \\", r"\midrule"]
        deltas = []
        for model in MODELS:
            cross = by_arm["probe-cross"].get(model)
            same = by_arm["probe-same"].get(model)
            if not cross or not same:
                continue
            c = cross[0]["test"]["per_class"]["broken_egg"]
            m = same[0]["test"]["per_class"]["broken_egg"]
            deltas.append(c["mAP50"] - m["mAP50"])
            lines.append(f"| {MODEL_LABELS[model]} | {c['recall']:.3f} | {m['recall']:.3f} | "
                         f"{c['mAP50']:.3f} | {m['mAP50']:.3f} | {c['mAP50'] - m['mAP50']:+.3f} |")
            probe_latex.append(f"{MODEL_LABELS[model]} & ${c['recall']:.3f}$ & ${m['recall']:.3f}$ & "
                               f"${c['mAP50']:.3f}$ & ${m['mAP50']:.3f}$ \\\\")
        probe_latex += [r"\bottomrule", r"\end{tabular}", ""]
        latex += probe_latex
        if deltas:
            lines += ["", f"Mean AP50 difference (cross - same): **{statistics.mean(deltas):+.3f}**"
                          f" over {len(deltas)} models. A session shortcut would make this strongly"
                          " negative, because the cross arm's egg comes from a session in which"
                          " training contained intact eggs only.", ""]

    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "results.md"), "w") as handle:
        handle.write(report)
    with open(os.path.join(args.out, "results_tables.tex"), "w") as handle:
        handle.write("\n".join(latex) + "\n")
    if pooled_out:
        with open(os.path.join(args.out, "pooled.json"), "w") as handle:
            json.dump(pooled_out, handle, indent=2, sort_keys=True)
    print(report)
    print(f"[collect] wrote {args.out}/results.md and results_tables.tex")


if __name__ == "__main__":
    main()
