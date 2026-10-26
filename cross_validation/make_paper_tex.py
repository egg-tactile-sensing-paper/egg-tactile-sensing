"""Emit the LaTeX fragments whose numbers come from the data alone.

Nothing here depends on a training run: the episode structure, the leakage audit and the
fold composition are fixed once the dataset and the fold assignment exist.  Running this
after make_folds.py keeps the paper's numbers sourced from the files rather than typed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import glob
import statistics
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, EPISODE_GAP_S, build_episodes, load_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper")

SHORT = {"normal_egg": "intact", "broken_egg": "cracked"}

# mAP50 as printed in the manuscript's Table 1, used only to identify which split those
# numbers came from.  Update here if the table changes.
COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
TABLE1_MAP50 = {"yolov8m": 0.850, "yolov9m": 0.856, "yolov10m": 0.816, "yolo11m": 0.829}


def notation(text: str) -> str:
    """Match the manuscript: subscripted metric names and the "whole egg" class vocabulary.

    Applied to the finished fragment rather than to each literal, so that the result-JSON
    keys ("mAP50", "normal_egg") keep their own spelling everywhere in the code.
    """
    for old, new in (("mAP50:95", r"mAP$_{50:95}$"), ("mAP50_95", r"mAP$_{50:95}$"),
                     ("mAP50", r"mAP$_{50}$"), ("AP50", r"AP$_{50}$"),
                     ("intact session", "whole-egg session"),
                     ("intact episode", "whole-egg episode"),
                     ("intact-egg", "whole-egg"), ("intact", "whole"),
                     ("in./cr.", "wh./cr.")):
        text = text.replace(old, new)
    return text


def utc(ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000.0, dt.timezone.utc)


def load_results(results_dir: str) -> dict:
    """Group the finished runs by arm.  Missing arms simply yield no paragraph."""
    out: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if path.endswith(".predictions.json"):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        tag = rec["tag"]
        if tag.startswith("probe-"):
            arm = tag
        else:
            arm = ("grouped" if tag.startswith("grouped") else "published") + "-" + rec["init"]
        out.setdefault(arm, {}).setdefault(rec["model"], []).append(rec)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()

    frames = load_frames(ROOT)
    episodes = build_episodes(frames)
    folds = json.load(open(os.path.join(HERE, "folds", "folds.json")))

    published = {}
    for f in frames:
        published[f.split] = published.get(f.split, 0) + 1
    instances = sum(len(f.classes) for f in frames)
    per_class = {}
    for f in frames:
        for c in f.classes:
            per_class[CLASS_NAMES[c]] = per_class.get(CLASS_NAMES[c], 0) + 1

    multi = sum(1 for e in episodes if len(e.splits) > 1)
    leaked = sum(
        1
        for e in episodes
        for f in e.frames
        if f.split == "test" and (e.splits - {"test"})
    )
    n_test = published.get("test", 0)

    # ---- Table 1: the 19 contact episodes -------------------------------------------
    rows = []
    for e in episodes:
        sp = {}
        for f in e.frames:
            sp[f.split] = sp.get(f.split, 0) + 1
        cell = ", ".join(f"{k[:2]}~{sp[k]}" for k in ("train", "val", "test") if k in sp)
        rows.append(
            rf"{e.index + 1} & {utc(e.start_ms):%b~%d\ %H:%M} & {SHORT[CLASS_NAMES[e.cls]]} & "
            rf"{e.size} & {e.duration_s:.0f} & {cell} \\"
        )
    tab1 = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{The %d contact episodes recovered from frame timestamps. Frames within one"
            % len(episodes),
            r"episode are a median %.2f\,s apart; a gap above %.0f\,s starts a new episode. Every"
            % (
                statistics.median(
                    (b.ts_ms - a.ts_ms) / 1000.0
                    for e in episodes
                    for a, b in zip(e.frames, e.frames[1:])
                ),
                EPISODE_GAP_S,
            ),
            r"episode is of a single egg and a single class. The last column shows how the released",
            r"split distributes each episode (tr/va/te).}",
            r"\label{tab:episodes}",
            r"\begin{tabular}{rllrrl}",
            r"\toprule",
            r"ep. & start (UTC) & class & frames & dur.\,(s) & released split \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    # ---- Table 2: fold composition --------------------------------------------------
    frows = []
    for fd in folds["folds"]:
        te = fd["test"]["episodes_per_class"]
        frows.append(
            rf"{fd['fold']} & {fd['train']['n_episodes']} & {fd['train']['n_images']} & "
            rf"{fd['val']['n_episodes']} & {fd['val']['n_images']} & "
            rf"{fd['test']['n_episodes']} & {fd['test']['n_images']} & "
            rf"{te.get('normal_egg', 0)}/{te.get('broken_egg', 0)} & "
            rf"{', '.join(str(x) for x in fd['test']['episodes'])} \\"
        )
    tr = [fd["train"]["n_images"] for fd in folds["folds"]]
    tab2 = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{Composition of the %d grouped folds. Episodes, not images, are the unit of"
            % folds["outer_folds"],
            r"partition, so no frame of a held-out egg is seen in training or model selection. Each",
            r"episode is held out exactly once, so the %d test sets partition all %d images."
            % (folds["outer_folds"], len(frames)),
            r"Validation episodes are drawn from the remaining pool and used only for checkpoint",
            r"selection.}",
            r"\label{tab:folds}",
            r"\begin{tabular}{rrrrrrrcl}",
            r"\toprule",
            r"& \multicolumn{2}{c}{train} & \multicolumn{2}{c}{val} & \multicolumn{2}{c}{test} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"fold & ep. & img & ep. & img & ep. & img & in./cr. & held-out episodes \\",
            r"\midrule",
            *frows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    # ---- prose ----------------------------------------------------------------------
    whole = [e for e in episodes if CLASS_NAMES[e.cls] == "normal_egg"]
    crack = [e for e in episodes if CLASS_NAMES[e.cls] == "broken_egg"]
    w_end, c_start = utc(max(e.frames[-1].ts_ms for e in whole)), utc(min(e.start_ms for e in crack))

    med_gap = statistics.median(
        (b.ts_ms - a.ts_ms) / 1000.0
        for e in episodes
        for a, b in zip(e.frames, e.frames[1:])
    )
    test_eps = {e.index for e in episodes for f in e.frames if f.split == "test"}
    te_n = [fd["test"]["n_episodes"] for fd in folds["folds"]]
    te_i = [fd["test"]["n_images"] for fd in folds["folds"]]
    va_n = [fd["val"]["n_episodes"] for fd in folds["folds"]]
    k = folds["outer_folds"]

    def rng(xs):
        return f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}--{max(xs)}"

    res = load_results(args.results_dir)
    MODELS = ("yolov8m", "yolov9m", "yolov10m", "yolo11m")

    def arm_complete(arm) -> bool:
        """Every model present, with every run it is supposed to have.

        A partially finished arm must not be averaged: the mean over the folds that happen
        to have landed first is not an estimate of anything, and silently quoting it in the
        paper would be worse than reporting nothing.
        """
        got = res.get(arm, {})
        expected = k if arm.startswith("grouped") else 1
        return all(len(got.get(m, [])) == expected for m in MODELS)

    def arm_mean(arm, metric="mAP50", split="test"):
        """Mean over models of each model's own mean over its runs, or None if incomplete."""
        if not arm_complete(arm):
            return None
        got = res[arm]
        return statistics.mean(
            statistics.mean(r[split][metric] for r in got[m]) for m in MODELS
        )

    def fmt(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    protocol_paras = [
        r"\paragraph{Unit of evaluation.}"
        f" The {len(frames)} tactile images are not {len(frames)} independent observations."
        " The GelSight frames are captured as continuous bursts while the sensor is held"
        " against one egg, and the acquisition time of every frame survives in its filename."
        f" Clustering frames by timestamp (a gap above {EPISODE_GAP_S:.0f}\\,s starts a new"
        f" burst) recovers {len(episodes)} contact episodes, {len(whole)} of intact eggs and"
        f" {len(crack)} of cracked eggs (Table~\\ref{{tab:episodes}}); within an episode"
        f" successive frames are a median {med_gap:.2f}\\,s apart and differ only slightly."
        f" The effective sample size of the dataset is therefore {len(episodes)} eggs, not"
        f" {len(frames)} images.",

        "The released split assigns images at random, which places near-duplicate frames of"
        f" the same physical egg on both sides of the partition: {multi} of the {len(episodes)}"
        f" episodes appear in more than one split, and every one of the {n_test} test images"
        " comes from an episode that also contributes a training or validation frame. (The"
        " export contains no byte-identical or augmented copies; the overlap is between"
        " distinct but adjacent frames of one continuous contact.) A score computed on that"
        " split measures recall of eggs the detector has already seen, and because the"
        f" {n_test} test images are spread over {len(test_eps)} already-seen eggs it also"
        " carries no usable estimate of variance.",

        r"\paragraph{Grouped cross-validation.}"
        f" We therefore re-evaluate every detector with {k}-fold cross-validation grouped by"
        f" contact episode and stratified by class. Each fold holds out {rng(te_n)} complete"
        f" episodes ({rng(te_i)} images, both classes always present) for testing and draws a"
        f" disjoint set of {rng(va_n)} episodes from the remaining pool for checkpoint"
        f" selection, leaving {rng(tr)} images for training (Table~\\ref{{tab:folds}}). Every"
        f" episode is held out exactly once, so the {k} test sets partition all {len(frames)}"
        " images and each image is scored by exactly one model that never saw its egg.",

        "We report two aggregates, because they answer different questions. The mean and"
        " standard deviation over folds describe how much the score moves when the held-out"
        " eggs change, which is the quantity the released single split cannot provide. The"
        " pooled score instead concatenates the per-fold detections and computes one average"
        f" precision over all {len(frames)} images under a single confidence ranking; it is"
        " the stricter and more directly comparable number, since averaging five small"
        " per-fold APs rewards folds that happen to contain few, easy instances. The gap"
        " between the two aggregates is itself a reading of how little data each fold holds.",

        r"\paragraph{Arms.}"
        " So that the protocol change is not confounded with a change of training recipe, we"
        " run a crossed design: both the released image-level split and the grouped folds,"
        " each with both the released initialisation (random weights, 350 epochs) and"
        " COCO-pretrained weights fine-tuned for 100 epochs. Everything else -- architecture"
        " family, image size, batch size, augmentation, seed -- is held fixed, and all four"
        " arms are scored by the same evaluation code on the test split. The effect of the"
        " protocol is then read within a single initialisation, and vice versa.",
    ]

    TABLE1_MEAN = statistics.mean(TABLE1_MAP50.values())
    _ps_runs = res.get("published-scratch", {})
    T1_VAL_GAP = [abs(TABLE1_MAP50[m] - _ps_runs[m][0]["val"]["mAP50"]) for m in MODELS if _ps_runs.get(m)]
    T1_TEST_GAP = [_ps_runs[m][0]["test"]["mAP50"] - TABLE1_MAP50[m] for m in MODELS if _ps_runs.get(m)]

    gp, pp = arm_mean("grouped-pretrained"), arm_mean("published-pretrained")
    gs, ps = arm_mean("grouped-scratch"), arm_mean("published-scratch")
    ps_val = arm_mean("published-scratch", split="val")
    # Arm means are *displayed* to 3 dp but *differenced* at full precision. Rounding the
    # operands first made every quoted difference disagree with the data by up to 0.001:
    # 0.919 - 0.880 prints as 0.039 while the real drop is 0.0399, which rounds to 0.040.
    # So differences and ratios of printed quantities are quoted to 2 dp -- the precision a
    # 3 dp operand actually supports -- which is correct whether the reader recomputes from
    # the raw runs or subtracts the two numbers printed beside it.

    provenance_paras = [
        r"\paragraph{What the released numbers measure.}"
        " The released training script calls Ultralytics' ``train`` entry point and nothing"
        " else, so the metrics it prints at the end of a run are those of the validation pass"
        " used for checkpoint selection, not of the held-out test split. Re-running that exact"
        " configuration and scoring both splits with our own evaluation code reproduces the"
        f" originally reported table on the validation split (mAP50 {fmt(ps_val)} against"
        f" {fmt(TABLE1_MEAN)} as printed, averaged over the four detectors) and not on the"
        f" test split, where the same models reach {fmt(ps)}. Per detector the printed values"
        f" lie within {max(T1_VAL_GAP):.3f} of our validation scores and"
        f" {min(T1_TEST_GAP):.3f}--{max(T1_TEST_GAP):.3f} below our test scores. We therefore"
        " read the previously reported figures as validation metrics and quote test metrics"
        " throughout, which raises rather than lowers the released configuration's apparent"
        " accuracy.",
    ] if ps is not None and ps_val is not None else []

    # The pooled scores come from collect.py, which already computes them over the union of
    # the held-out images; recomputing them here would only risk the two disagreeing.
    pooled_scores: dict[str, dict[str, dict]] = {}
    pooled_path = os.path.join(HERE, "analysis", "pooled.json")
    if os.path.exists(pooled_path):
        with open(pooled_path) as fh:
            pooled_scores = json.load(fh)

    def pooled_vals(arm):
        """Pooled mAP50 for every model in an arm, or [] unless all of them are there."""
        got = pooled_scores.get(arm, {})
        vals = [got[m]["mAP50"] for m in MODELS if m in got]
        return vals if len(vals) == len(MODELS) else []

    effect_paras = []
    if gp is not None and pp is not None:
        pairs = [("COCO-pretrained weights", pp, gp)]
        if gs is not None and ps is not None:
            pairs.append(("the released from-scratch recipe", ps, gs))
        bits = "; with ".join(
            f"{name}, mAP50 falls from {fmt(a)} on the released split to {fmt(b)} under"
            f" grouped cross-validation, a drop of {abs(b - a):.2f}" for name, a, b in pairs
        )
        text = (r"\paragraph{What the leakage was worth.}"
                " Holding the training recipe fixed and changing only the partition:"
                f" with {bits}.")
        if len(pairs) == 2:
            ratio = abs(pairs[1][2] - pairs[1][1]) / abs(pairs[0][2] - pairs[0][1])
            text += (f" The premium is {ratio:.1f} times as large for the recipe the released"
                     " code actually uses, which is the ordering one would expect: a detector"
                     " initialised at random has no external prior to fall back on, so it"
                     " leans harder on the training frames themselves and gains more when"
                     " near-duplicates of its test images sit in the training set.")
        pooled_bits = ", ".join(
            f"{min(vals):.3f}--{max(vals):.3f} {label}" for label, vals in
            (("from scratch", pooled_vals("grouped-scratch")),
             ("pretrained", pooled_vals("grouped-pretrained"))) if vals
        )
        if pooled_bits:
            text += (f" Pooling the per-fold detections lowers the estimate further, to"
                     f" {pooled_bits}, since one confidence ranking over all {len(frames)}"
                     " images is not flattered by whichever fold happened to hold few and"
                     " easy instances. We take the per-fold mean as the headline and the"
                     " pooled value as the conservative bound.")
        else:
            text += (" The pooled cross-validated estimate is the more conservative reading of"
                     " the same runs.")
        text += (" On either reading the released protocol is optimistic by a visible margin,"
                 " without the leak manufacturing the result: the detectors still separate"
                 " cracked from intact eggs whose episodes they have never seen.")
        effect_paras = [text]

    probe_manifest = os.path.join(HERE, "probe", "probe.json")
    probe_ep = 11
    if os.path.exists(probe_manifest):
        with open(probe_manifest) as fh:
            probe_ep = json.load(fh)["arms"]["cross"]["episodes"][0]

    cross = res.get("probe-cross", {})
    same = res.get("probe-same", {})
    deltas = [
        cross[m][0]["test"]["per_class"]["broken_egg"]["mAP50"]
        - same[m][0]["test"]["per_class"]["broken_egg"]["mAP50"]
        for m in MODELS if cross.get(m) and same.get(m)
    ]
    cross_r = [cross[m][0]["test"]["per_class"]["broken_egg"]["recall"] for m in MODELS if cross.get(m)]

    limitation_paras = [
        r"\paragraph{Class is confounded with acquisition session, and we tested whether it matters.}"
        " The dataset was collected in class order: every intact-egg episode was recorded"
        f" before every cracked-egg episode (intact {utc(min(e.start_ms for e in whole)):%b~%d %H:%M}--"
        f"{w_end:%b~%d %H:%M}, cracked {c_start:%b~%d %H:%M}--"
        f"{utc(max(e.frames[-1].ts_ms for e in crack)):%b~%d %H:%M} UTC), with the class"
        " boundary falling at a session break. Any systematic change across that boundary --"
        " gel wear, membrane replacement, illumination from the sensor's internal LEDs,"
        " mounting pressure, ambient temperature -- is correlated with the label and is in"
        " principle available as a shortcut, and no partition of these episodes can separate"
        " the two. Rather than leave that as a caveat we measured it.",
    ]

    if deltas:
        limitation_paras.append(
            f"Episode {probe_ep} is the one cracked egg recorded inside the intact session,"
            " nineteen minutes after the last intact episode ends; every other cracked egg"
            " belongs to the following session. Training on the intact session plus the"
            " cracked session's other eggs and then testing on that episode asks the"
            " question directly: a detector keyed to session appearance should fail there,"
            " having seen only intact eggs from that session, while a detector keyed to crack"
            " morphology should not. Against a control test set of cracked eggs drawn from"
            " the cracked session with identical training data, the cross-session egg is"
            f" detected equally well -- mean AP50 difference {statistics.mean(deltas):+.3f}"
            f" over the {COUNT_WORDS.get(len(deltas), len(deltas))} detectors, with recall"
            f" between {min(cross_r):.3f} and"
            f" {max(cross_r):.3f}. The shortcut is available but the models are not using it,"
            " so the cross-validated scores are not explained by the confound."
        )
        limitation_paras.append(
            "That test rests on a single egg, nine frames, and that is the most the dataset"
            " permits: it contains exactly one cracked episode outside the cracked session and"
            " no intact episode inside it, so no larger version of the probe and no reverse"
            " direction exist. A weaker session effect would not be visible at this sample"
            " size. Re-acquiring a modest set of episodes with the two classes interleaved"
            " within each session remains the way to settle it -- now to confirm what one egg"
            " indicates, rather than to rule out a likely artefact."
        )

    # The quantitative version of this point lives in the results section; repeating the
    # standard deviations here would only be redundancy for a reviewer to notice.
    limitation_paras.append(
        r"\paragraph{Scale.}"
        f" With {len(episodes)} eggs the sample is small, and the grouped protocol makes that"
        " explicit rather than hiding it: the spread across folds is wider than the differences"
        " between architectures, which is why we quote a baseline range instead of a ranking."
        " A single split cannot expose that spread, and so invites exactly the ranking the data"
        " cannot support. The absolute numbers here should be read as evidence about the"
        " sensing modality, not as a benchmark result."
    )

    # ---------------- results section ----------------
    MODEL_LABEL = {"yolov8m": "YOLOv8m", "yolov9m": "YOLOv9m",
                   "yolov10m": "YOLOv10m", "yolo11m": "YOLOv11m"}
    METRICS = ("precision", "recall", "mAP50", "mAP50_95")
    ARM_ORDER = (
        ("published-scratch", "Released split --- random init, 350 epochs (the released recipe)"),
        ("grouped-scratch", "Grouped CV --- random init, 350 epochs"),
        ("published-pretrained", "Released split --- COCO-pretrained, 100 epochs"),
        ("grouped-pretrained", "Grouped CV --- COCO-pretrained, 100 epochs"),
    )

    def model_mean(arm, metric="mAP50"):
        return {m: round(statistics.mean(r["test"][metric] for r in res[arm][m]), 3)
                for m in MODELS}

    def model_sd(arm, metric="mAP50"):
        return {m: round(statistics.stdev([r["test"][metric] for r in res[arm][m]]), 3)
                for m in MODELS}

    def class_stat(arm, cls, field):
        return [r["test"]["per_class"][cls][field] for m in MODELS for r in res[arm][m]
                if cls in r["test"]["per_class"]]

    results_table = pooled_table = None
    results_paras = []

    if all(arm_complete(a) for a, _ in ARM_ORDER):
        rows = [r"\begin{table}[t]", r"\centering\small",
                r"\caption{Detection accuracy on the held-out test data under each protocol and"
                f" initialisation. Grouped rows are the mean and standard deviation over the {k}"
                r" folds, each holding out a different set of whole eggs; released-split rows are"
                r" a single run, which is all that partition can provide. Changing only the"
                r" partition moves the score further than changing the architecture or the"
                r" initialisation does.}",
                r"\label{tab:main}",
                r"\begin{tabular}{lcccc}", r"\toprule",
                r"Model & Precision & Recall & mAP50 & mAP50:95 \\"]
        for arm, label in ARM_ORDER:
            rows += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\"]
            is_grouped = arm.startswith("grouped")
            for m in MODELS:
                cells = []
                for metric in METRICS:
                    vals = [r["test"][metric] for r in res[arm][m]]
                    cells.append(f"${statistics.mean(vals):.3f} \\pm {statistics.stdev(vals):.3f}$"
                                 if is_grouped else f"${vals[0]:.3f}$")
                rows.append(f"{MODEL_LABEL[m]} & " + " & ".join(cells) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        results_table = "\n".join(rows)

    if pooled_vals("grouped-scratch") and pooled_vals("grouped-pretrained"):
        rows = [r"\begin{table}[t]", r"\centering\small",
                r"\caption{Pooled grouped cross-validation: the per-fold detections are"
                f" concatenated and scored as one set, so all {len(frames)} images contribute"
                r" under a single confidence ranking rather than being averaged as"
                r" {k} small per-fold scores. This is the conservative estimate of the two"
                r" aggregates.}".replace("{k}", str(k)),
                r"\label{tab:pooled}",
                r"\begin{tabular}{lcccc}", r"\toprule",
                r"& \multicolumn{2}{c}{Random init, 350 ep}"
                r" & \multicolumn{2}{c}{COCO-pretrained, 100 ep} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                r"Model & mAP50 & mAP50:95 & mAP50 & mAP50:95 \\", r"\midrule"]
        for m in MODELS:
            a = pooled_scores["grouped-scratch"][m]
            b = pooled_scores["grouped-pretrained"][m]
            rows.append(f"{MODEL_LABEL[m]} & ${a['mAP50']:.3f}$ & ${a['mAP50_95']:.3f}$ &"
                        f" ${b['mAP50']:.3f}$ & ${b['mAP50_95']:.3f}$ \\\\")
        rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        pooled_table = "\n".join(rows)

    if results_table is not None:
        mu_p, mu_s = model_mean("grouped-pretrained"), model_mean("grouped-scratch")
        sd_p, sd_s = model_sd("grouped-pretrained"), model_sd("grouped-scratch")
        best_p = max(mu_p, key=mu_p.get)
        best_s = max(mu_s, key=mu_s.get)

        pooled_clause = ""
        if pooled_table is not None:
            pp_v, ps_v = pooled_vals("grouped-pretrained"), pooled_vals("grouped-scratch")
            pooled_clause = (f" Pooling the folds instead of averaging them gives"
                             f" {min(pp_v):.3f}--{max(pp_v):.3f} and"
                             f" {min(ps_v):.3f}--{max(ps_v):.3f}"
                             r" (Table~\ref{tab:pooled}).")

        results_paras.append(
            r"\paragraph{Accuracy on eggs the detector has never seen.}"
            f" Averaged over the {k} folds, the four detectors reach mAP50"
            f" {min(mu_p.values()):.3f}--{max(mu_p.values()):.3f} from COCO-pretrained weights"
            f" and {min(mu_s.values()):.3f}--{max(mu_s.values()):.3f} from random"
            r" initialisation (Table~\ref{tab:main})." + pooled_clause +
            " The qualitative claim the dataset was collected to support therefore holds under"
            " a protocol that never scores a model on an egg it was trained on: a GelSight"
            " contact image carries enough signal to localise and classify a shell crack."
            " What does not hold is the precision with which that result was previously"
            " quoted."
        )

        if best_p != best_s:
            rank_clause = (f" The leading architecture is {MODEL_LABEL[best_p]} in the first"
                           f" case and {MODEL_LABEL[best_s]} in the second: changing the"
                           " initialisation reorders them.")
        else:
            rank_clause = (f" {MODEL_LABEL[best_p]} leads under both initialisations, but by"
                           " less than a single fold's worth of variation.")
        results_paras.append(
            r"\paragraph{The four detectors are not separable.}"
            f" Fold-to-fold standard deviation of mAP50 is"
            f" {min(sd_p.values()):.3f}--{max(sd_p.values()):.3f} with pretrained weights and"
            f" {min(sd_s.values()):.3f}--{max(sd_s.values()):.3f} from scratch, against a"
            f" spread between the detector means of"
            f" {max(mu_p.values()) - min(mu_p.values()):.2f} and"
            f" {max(mu_s.values()) - min(mu_s.values()):.2f} respectively." + rank_clause +
            " Which model wins on this dataset is a statement about which eggs fell in which"
            " split, so we report the four as a baseline range and draw no architectural"
            " conclusion. This is the comparison a single split is least equipped to make and"
            " most likely to appear to settle."
        )

        results_paras.append(
            r"\paragraph{The natural-image prior is not doing the work.}"
            f" Under the grouped protocol, pretrained initialisation averages {fmt(gp)} mAP50"
            f" against {fmt(gs)} from random weights, a gap of {abs(gp - gs):.2f} that sits"
            " inside the fold-to-fold variation; on the released split the ordering is the"
            f" other way round ({fmt(ps)} from scratch against {fmt(pp)} pretrained). Neither"
            " initialisation is reliably ahead, so the released recipe's 350 epochs from"
            " random weights costs nothing in accuracy, and the appearance of a crack in a"
            f" gel image is evidently learnable from the {rng(tr)} training frames a fold"
            " provides without transfer from natural images. The practical reading is that"
            " this task is limited by how many distinct eggs have been touched, not by"
            " initialisation or by the choice among recent YOLO variants."
        )

        r_int = class_stat("grouped-pretrained", "normal_egg", "recall") + \
            class_stat("grouped-scratch", "normal_egg", "recall")
        r_crk = class_stat("grouped-pretrained", "broken_egg", "recall") + \
            class_stat("grouped-scratch", "broken_egg", "recall")
        a_int = class_stat("grouped-pretrained", "normal_egg", "mAP50") + \
            class_stat("grouped-scratch", "normal_egg", "mAP50")
        a_crk = class_stat("grouped-pretrained", "broken_egg", "mAP50") + \
            class_stat("grouped-scratch", "broken_egg", "mAP50")
        pub_crk = [res[a][m][0]["test"]["per_class"]["broken_egg"]["recall"]
                   for a in ("published-scratch", "published-pretrained") for m in MODELS
                   if "broken_egg" in res[a][m][0]["test"]["per_class"]]
        if r_int and r_crk:
            results_paras.append(
                r"\paragraph{The harder class is the one that matters.}"
                " The aggregate mAP50 averages over both classes and so flatters the task."
                f" Across every grouped run, intact eggs are recalled at {statistics.mean(r_int):.3f}"
                f" on average but cracked eggs at only {statistics.mean(r_crk):.3f}, and in the"
                f" worst fold cracked-egg recall falls to {min(r_crk):.3f} -- roughly half the"
                " cracks in that held-out set are missed. Average precision shows the same"
                f" ordering ({statistics.mean(a_int):.3f} against {statistics.mean(a_crk):.3f})."
                " On the released split the same ordering appears but far more gently:"
                f" cracked-egg recall averages {statistics.mean(pub_crk):.3f} there and never"
                f" falls below {min(pub_crk):.3f}. The leaky split thus overstates the average"
                " and conceals the tail, which is the failure mode that matters: in grading a"
                " missed crack ships a defective egg, while a false alarm costs one good egg."
                " Any deployment of this modality should be tuned on cracked-egg recall"
                " rather"
                " than on mAP50, and reported per class."
            )

    def doc(header, paras):
        return header + "\n" + "\n\n".join(
            textwrap.fill(p, width=92, break_long_words=False, break_on_hyphens=False)
            for p in paras
        ) + "\n"

    outputs = [
        ("acquisition_table.tex", tab1 + "\n"),
        ("folds_table.tex", tab2 + "\n"),
        ("protocol.tex", doc(r"% --- replaces the split description in the experimental setup ---",
                             protocol_paras)),
        ("limitations.tex", doc(r"% --- limitations ---", limitation_paras)),
    ]
    if provenance_paras:
        outputs.append(("provenance.tex", doc(r"% --- what the previously reported numbers measure ---",
                                              provenance_paras)))
    if effect_paras:
        outputs.append(("leakage_effect.tex", doc(r"% --- size of the protocol effect ---",
                                                  effect_paras)))
    if results_table is not None:
        outputs.append(("results_table.tex", results_table + "\n"))
    if pooled_table is not None:
        outputs.append(("pooled_table.tex", pooled_table + "\n"))
    if results_paras:
        outputs.append(("results.tex", doc(r"% --- results ---", results_paras)))

    for name, body in outputs:
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(notation(body))
        print(f"wrote paper/{name}")


if __name__ == "__main__":
    main()
