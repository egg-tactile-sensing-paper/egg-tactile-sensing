"""Describe the acquisition structure of the dataset and audit the published split.

Writes a per-episode table and the leakage figures that motivate grouped evaluation.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import statistics

from dataset import CLASS_NAMES, EPISODE_GAP_S, build_episodes, load_frames


def utc(ts_ms: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--gap", type=float, default=EPISODE_GAP_S)
    parser.add_argument("--out", default=None, help="directory for the generated tables")
    args = parser.parse_args()

    frames = load_frames(args.root)
    episodes = build_episodes(frames, args.gap)
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    gaps = [(b.ts_ms - a.ts_ms) / 1000 for a, b in zip(frames, frames[1:])]
    within = [g for g in gaps if g <= args.gap]
    per_split = collections.Counter(f.split for f in frames)
    instances = collections.Counter(c for f in frames for c in f.classes)

    lines = ["# Acquisition structure of the tactile dataset", ""]
    lines.append(f"{len(frames)} images, {sum(instances.values())} annotated instances "
                 f"({instances[0]} {CLASS_NAMES[0]}, {instances[1]} {CLASS_NAMES[1]}).")
    lines.append(f"Published split: " + ", ".join(f"{k} {per_split[k]}" for k in ("train", "val", "test")) + ".")
    lines.append("")
    lines.append(f"Frames within one contact are {statistics.median(within):.2f} s apart (median); "
                 f"a gap above {args.gap:g} s separates two contacts. "
                 f"That leaves **{len(episodes)} contact episodes**, every one of a single class.")
    lines.append("")

    # Per-episode table: the evidence that the dataset has far fewer independent units
    # than it has images.
    lines += ["| episode | start (UTC) | images | class | duration (s) | published split |",
              "| --- | --- | --- | --- | --- | --- |"]
    for e in episodes:
        split_counts = collections.Counter(f.split for f in e.frames)
        spread = ", ".join(f"{k} {split_counts[k]}" for k in ("train", "val", "test") if split_counts[k])
        lines.append(f"| {e.index + 1} | {utc(e.start_ms)} | {e.size} | {CLASS_NAMES[e.cls]} "
                     f"| {e.duration_s:.1f} | {spread} |")
    lines.append("")

    # Rule out the cruder kind of leakage first: exported augmentations or re-used frames
    # would show up as repeated timestamps or byte-identical files.
    digests = collections.defaultdict(list)
    for frame in frames:
        with open(frame.image, "rb") as handle:
            digests[hashlib.md5(handle.read()).hexdigest()].append(frame.stem)
    duplicate_files = sum(1 for stems in digests.values() if len(stems) > 1)
    repeated_timestamps = len(frames) - len({f.ts_ms for f in frames})

    # Leakage audit of the published split.
    multi = [e for e in episodes if len(e.splits) > 1]
    test_contaminated = sum(
        1 for e in episodes for f in e.frames if f.split == "test" and e.splits - {"test"}
    )
    n_test = per_split["test"]
    by_class = collections.Counter(CLASS_NAMES[e.cls] for e in episodes)
    lines += [
        "## Audit of the published split",
        "",
        f"- Episodes appearing in more than one split: **{len(multi)} of {len(episodes)}**.",
        f"- Test images that share an episode with a training or validation frame: "
        f"**{test_contaminated} of {n_test}**.",
        f"- Byte-identical image files: **{duplicate_files}**; repeated timestamps: "
        f"**{repeated_timestamps}** (so the export contains no duplicated or augmented copies).",
        f"- Independent units actually available: **{len(episodes)}** "
        f"({by_class[CLASS_NAMES[0]]} whole, {by_class[CLASS_NAMES[1]]} cracked), not {len(frames)}.",
        "",
        "A random image-level split therefore places near-duplicate frames of the same physical",
        "egg on both sides of the partition, so the reported scores measure recall of seen eggs",
        "rather than generalisation to new ones.",
        "",
    ]

    # Acquisition-time confound: are the two classes separated in time?
    spans = {}
    for cls in (0, 1):
        members = [e for e in episodes if e.cls == cls]
        spans[CLASS_NAMES[cls]] = (utc(min(e.start_ms for e in members)),
                                   utc(max(e.start_ms for e in members)))
    lines += ["## Acquisition-time confound", ""]
    for name, (first, last) in spans.items():
        lines.append(f"- {name}: {first} to {last}")
    overlap = (min(e.start_ms for e in episodes if e.cls == 1)
               < max(e.start_ms for e in episodes if e.cls == 0))
    lines += [
        "",
        ("The two classes overlap in time." if overlap else
         "**Every whole-egg episode precedes every cracked-egg episode**, so class is confounded "
         "with session: sensor drift, gel wear or lighting changes between sessions are a "
         "sufficient cue for the detector. Grouped cross-validation does not remove this; only "
         "interleaved re-acquisition of both classes within each session does."),
        "",
    ]

    report = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "acquisition_structure.md"), "w") as handle:
        handle.write(report)

    summary = {
        "images": len(frames),
        "instances": sum(instances.values()),
        "instances_per_class": {CLASS_NAMES[c]: n for c, n in sorted(instances.items())},
        "published_split": dict(per_split),
        "episode_gap_s": args.gap,
        "episodes": len(episodes),
        "episodes_per_class": dict(by_class),
        "episodes_in_multiple_splits": len(multi),
        "test_images_sharing_an_episode": test_contaminated,
        "median_within_contact_gap_s": round(statistics.median(within), 2),
    }
    with open(os.path.join(out_dir, "acquisition_structure.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(report)
    print(f"[analyze] wrote {out_dir}/acquisition_structure.{{md,json}}")


if __name__ == "__main__":
    main()
