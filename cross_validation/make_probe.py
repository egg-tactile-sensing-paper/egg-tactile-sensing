"""Build the session-confound probe.

The limitation we report is that class is confounded with acquisition session: every intact
episode was recorded before every cracked episode.  That is an argument, not a measurement.
This probe turns it into one.

Episode 11 is the single cracked egg recorded in the *intact* block (Dec 5 16:22, nineteen
minutes after the last intact episode ends); every other cracked episode is from the next
day.  So train on intact Dec 5 + cracked Dec 6 and then ask two questions of the same model:

    arm "cross"  test = episode 11        cracked, recorded in the intact session
    arm "same"   test = episodes 18, 19   cracked, recorded in the cracked session

Both test sets are cracked-only and the training data is identical, so the only difference is
which session the test egg came from.  A detector that has learned crack morphology scores
alike on both.  A detector that has learned "this session looks intact" fails on "cross" and
not on "same" -- which is the shortcut we cannot rule out by any partition of this dataset.

Episode 11 is one egg (9 frames), so this reads as a direction and an effect size, not a
precise estimate.  It is still the only cross-session cracked egg the dataset contains.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, Episode, build_episodes, load_frames

# 1-based episode numbers, matching folds.json and the paper's tables.
TRAIN_EPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16)
VAL_EPS = (9, 17)
ARMS = {"cross": (11,), "same": (18, 19)}


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def summarise(members: list[Episode]) -> dict:
    per_class = collections.Counter(CLASS_NAMES[e.cls] for e in members)
    return {
        "episodes": [e.index + 1 for e in members],
        "n_episodes": len(members),
        "n_images": sum(e.size for e in members),
        "episodes_per_class": dict(per_class),
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.dirname(here))
    parser.add_argument("--out", default=os.path.join(here, "probe"))
    args = parser.parse_args()

    episodes = {e.index + 1: e for e in build_episodes(load_frames(args.root))}

    # The probe is only meaningful if the arms really do differ in session and agree in class.
    assert CLASS_NAMES[episodes[11].cls] == "broken_egg", "episode 11 is not a cracked egg"
    for arm, eps in ARMS.items():
        classes = {CLASS_NAMES[episodes[i].cls] for i in eps}
        assert classes == {"broken_egg"}, f"arm {arm} is not cracked-only: {classes}"
    train_cracked = [i for i in TRAIN_EPS if CLASS_NAMES[episodes[i].cls] == "broken_egg"]
    day = lambda i: episodes[i].start_ms // 86_400_000
    assert day(11) != day(train_cracked[0]), "episode 11 shares a day with the cracked training eggs"
    assert all(day(i) == day(train_cracked[0]) for i in train_cracked), "cracked training eggs span days"
    assert all(day(i) == day(train_cracked[0]) for i in ARMS["same"]), "the 'same' arm is not in the cracked session"
    overlap = set(TRAIN_EPS) & set(VAL_EPS) | set(TRAIN_EPS) & {i for e in ARMS.values() for i in e}
    assert not overlap, f"episodes used twice: {sorted(overlap)}"

    manifest = {"arms": {}, "train": summarise([episodes[i] for i in TRAIN_EPS]),
                "val": summarise([episodes[i] for i in VAL_EPS])}
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)

    for arm, test_eps in ARMS.items():
        arm_dir = os.path.join(args.out, arm)
        parts = {"train": TRAIN_EPS, "val": VAL_EPS, "test": test_eps}
        for split, eps in parts.items():
            image_dir = os.path.join(arm_dir, "images", split)
            label_dir = os.path.join(arm_dir, "labels", split)
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            for i in eps:
                for frame in episodes[i].frames:
                    link(frame.image, os.path.join(image_dir, os.path.basename(frame.image)))
                    link(frame.label, os.path.join(label_dir, os.path.basename(frame.label)))

        with open(os.path.join(arm_dir, "data.yaml"), "w") as handle:
            handle.write(
                f"# session-confound probe, arm '{arm}': test episodes {list(test_eps)}\n"
                f"path: {os.path.abspath(arm_dir)}\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items()))
            )

        rec = summarise([episodes[i] for i in test_eps])
        rec["data_yaml"] = os.path.relpath(os.path.join(arm_dir, "data.yaml"), args.root)
        rec["session_day"] = sorted({int(day(i)) for i in test_eps})
        manifest["arms"][arm] = rec
        print(f"  {arm:<6} test eps {list(test_eps)} -> {rec['n_images']:>3} images "
              f"({', '.join(f'{k} {v}' for k, v in rec['episodes_per_class'].items())})")

    with open(os.path.join(args.out, "probe.json"), "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"[probe] train {manifest['train']['n_images']} images "
          f"({manifest['train']['episodes_per_class']}), "
          f"val {manifest['val']['n_images']} images -> {args.out}")


if __name__ == "__main__":
    main()
