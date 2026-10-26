"""Build grouped, class-stratified cross-validation folds.

The grouping unit is the contact episode recovered in dataset.py, so no frame of a held-out
egg is ever seen during training.  Each outer fold also carves its own validation set out of
its *training* episodes, so checkpoint selection never touches the held-out eggs either.

Layout produced for every fold (symlinks, so the images are stored once):

    folds/fold<k>/images/{train,val,test}/...
    folds/fold<k>/labels/{train,val,test}/...
    folds/fold<k>/data.yaml

Also written, and the only parts of folds/ that are committed: folds.json, the episode
composition of each fold, and episode_assignments.csv, one row per image giving its episode,
class, released split and role in each fold.  Those two carry the whole protocol; the trees
above are symlinks into data/ and are rebuilt by running this script.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil

from dataset import CLASS_NAMES, EPISODE_GAP_S, Episode, build_episodes, load_frames, snake_assign


def assign(episodes: list[Episode], k: int) -> dict[int, int]:
    """Stratified grouping: snake-assign each class separately, then merge."""
    assignment: dict[int, int] = {}
    for cls in sorted(CLASS_NAMES):
        members = [e for e in episodes if e.cls == cls]
        assignment.update(snake_assign(members, k))
    return assignment


def pick_validation(pool: list[Episode], inner: dict[int, int], k: int,
                    fraction: float) -> list[Episode]:
    """Use the inner group that is closest to the target size and still holds both classes.

    The snake order puts the largest episodes in group 0, so taking that group by default
    would spend the biggest eggs on checkpoint selection; choosing by size keeps the
    validation set near the intended fraction instead.
    """
    target = fraction * sum(e.size for e in pool)
    candidates = []
    for group in range(k):
        members = [e for e in pool if inner[e.index] == group]
        if len({e.cls for e in members}) < len(CLASS_NAMES):
            continue
        candidates.append((abs(sum(e.size for e in members) - target), group, members))
    if not candidates:
        raise ValueError("no inner group contains both classes")
    return min(candidates)[2]


def link(src: str, dst: str) -> None:
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--folds", type=int, default=5, help="outer folds (one held-out egg set each)")
    parser.add_argument("--inner", type=int, default=5, help="inner groups cut from the training pool")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="target share of the training pool used for checkpoint selection")
    parser.add_argument("--gap", type=float, default=EPISODE_GAP_S)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    episodes = build_episodes(load_frames(args.root), args.gap)
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "folds")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    outer = assign(episodes, args.folds)
    manifest = {
        "episode_gap_s": args.gap,
        "episodes": len(episodes),
        "outer_folds": args.folds,
        "inner_groups": args.inner,
        "classes": CLASS_NAMES,
        "folds": [],
    }

    roles: dict[int, dict[int, str]] = {}   # fold -> episode index -> train/val/test

    header = f"{'fold':>4} | {'train eps/imgs':>16} | {'val eps/imgs':>14} | {'test eps/imgs':>15} | test episodes"
    print(header)
    print("-" * len(header))

    for fold in range(args.folds):
        held_out = [e for e in episodes if outer[e.index] == fold]
        pool = [e for e in episodes if outer[e.index] != fold]
        inner = assign(pool, args.inner)
        val = pick_validation(pool, inner, args.inner, args.val_fraction)
        chosen = {e.index for e in val}
        train = [e for e in pool if e.index not in chosen]
        roles[fold] = {e.index: split for split, members in
                       (("train", train), ("val", val), ("test", held_out)) for e in members}

        fold_dir = os.path.join(out_dir, f"fold{fold}")
        for split, members in (("train", train), ("val", val), ("test", held_out)):
            image_dir = os.path.join(fold_dir, "images", split)
            label_dir = os.path.join(fold_dir, "labels", split)
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            for episode in members:
                for frame in episode.frames:
                    link(frame.image, os.path.join(image_dir, os.path.basename(frame.image)))
                    link(frame.label, os.path.join(label_dir, os.path.basename(frame.label)))

        with open(os.path.join(fold_dir, "data.yaml"), "w") as handle:
            handle.write(
                f"# grouped CV fold {fold} of {args.folds}: the test episodes are unseen eggs\n"
                f"path: {os.path.abspath(fold_dir)}\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items()))
            )

        def summarise(members: list[Episode]) -> dict:
            per_class = collections.Counter(CLASS_NAMES[e.cls] for e in members)
            return {
                "episodes": [e.index + 1 for e in members],
                "n_episodes": len(members),
                "n_images": sum(e.size for e in members),
                "episodes_per_class": dict(per_class),
            }

        record = {
            "fold": fold,
            "train": summarise(train),
            "val": summarise(val),
            "test": summarise(held_out),
            "data_yaml": os.path.relpath(os.path.join(fold_dir, "data.yaml"), args.root),
        }
        manifest["folds"].append(record)
        print(f"{fold:>4} | {record['train']['n_episodes']:>5} / {record['train']['n_images']:>8} "
              f"| {record['val']['n_episodes']:>4} / {record['val']['n_images']:>7} "
              f"| {record['test']['n_episodes']:>5} / {record['test']['n_images']:>7} "
              f"| {record['test']['episodes']} "
              f"{record['test']['episodes_per_class']}")

    # The published split, rewritten as its own data.yaml, so the two protocols can be run
    # through exactly the same training code and compared like for like.  The manifest
    # records paths relative to the repository root; the data.yaml files themselves hold
    # absolute paths, because that is what ultralytics resolves reliably.
    published = os.path.join(out_dir, "published")
    os.makedirs(published, exist_ok=True)
    with open(os.path.join(published, "data.yaml"), "w") as handle:
        handle.write(
            "# the release's own 70/20/10 image-level split, for the leakage comparison\n"
            f"path: {os.path.abspath(os.path.join(args.root, 'data'))}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items()))
        )
    manifest["published_data_yaml"] = os.path.relpath(
        os.path.join(published, "data.yaml"), args.root)

    # Every episode must be held out exactly once, and no episode may take two roles in a fold.
    held = [i for record in manifest["folds"] for i in record["test"]["episodes"]]
    assert sorted(held) == list(range(1, len(episodes) + 1)), "episodes are not partitioned by the folds"
    for record in manifest["folds"]:
        train_eps, val_eps, test_eps = (set(record[s]["episodes"]) for s in ("train", "val", "test"))
        assert not (train_eps & val_eps) and not (train_eps & test_eps) and not (val_eps & test_eps)

    with open(os.path.join(out_dir, "folds.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    # One row per image: everything needed to reconstruct both protocols without this script.
    assignments = os.path.join(out_dir, "episode_assignments.csv")
    with open(assignments, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "episode", "class", "published_split"] +
                        [f"fold{k}" for k in range(args.folds)])
        for episode in sorted(episodes, key=lambda e: e.start_ms):
            for frame in episode.frames:
                writer.writerow([os.path.basename(frame.image), episode.index + 1,
                                 CLASS_NAMES[episode.cls], frame.split] +
                                [roles[k][episode.index] for k in range(args.folds)])
    print(f"\n[folds] {args.folds} folds under {out_dir}; manifest in folds.json,"
          f" per-image assignments in episode_assignments.csv")


if __name__ == "__main__":
    main()
