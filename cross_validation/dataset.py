"""Shared dataset handling: read the published splits and recover contact episodes.

Every image file is named gelsight-frame-<unix_ms>_jpeg.rf.<roboflow_hash>.jpg, so the
acquisition time of each frame survives the Roboflow export.  Frames recorded within a
few seconds of each other belong to one continuous contact with one physical egg; those
bursts are the unit that must not be split across training and evaluation data.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

CLASS_NAMES = {0: "normal_egg", 1: "broken_egg"}
SPLITS = ("train", "val", "test")
TS_RE = re.compile(r"gelsight-frame-(\d+)_")

# Frames more than this far apart start a new contact episode.  The gap histogram is
# strongly bimodal (within a contact: ~1.5 s; between contacts: minutes), so anything
# from 10 s to several minutes recovers the same 19 episodes.
EPISODE_GAP_S = 30.0


@dataclass
class Frame:
    stem: str
    split: str          # the split this frame has in the published release
    ts_ms: int
    classes: tuple[int, ...]
    image: str
    label: str


@dataclass
class Episode:
    index: int
    frames: list[Frame] = field(default_factory=list)

    @property
    def cls(self) -> int:
        seen = {c for f in self.frames for c in f.classes}
        if len(seen) != 1:
            raise ValueError(f"episode {self.index} mixes classes {seen}")
        return next(iter(seen))

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def start_ms(self) -> int:
        return self.frames[0].ts_ms

    @property
    def duration_s(self) -> float:
        return (self.frames[-1].ts_ms - self.frames[0].ts_ms) / 1000.0

    @property
    def splits(self) -> set[str]:
        return {f.split for f in self.frames}


def load_frames(root: str) -> list[Frame]:
    """Read every frame of the published release, in acquisition order."""
    frames: list[Frame] = []
    for split in SPLITS:
        image_dir = os.path.join(root, "data", "images", split)
        label_dir = os.path.join(root, "data", "labels", split)
        for name in sorted(os.listdir(image_dir)):
            match = TS_RE.match(name)
            if match is None:
                raise ValueError(f"{name} does not carry a timestamp")
            stem = os.path.splitext(name)[0]
            label = os.path.join(label_dir, stem + ".txt")
            with open(label) as handle:
                classes = tuple(int(line.split()[0]) for line in handle if line.split())
            frames.append(
                Frame(
                    stem=stem,
                    split=split,
                    ts_ms=int(match.group(1)),
                    classes=classes,
                    image=os.path.join(image_dir, name),
                    label=label,
                )
            )
    frames.sort(key=lambda f: f.ts_ms)
    return frames


def build_episodes(frames: list[Frame], gap_s: float = EPISODE_GAP_S) -> list[Episode]:
    """Cut the acquisition-ordered frames into contact episodes on the time gaps."""
    episodes: list[Episode] = [Episode(index=0, frames=[frames[0]])]
    for previous, current in zip(frames, frames[1:]):
        if (current.ts_ms - previous.ts_ms) / 1000.0 > gap_s:
            episodes.append(Episode(index=len(episodes)))
        episodes[-1].frames.append(current)
    for episode in episodes:
        episode.cls  # raises if an episode is not class-pure
    return episodes


def snake_assign(episodes: list[Episode], k: int) -> dict[int, int]:
    """Assign episodes to k groups, balancing both episode count and image count.

    Episodes go to the groups in a back-and-forth order after sorting by size, which
    keeps the group sizes close without any randomness: the assignment is a function of
    the data alone and is identical on every machine.
    """
    order = sorted(episodes, key=lambda e: (-e.size, e.start_ms))
    assignment: dict[int, int] = {}
    for position, episode in enumerate(order):
        lap, step = divmod(position, k)
        assignment[episode.index] = step if lap % 2 == 0 else k - 1 - step
    return assignment
