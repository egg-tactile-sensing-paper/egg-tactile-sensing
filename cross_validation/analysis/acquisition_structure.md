# Acquisition structure of the tactile dataset

377 images, 387 annotated instances (207 normal_egg, 180 broken_egg).
Published split: train 264, val 75, test 38.

Frames within one contact are 1.43 s apart (median); a gap above 30 s separates two contacts. That leaves **19 contact episodes**, every one of a single class.

| episode | start (UTC) | images | class | duration (s) | published split |
| --- | --- | --- | --- | --- | --- |
| 1 | 2024-12-05 14:52:28 | 7 | normal_egg | 30.7 | train 6, test 1 |
| 2 | 2024-12-05 14:56:40 | 8 | normal_egg | 17.9 | train 7, val 1 |
| 3 | 2024-12-05 14:58:49 | 12 | normal_egg | 35.2 | train 8, val 2, test 2 |
| 4 | 2024-12-05 15:04:08 | 1 | normal_egg | 0.0 | train 1 |
| 5 | 2024-12-05 15:09:27 | 17 | normal_egg | 72.0 | train 14, val 1, test 2 |
| 6 | 2024-12-05 15:15:44 | 14 | normal_egg | 31.6 | train 9, val 4, test 1 |
| 7 | 2024-12-05 15:26:52 | 34 | normal_egg | 114.2 | train 25, val 7, test 2 |
| 8 | 2024-12-05 15:39:12 | 44 | normal_egg | 129.6 | train 29, val 11, test 4 |
| 9 | 2024-12-05 15:48:33 | 37 | normal_egg | 107.7 | train 25, val 6, test 6 |
| 10 | 2024-12-05 16:01:47 | 32 | normal_egg | 85.2 | train 20, val 9, test 3 |
| 11 | 2024-12-05 16:22:01 | 9 | broken_egg | 22.2 | train 7, val 1, test 1 |
| 12 | 2024-12-06 13:50:00 | 13 | broken_egg | 46.0 | train 5, val 7, test 1 |
| 13 | 2024-12-06 13:59:52 | 12 | broken_egg | 28.0 | train 10, val 2 |
| 14 | 2024-12-06 14:08:58 | 18 | broken_egg | 52.8 | train 14, val 1, test 3 |
| 15 | 2024-12-06 14:16:12 | 16 | broken_egg | 46.5 | train 13, val 2, test 1 |
| 16 | 2024-12-06 14:43:13 | 38 | broken_egg | 102.8 | train 21, val 14, test 3 |
| 17 | 2024-12-06 15:14:47 | 31 | broken_egg | 80.0 | train 24, val 3, test 4 |
| 18 | 2024-12-06 15:31:51 | 16 | broken_egg | 42.2 | train 12, val 3, test 1 |
| 19 | 2024-12-06 15:37:34 | 18 | broken_egg | 44.5 | train 14, val 1, test 3 |

## Audit of the published split

- Episodes appearing in more than one split: **18 of 19**.
- Test images that share an episode with a training or validation frame: **38 of 38**.
- Byte-identical image files: **0**; repeated timestamps: **0** (so the export contains no duplicated or augmented copies).
- Independent units actually available: **19** (10 whole, 9 cracked), not 377.

A random image-level split therefore places near-duplicate frames of the same physical
egg on both sides of the partition, so the reported scores measure recall of seen eggs
rather than generalisation to new ones.

## Acquisition-time confound

- normal_egg: 2024-12-05 14:52:28 to 2024-12-05 16:01:47
- broken_egg: 2024-12-05 16:22:01 to 2024-12-06 15:37:34

**Every whole-egg episode precedes every cracked-egg episode**, so class is confounded with session: sensor drift, gel wear or lighting changes between sessions are a sufficient cue for the detector. Grouped cross-validation does not remove this; only interleaved re-acquisition of both classes within each session does.

