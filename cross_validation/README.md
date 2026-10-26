# Grouped cross-validation for the egg tactile dataset

The published release splits the 377 tactile images at random into 264/75/38. The filenames
carry the acquisition time (`gelsight-frame-<unix_ms>_...`), and reconstructing it shows that
the images are **19 contact episodes**, not 377 independent samples: within an episode the
frames are ~1.5 s apart and show the same physical egg. 18 of the 19 episodes appear in more
than one split, and **all 38 test images share an episode with a training or validation
frame**, so the published scores measure recall of eggs the model has already seen.

This directory re-evaluates the four baselines with the egg as the unit of splitting. Every
number in the paper's Tables 1-4 is regenerated from the files here; nothing is typed in.

Episodes are inferred from timestamps. They have **not** been independently verified as
distinct physical eggs, and the acquisition schedule was not interleaved across classes
(every whole-egg episode precedes every cracked-egg episode), so session effects cannot be
excluded as a predictive cue. `make_probe.py` measures what little the data allows on that
point; the paper's Limitations section states both caveats.

## Protocol

- **Group** = contact episode (gap > 30 s starts a new one). Every episode is class-pure.
- **Outer**: 5 folds, stratified by class, balanced on both episode and image count. Each
  fold holds out 3-4 eggs, 63-84 images; every egg is held out exactly once.
- **Inner**: the validation set used for checkpoint selection is cut from the *training*
  episodes of that fold (~20 % of them), so the held-out eggs are never used to pick a model.
- **Reported**: mean ± sd over the five folds, per-fold scores, and a pooled score over the
  union of the held-out images.

The fold assignment contains no randomness: `dataset.snake_assign` is a pure function of the
episode sizes and start times, so `make_folds.py` produces the same five folds on any machine.

Four arms separate the protocol effect from the initialisation effect:

| arm | split | init | epochs | runs |
| --- | --- | --- | --- | --- |
| `grouped-pretrained` | 5 grouped folds | COCO weights | 100 | 20 |
| `grouped-scratch` | 5 grouped folds | random init (as published) | 350 | 20 |
| `published-pretrained` | the release's 70/20/10 | COCO weights | 100 | 4 |
| `published-scratch` | the release's 70/20/10 | random init (as published) | 350 | 4 |
| `probe-cross` / `probe-same` | session-confound probe | COCO weights | 100 | 8 |

`published-scratch` reproduces the setting of the original `main_code.py`
(`YOLO('yolov9m.yaml')`, 350 epochs), so the comparison with `grouped-scratch` isolates
exactly what the evaluation protocol is worth. Within an arm every run shares `imgsz 640`,
`batch 16`, `seed 0` and the Ultralytics default optimiser and augmentation settings; the two
initialisations differ in epoch budget as well as in weights, so their difference is a
statement about the two recipes rather than about pretraining alone.

## Environment

Python 3.10.17, CUDA 12.1, one GPU per run (NVIDIA A100 or L4 class), peak GPU memory under
10 GB at batch 16. See `requirements.txt` for pinned versions; the ones that matter are
`torch==2.5.1+cu121` and `ultralytics==8.4.189`.

```bash
python -m venv venv && . venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r cross_validation/requirements.txt
```

`train_fold.py` runs Ultralytics offline, which is how every reported run was produced: it
expects `yolov8m.pt`, `yolov9m.pt`, `yolov10m.pt`, `yolo11m.pt` and `yolo11n.pt` (the last for
the AMP check) to be staged in `--weights-dir` beforehand, from
`https://github.com/ultralytics/assets/releases/download/v8.3.0`. Pass `--allow-download` to
let Ultralytics fetch them instead.

The 56 reported runs took **12.6 GPU-hours** in total: median 7.5 min for a 100-epoch run and
23.1 min for a 350-epoch run.

## Running it

All commands are run from the repository root.

```bash
python cross_validation/analyze_splits.py   # acquisition structure and leakage audit
python cross_validation/make_folds.py       # builds folds/fold{0..4} and folds.json
python cross_validation/make_probe.py       # builds the session-confound probe
python cross_validation/make_runlist.py     # writes a runlist; the three used are committed

# one run; repeat over a runlist's rows (the committed results were produced this way,
# one SLURM array task per row)
while IFS=$'\t' read -r model tag data init arm epochs; do
  [ "${model#\#}" = "$model" ] || continue                      # skip the header
  out=cross_validation/results/$model-$tag-$init.json
  [ -f "$out" ] && continue                                     # already done
  python cross_validation/train_fold.py \
    --model "$model" --tag "$tag" --data "$data" --init "$init" --epochs "${epochs:-100}"
done < cross_validation/runs_scratch.tsv

python cross_validation/collect.py          # tables into analysis/
python cross_validation/make_paper_tex.py   # LaTeX fragments into paper/
```

The `[ -f "$out" ] && continue` test is what let the runlists be re-driven after a timeout
without repeating finished work; `train_fold.py` itself does not check, and will overwrite.
`collect.py` and `make_paper_tex.py` read
`results/` and need no GPU, so the paper's tables can be rebuilt from this repository alone:

```bash
python cross_validation/collect.py && python cross_validation/make_paper_tex.py
```

## Files

| file | purpose |
| --- | --- |
| `dataset.py` | reads the release, recovers contact episodes from the timestamps |
| `analyze_splits.py` | episode table, leakage audit, acquisition-time confound |
| `make_folds.py` | builds the grouped folds as symlink trees plus `data.yaml` |
| `make_runlist.py` | generates a runlist (`--arms`, `--out`) |
| `train_fold.py` | trains one (model, fold) and records **both** val and test metrics |
| `collect.py` | mean ± sd, per-fold and pooled tables, in Markdown and LaTeX; caches the pooled scores to `analysis/pooled.json` |
| `make_probe.py` | builds the session-confound probe (cross- vs same-session cracked egg) |
| `make_paper_tex.py` | the manuscript's LaTeX fragments, numbers sourced from data and runs |
| `folds/folds.json` | fold composition: which episodes are train/val/test in each fold |
| `folds/episode_assignments.csv` | per-image manifest: episode, class, released split, and the role of that image in each of the five folds |
| `probe/probe.json` | the probe's two arms |
| `runs_*.tsv` | the exact runlists that were executed (see below) |
| `results/` | one JSON per run: config, host, wall-clock, and val **and** test metrics overall and per class, at full precision, plus the raw per-image detections |
| `analysis/` | generated reports, including `pooled.json` |
| `paper/` | generated LaTeX fragments for the manuscript |

`runs.tsv` is the original 48-run matrix and is **superseded** by the three runlists that were
actually executed:

| runlist | arm | runs | config |
| --- | --- | --- | --- |
| `runs_lean.tsv` | grouped-pretrained | 20 | COCO weights, 100 ep, grouped 5-fold CV |
| `runs_lean.tsv` | published-pretrained | 4 | COCO weights, 100 ep, released split |
| `runs_extra.tsv` | probe-cross / probe-same | 8 | session-confound probe, 100 ep |
| `runs_extra.tsv` | published-scratch | 4 | **the released config**: random init, 350 ep |
| `runs_scratch.tsv` | grouped-scratch | 20 | released config under grouped CV |

The crossed protocol x initialisation design is what lets the protocol effect be read within
one training recipe. `runs_extra.tsv` and `runs_scratch.tsv` carry a sixth column giving each
run's epoch budget; `runs_lean.tsv` has no such column and was driven with a single epoch
setting.

### Not in this repository

The fold and probe image trees (`folds/fold*/images`, `probe/*/images`) are symlinks into
`data/`, so only the manifests above are committed -- `make_folds.py` and `make_probe.py`
rebuild the trees. Checkpoints, Ultralytics run directories and the virtual environment are
also left out; `train_fold.py` writes them under `$TACTILE_WORK` (default
`cross_validation/work/`). The SLURM array wrappers that drove the runs on our cluster are not
included either, since they encode one site's partitions and account; the loop above does the
same work. The `results/*.json` files have the run environment's hostname, scheduler job id and
absolute paths removed; every metric, the training configuration and the image counts are
exactly as written by `train_fold.py`.

## Paper fragments

`make_paper_tex.py` writes `paper/*.tex` from `dataset.py`, `folds/folds.json`,
`probe/probe.json`, `analysis/pooled.json` and the result JSON, so every number in the
manuscript traces back to a file rather than being typed in.

| fragment | paper | needs | content |
| --- | --- | --- | --- |
| `acquisition_table.tex` | Table 3 | dataset only | the 19 episodes and how the released split scatters each |
| `folds_table.tex` | Table 1 | folds only | fold composition |
| `protocol.tex` | §3.2 | folds only | unit of evaluation, leakage audit, grouped CV, the two aggregates, the crossed arms |
| `provenance.tex` | §3.3 | published-scratch | the released table reports validation, not test, metrics |
| `results_table.tex` | Table 2 | all four main arms | scores under each protocol x initialisation |
| `pooled_table.tex` | Table 4 | both grouped arms | pooled estimates (needs `analysis/pooled.json`) |
| `results.tex` | §3.3 | all four main arms | fold spread, detector non-separability, initialisation, per-class asymmetry |
| `leakage_effect.tex` | §3.3 | all four main arms | the protocol effect within a fixed recipe |
| `limitations.tex` | §4 | probe arms | confound probe, its caveats, scale |

`acquisition_table`, `folds_table` and the first half of `protocol` need no training run at all.
Everything that quotes a run appears only once its arms are **complete**: `arm_mean` returns
`None` for a partially finished arm rather than averaging whichever folds landed first, and the
fragment is then not written.

Precision rule: a quantity measured directly (an arm mean, a per-model mean or sd, a pooled
score) is printed to **three** decimals; a *difference or ratio of two such printed quantities*
is printed to **two**. Rounding the operands first and subtracting those -- which this code did
until 2026-10-26 -- made every quoted difference disagree with the raw runs by up to 0.001:
`0.919 - 0.880` prints as 0.039 while the true drop is 0.0399, i.e. 0.040, and the premium ratio
came out 2.1 instead of 2.0. Differencing at full precision and quoting one decimal fewer is
correct both ways round -- the reader gets the same answer whether they recompute from
`results/*.json` or subtract the two numbers printed beside it. `collect.py` writes the raw
analysis tables separately.
