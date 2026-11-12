# Huawei TechArena 2026 - Phase 1 click prediction

Our seven-person student team took part in Huawei Ireland TechArena for the first time. We developed a recommendation click prediction system and did not qualify for Phase 2. This repository collects the code and experimental evidence from that work.

## Task and data

Each event presents ten candidate items with at most one click. The goal is a probability for every evaluation row. Ranking uses global row-level ROC AUC, including candidates in events with no click. Training contains 6,868,240 rows, evaluation 529,390. Histories contain up to twenty earlier clicks.

Obtain the official files through the [competition platform](https://platform-techarena-ireland.hackathon.com/#/event), subject to its access and usage terms. Place `train.csv` and `evaluation.csv` in the repository root. The training period is January 20–26, 2025 UTC. Evaluation contains each user's first event at or after January 27.

## Development and results

Work progressed from statistical and LightGBM baselines to history-aware user–item models, matrix factorization, temporal and contextual features, calibration and ensembles. Later experiments used impression-derived spectral embeddings, slate competition and group transitions. Experimental files are retained even when their results were rejected.

The recommended generated submission is `submission_techarena_v194_transitions.csv`. Its verification report is in the root. V194 builds on V37 and adds impression, history, candidate competition and group transition scores.

| V194 temporal cutoff, UTC | Fitting events | Validation events | Local ROC AUC |
| --- | ---: | ---: | ---: |
| January 24, 2025 | 392,831 | 47,928 | 0.787668786 |
| January 25, 2025 | 490,420 | 43,964 | 0.810415362 |
| January 26, 2025 | 588,873 | 35,138 | 0.829197905 |

These values are recorded under `v194_impressions` in `arena_runs/experiments.jsonl`. Validation fits strictly before the cutoff and selects the first subsequent event per user. The available later training events truncate each validation horizon. The first two runs use raw V37 scores, while January 26 uses its calibrated final scores. Their absolute AUCs therefore reflect different base pipelines and populations. Comparisons between candidates should use identical rows and base scores.

The validation frames were reused during extensive model selection. These are development estimates with temporal and model-selection limitations. V187 had a higher isolated January 26 score but weaker transfer to earlier cutoffs. Later V202–V204 experiments did not establish a replacement for V194.

We did not establish the target 0.834673. An official leaderboard screenshot shared during development showed 0.818246 for an earlier submission. Its exact artifact mapping and public/private status were not recorded. V194's official score is unknown. Local validation numbers must not be presented as leaderboard scores.

## Repository layout

- Root scripts preserve early data preparation, baseline training and validation work.
- `arena/` contains model implementations, versioned runners, tests, diagnostics and historical result notes through V204.
- `arena_runs/` contains lightweight configurations, metrics, logs and provenance. Prediction arrays and trained models are generated locally.
- `arena_reproductions/` retains lightweight records from a reproduction attempt.
- Root `*.verification.json` files describe generated submissions, including hashes and row coverage.

Historical documents, including `arena/README.md`, describe the selected version at their original writing time. This root README identifies the current publication recommendation. Absolute personal machine paths in newly imported text files have been replaced with project-relative paths or `<LOCAL_HOME>`. Historical provenance describes the original local execution, rather than included binary artifacts.

## Setup and reproduction

Python 3.12 was used locally. Create a virtual environment and install the dependencies. Versions are not pinned because a complete historical environment lock was not recorded. PyTorch and implicit availability depends on platform support.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python arena/prepare.py
```

`prepare.py` streams the original CSVs into `arena_data/`. It leaves an existing completed cache unchanged. Memory and runtime vary by experiment, especially for earlier scripts.

The V194 application stage requires previously generated V37 scores and event alignment arrays. With the historical upstream artifacts restored or regenerated, run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python arena/run_v194_submission.py
python arena/submission.py --predictions arena_runs/v194_evaluation.npy --out submission_techarena_v194_transitions.csv
```

The first command refuses to overwrite its existing evaluation prediction file. The second refuses to overwrite an existing CSV. Preserve previous artifacts before repeating a run.

A fresh clone does not include the upstream arrays needed by V194. V37 itself depends on V31, V32, V36 and V30 model outputs. Their configurations and runner commands are retained in `arena/V25_RESULTS.md`, `arena/V31_RESULTS.md`, `arena/V37_RESULTS.md` and `arena/run_v17_checks.py`. For example, the V31 runner uses `--variant decay --broad --full`, V32 uses `--variant adaptive --broad --full`, and V37 is assembled with `python arena/run_v37_join.py --out v37_repro --full` once its prerequisite artifacts exist. Output names must match downstream expectations. The project currently has no single clean-clone command that rebuilds this complete dependency chain; the application commands above require those prerequisites.

Lightweight checks can be run without retraining:

```sh
python -m unittest discover -s arena -p 'test_*.py'
```

Some tests require the model dependencies above. Submission verification additionally requires the original evaluation CSV and prepared row IDs.

## Publication scope

This import includes project source, experiment reports and lightweight execution records. The local instructions inspected did not establish redistribution permission for competition data or prediction CSVs carrying evaluation identifiers, so new CSVs and official instruction documents are excluded. Verification reports and generation code are included instead. Original local files are preserved.

Large models, prepared arrays, caches, databases, installed dependencies and temporary files are excluded. Files that were already tracked remotely, including the example submission, validation features and supplied starter, are preserved. Their presence does not establish redistribution rights. Official supplied material retains its original attribution; this publication adds no license grant for it.

The initial files were imported with one commit per file. The remaining files were published together in a batch at the project owner's request. All commits use their actual publication dates and record the publication process rather than the historical chronology of experiments.
