# CPE342 Karena Player Intelligence System: source code (branch `runs/wirachat`)

Final public Kaggle score **0.84167** (file V57a); selected for the final ranking: V57a and V53c (0.84113). Technical report: [`../report/report.md`](../report/report.md).

## Layout
One folder per task, each with a README (method, results, the exact scripts and commands):

| Folder | Task | Final method | Validation |
|---|---|---|---|
| [`Task1/`](Task1/README.md) | anti-cheat pre-filter (binary, F2) | LightGBM raw + engineered, threshold 0.18 | OOF F2 0.837 |
| [`Task2/`](Task2/README.md) | player segment (4 classes, macro F1) | LightGBM multiclass, tuned class weights | OOF macro F1 0.7524 |
| [`Task3/`](Task3/README.md) | monthly spending (NMAE) | two-stage lognormal median + L1 LightGBM blend | OOF 0.7907 |
| [`Task4/`](Task4/README.md) | game title from screenshots (macro F1) | fine-tuned DINOv2-L/B + ConvNeXt ensemble, EM class prior | held-out clusters 0.98; leaderboard |
| [`Task5/`](Task5/README.md) | account anomaly detection (F3, unsupervised) | gap-based groups + missing-value classifier + tail rows | leaderboard (no labels) |

## Why the scripts are linked, not copied into these folders
The scripts were written round by round (`../round-1`, `../round-2`, `../round-3`), they share helpers (`round-1/r1_common.py`) and intermediate arrays, and the team workflow keeps each round's code in its own folder. A copy here would drift from the version that produced the scores, so every task README links the exact files instead.

## How to run
- Python 3.11, packages in the repository's `requirements.txt` plus `lightgbm`, `torch`, `timm`, `huggingface_hub`.
- Data: the host's corrected release (2026-11-12) in a folder given by the environment variable `CPE342_DATA` (read by `round-1/r1_common.py`). The dataset tracked in the repository is the older release.
- Run scripts from their own folder (they find their neighbours relative to `__file__`).
- Final build of the best file: [`../round-3/build_final_v57.sh`](../round-3/build_final_v57.sh).
- Not committed: submission CSVs, prediction arrays (`.npy`), model weights, data, API credentials.
- Not re-run end to end after the final cleanup.

## Round notes (every experiment with its Kaggle score)
[`../round-1/notes.md`](../round-1/notes.md), [`../round-2/notes.md`](../round-2/notes.md), [`../round-3/notes.md`](../round-3/notes.md).
