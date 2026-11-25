# Task 4: game title detection (5 classes, macro F1)

Final method: fine-tune pretrained backbones (DINOv2-L, DINOv2-B, ConvNeXt-B/S/T; pretrained models are allowed for this task) on all training images, average their test probabilities with integer weights, then apply a class-prior (Saerens EM) correction estimated from the ensemble probabilities. Kaggle public score of the final file V57a: 0.84167.

| File | Purpose |
|---|---|
| `train_backbone.py` | trains ONE member and writes its test probabilities (224x448, strong augmentation drawn fresh every epoch, class-balanced loss, flip test-time augmentation at three scales by default (`--scales 1.0` for the DINOv2 members); optional layer-wise lr decay, EMA, CutMix/MixUp) |
| `train_all_members.sh` | the exact command of each of the 25 members (about 130 GPU hours in total on a laptop RTX 4060) |
| `task4_ensemble.ipynb` | weighted average of the members, class-prior correction, mapping to the ids of `test.csv`; writes `../outputs/task4_pred_*.npy` |

## Data layout expected
`CPE342_DATA` = Kaggle download folder. `train_backbone.py` reads `04_image_classification/train.csv` and `test.csv` (columns `id, file_name, label`) and the images `04_image_classification/train/<file_name>` and `test/<file_name>`. If the csv files and the image folders are in different places, set `CPE342_T4_CSV_DIR` and `CPE342_T4_IMAGES`.

## Run
```bash
export CPE342_DATA=/path/to/kaggle/download
python train_backbone.py --arch convnext_tiny --bs 16 --fresh-aug --tag smoke --smoke --scales 1.0   # 1-minute check (640 images, 1 epoch)
bash train_all_members.sh                                                                          # all members (days); or run single lines
CPE342_T4_PROB_ORDER=ans CPE342_T4_PROBS=../outputs/task4_probs jupyter nbconvert --execute --inplace task4_ensemble.ipynb
```
Retraining all members is slow, so the saved member probabilities (`t4_<tag>_test_prob.npy`, 25 files of 25,889 x 5 numbers) must be supplied separately with the submission (for example in the LEB2 archive) and are **not** committed to git (size and repository rules). With those files in `../outputs/task4_probs/`:
- our saved files follow the row order of the *original* `test.csv` (image ids), which is the default `CPE342_T4_PROB_ORDER=v1`; the notebook reads that file from the repository (`cpe342-kaggle-main/04_image_classification/test.csv`, or set `CPE342_T4_V1_TEST_CSV`);
- files written by `train_backbone.py` follow the corrected `test.csv`, use `CPE342_T4_PROB_ORDER=ans`.

## Class-prior correction (EM)
After averaging, the test class shares are estimated from the ensemble's own probabilities (Saerens EM, no labels) and the probabilities are multiplied by the estimated shares. It is worth +0.0017 to +0.0019 on the public leaderboard (V57a 0.84167 against 0.83980 without it, same members). `task4_ensemble.ipynb` also writes the version without it (`task4_pred_v57a_noEM.npy`).

## Notes
- The second selected file V53c uses one extra member, `exp_b_sink` (a DINOv2-B new-recipe model trained on 80% of the images with a held-out validation split). It was trained with an earlier version of the training script (validation clusters, not part of this final script), so the notebook uses it only if the file exists.
- Task 4 results depend on GPU kernels and library versions; retraining gives probabilities close to, but not bit-identical with, the saved ones.
- Early experiments (embeddings, held-out-cluster validation, CLIP, resolution and weight sweeps) are in `../../round-2` and `../../round-3`; they are not needed to reproduce the submission.
