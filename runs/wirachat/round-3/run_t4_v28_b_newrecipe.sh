#!/bin/bash
# Task 4 v28: DINOv2-Base with the new fine-tuning recipe on all training images (about 2 h per run): a cheaper, different-size member for the new-recipe ensemble.
# Same recipe as the DINOv2-B method test (layer-wise lr decay 0.8, top lr 1e-4, EMA 0.999, CutMix/MixUp on half of the batches). Local training only; no Kaggle access.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --llrd 0.8 --ema 0.999 --mix 0.5 --all --scales 1.0 --fresh-aug --workers 4 --seed 301 --tag dinov2b_new_s301 > t4_ft28_dinov2b_new_s301_log.txt 2>&1
