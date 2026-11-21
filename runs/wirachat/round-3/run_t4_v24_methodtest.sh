#!/bin/bash
# Task 4 v24: does a better DINOv2 fine-tuning recipe help? Two DINOv2-B runs on the SAME split (train on ~25.5k images, score the 6,000 held-out-cluster images,
# plain and colour-stress macro-F1), 10 epochs each: (1) baseline = the recipe of the DINOv2-B members (uniform lr 4e-5); (2) layer-wise lr decay 0.8 (top lr 1e-4)
# + EMA of the weights (0.999) + CutMix/MixUp on half of the batches. Waits for DINOv2-L seed 166 so the GPU is never shared. Local training only; no Kaggle access.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_dinov2l_fresh_s166_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft23_dinov2l_fresh_s166_log.txt; do sleep 60; done
sleep 30
$PY -W ignore t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --scales 1.0 --fresh-aug --workers 4 --seed 1 --tag exp_b_base > t4_ft24_exp_b_base_log.txt 2>&1
$PY -W ignore t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --llrd 0.8 --ema 0.999 --mix 0.5 --scales 1.0 --fresh-aug --workers 4 --seed 1 --tag exp_b_sink > t4_ft24_exp_b_sink_log.txt 2>&1
