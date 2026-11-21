#!/bin/bash
# Task 4 v25: DINOv2-Large with the new fine-tuning recipe that beat the baseline in the DINOv2-B method test (held-out plain macro-F1 0.9825 vs 0.9760, colour-stress 0.9744 vs 0.9686):
# layer-wise lr decay 0.9 (top lr 6e-5, i.e. lowest block 25x smaller), EMA of the weights (decay 0.999, copy on the CPU, updated every 8 steps), CutMix/MixUp on half of the batches.
# All training images (--all); everything else as the plain DINOv2-L seeds. Waits for the B test to finish so the GPU is not shared. Local training only; no Kaggle access.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_exp_b_sink_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft24_exp_b_sink_log.txt; do sleep 30; done
sleep 20
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 6e-5 --llrd 0.9 --ema 0.999 --ema-cpu --ema-every 8 --mix 0.5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed 201 --tag dinov2l_new_s201 > t4_ft25_dinov2l_new_s201_log.txt 2>&1
