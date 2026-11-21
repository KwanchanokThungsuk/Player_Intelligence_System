#!/bin/bash
# Task 4 v27: more DINOv2-Large runs with the new fine-tuning recipe (V50b, new model at weight 49, scored 0.83960 vs 0.83820 for V45a), seeds 203 and 204, one after the other.
# Waits for the seed 202 run so the GPU is never shared. Same settings as run_t4_v25_large_newrecipe.sh. Local training only; no Kaggle access.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_dinov2l_new_s202_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft26_dinov2l_new_s202_log.txt; do sleep 60; done
sleep 30
for S in 203 204; do
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 6e-5 --llrd 0.9 --ema 0.999 --ema-cpu --ema-every 8 --mix 0.5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed $S --tag dinov2l_new_s$S > t4_ft27_dinov2l_new_s${S}_log.txt 2>&1
done
