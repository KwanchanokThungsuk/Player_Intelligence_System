#!/bin/bash
# Task 4 v26: second DINOv2-Large run with the new fine-tuning recipe (seed 202); same settings as run_t4_v25_large_newrecipe.sh. Local training only; no Kaggle access.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 6e-5 --llrd 0.9 --ema 0.999 --ema-cpu --ema-every 8 --mix 0.5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed 202 --tag dinov2l_new_s202 > t4_ft26_dinov2l_new_s202_log.txt 2>&1
