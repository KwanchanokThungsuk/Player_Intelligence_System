#!/bin/bash
# Task 4 v19: eighth plain DINOv2-Large seed (seed 122), started right after seed 99 finished so the GPU never idles. Local training only (no Kaggle access).
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed 122 --tag dinov2l_fresh_s122 > t4_ft19_dinov2l_fresh_s122_log.txt 2>&1
