#!/bin/bash
# Task 4 v17: sixth plain DINOv2-Large seed; waits for the fifth (v16) to finish so the GPU never idles. Trains locally only (no Kaggle access). --workers 2 (see v16).
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_dinov2l_fresh_s88_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft16_dinov2l_fresh_s88_log.txt; do sleep 60; done
sleep 30
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed 99 --tag dinov2l_fresh_s99 > t4_ft17_dinov2l_fresh_s99_log.txt 2>&1
