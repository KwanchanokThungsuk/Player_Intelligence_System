#!/bin/bash
# Round 3: two sequential Task 4 runs (high resolution, stronger augmentation, multi-scale TTA)
cd /c/Users/Wirachat/Desktop/CPE342-KAGGLE
PY=.venv/Scripts/python
$PY -W ignore runs/wirachat/round-3/t4_finetune2.py --arch convnext_tiny --h 224 --w 448 --bs 32 --epochs 12 --tag cnxt_hr > runs/wirachat/round-3/t4_ft2_cnxt_hr_log.txt 2>&1
$PY -W ignore runs/wirachat/round-3/t4_finetune2.py --arch efficientnet_v2_s --h 224 --w 448 --bs 32 --epochs 12 --lr 5e-4 --all --tag effv2s_hr_all > runs/wirachat/round-3/t4_ft2_effv2s_log.txt 2>&1
