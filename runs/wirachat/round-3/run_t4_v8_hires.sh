#!/bin/bash
# Task 4 v8: higher-resolution fresh-augmentation ConvNeXt-B (IN-22k) at 288x576, 12 epochs (tests whether resolution closes the held-out 0.97 vs leaderboard ~0.90 gap)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:convnext_base.fb_in22k_ft_in1k --h 288 --w 576 --bs 10 --epochs 12 --lr 1e-4 --all --fresh-aug --seed 11 --tag cnxtb22k_hires > t4_ft8_cnxtb22k_hires_log.txt 2>&1
