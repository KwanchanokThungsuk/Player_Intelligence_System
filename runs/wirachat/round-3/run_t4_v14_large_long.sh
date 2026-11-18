#!/bin/bash
# Task 4 v14: DINOv2-Large variant, longer schedule (12 epochs) and higher lr (3e-5): tests whether more training beats more seeds (seed 66)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 12 --lr 3e-5 --all --scales 1.0 --fresh-aug --gc --seed 66 --tag dinov2l_long_s66 > t4_ft14_dinov2l_long_s66_log.txt 2>&1
