#!/bin/bash
# Task 4 v12: third DINOv2-Large seed (DINOv2-L gave the largest single gain: V32a 0.83558)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --all --scales 1.0 --fresh-aug --gc --seed 44 --tag dinov2l_fresh_s44 > t4_ft12_dinov2l_fresh_s44_log.txt 2>&1
