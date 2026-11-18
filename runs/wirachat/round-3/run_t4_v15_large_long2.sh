#!/bin/bash
# Task 4 v15: second seed of the longer DINOv2-Large variant (12 epochs, lr 3e-5; first one reached the lowest training loss 0.5419)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 12 --lr 3e-5 --all --scales 1.0 --fresh-aug --gc --seed 77 --tag dinov2l_long_s77 > t4_ft15_dinov2l_long_s77_log.txt 2>&1
