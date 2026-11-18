#!/bin/bash
# Task 4 v16: fifth plain DINOv2-Large seed (seed 88). Re-run with --workers 2: the first attempt crashed with Windows error 1455 (commit memory exhausted:
# a 18 GB WSL VM was holding RAM, leaving about 1.5 GB free) while using 10 data-loading workers.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --all --scales 1.0 --fresh-aug --gc --workers 2 --seed 88 --tag dinov2l_fresh_s88 > t4_ft16_dinov2l_fresh_s88_log.txt 2>&1
