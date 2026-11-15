#!/bin/bash
# Task 4 v5: same recipe as v3 but with NEW random augmentations every epoch (--fresh-aug); v1-v4 reused one fixed augmentation per image.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY t4_finetune2.py --arch timm:convnext_base.fb_in22k_ft_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --all --fresh-aug --tag cnxtb22k_fresh > t4_ft5_cnxtb22k_fresh_log.txt 2>&1
$PY t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --all --scales 1.0 --fresh-aug --tag dinov2b_fresh > t4_ft5_dinov2b_fresh_log.txt 2>&1
