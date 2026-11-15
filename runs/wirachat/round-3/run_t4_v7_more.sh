#!/bin/bash
# Task 4 v7: more fresh-augmentation members: CLIP-pretrained ConvNeXt-B, then a second DINOv2-B (different seed)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch timm:convnext_base.clip_laion2b_augreg_ft_in12k_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --all --fresh-aug --tag cnxtbclip_fresh > t4_ft7_cnxtbclip_fresh_log.txt 2>&1
$PY -W ignore t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --all --scales 1.0 --fresh-aug --seed 7 --tag dinov2b_fresh_s7 > t4_ft7_dinov2b_fresh_s7_log.txt 2>&1
