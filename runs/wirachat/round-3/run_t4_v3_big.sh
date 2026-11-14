#!/bin/bash
# Round 3: larger pretrained backbones (timm), all training images, 224x448, strong augmentation, multi-scale TTA
cd /c/Users/Wirachat/Desktop/CPE342-KAGGLE
PY=.venv/Scripts/python
$PY -W ignore runs/wirachat/round-3/t4_finetune2.py --arch timm:convnext_base.fb_in22k_ft_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --all --tag cnxtb22k_all > runs/wirachat/round-3/t4_ft3_cnxtb22k_log.txt 2>&1
$PY -W ignore runs/wirachat/round-3/t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --all --scales 1.0 --tag dinov2b_all > runs/wirachat/round-3/t4_ft3_dinov2b_log.txt 2>&1
