#!/bin/bash
# Exact training commands of the 25 members of the final Task 4 ensemble (file V57a). Run from this folder, one after another (one GPU).
# Environment (see README.md): CPE342_DATA = Kaggle download folder, CPE342_T4_IMAGES = folder that holds train/ and test/ with the jpg files (default: $CPE342_DATA/04_image_classification).
# Each command writes ../outputs/task4_probs/t4_<tag>_test_prob.npy (rows follow the corrected test.csv, so use CPE342_T4_PROB_ORDER=ans in task4_ensemble.ipynb).
# Time on an RTX 4060 Laptop GPU (8 GB): about 0.5-2 h for the ConvNeXt / DINOv2-B members, about 6.5 h per baseline DINOv2-L, about 8 h per new-recipe DINOv2-L (roughly 130 GPU hours in total).
# Quick check of the whole pipeline in about a minute: python train_backbone.py --arch convnext_tiny --bs 16 --fresh-aug --tag smoke --smoke --scales 1.0
set -e
PY=python
T="$PY -W ignore train_backbone.py"

# ---- 8 "other" members (weights 4 and 2 in the ensemble) ------------------------------------------------------------------------------------------
$T --arch timm:convnext_base.fb_in22k_ft_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --fresh-aug --tag cnxtb22k_fresh
$T --arch timm:convnext_base.fb_in22k_ft_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --fresh-aug --seed 5 --tag cnxtb22k_fresh_s5
$T --arch timm:convnext_base.fb_in22k_ft_in1k --h 288 --w 576 --bs 10 --epochs 12 --lr 1e-4 --fresh-aug --seed 11 --tag cnxtb22k_hires
$T --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --scales 1.0 --fresh-aug --tag dinov2b_fresh
$T --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --scales 1.0 --fresh-aug --seed 7 --tag dinov2b_fresh_s7
$T --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --scales 1.0 --fresh-aug --seed 3 --tag dinov2b_fresh_s3
$T --arch convnext_small --h 224 --w 448 --bs 24 --epochs 12 --fresh-aug --tag cnxts_fresh
$T --arch convnext_tiny --h 224 --w 448 --bs 32 --epochs 12 --fresh-aug --tag cnxt_fresh

# ---- 10 baseline DINOv2-L members (weight 7 each): lr 2e-5, no layer decay / EMA / mixing -----------------------------------------------------------
for S in 21 33 44 55 88 99 111 122 133 144; do
  TAG=dinov2l_fresh_s$S; [ "$S" = "21" ] && TAG=dinov2l_fresh          # the first seed (21) has no suffix
  $T --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --scales 1.0 --fresh-aug --gc --workers 2 --seed $S --tag $TAG
done

# ---- 6 new-recipe DINOv2-L members (weight 33 each): layer-wise lr decay 0.9, EMA 0.999 (on the CPU), CutMix/MixUp on half of the batches ---------------
for S in 201 202 203 204 205 206; do
  $T --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 6e-5 --llrd 0.9 --ema 0.999 --ema-cpu --ema-every 8 --mix 0.5 --scales 1.0 --fresh-aug --gc --workers 2 --seed $S --tag dinov2l_new_s$S
done

# ---- 1 new-recipe DINOv2-B member (weight 20) ----------------------------------------------------------------------------------------------------------
$T --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --llrd 0.8 --ema 0.999 --mix 0.5 --scales 1.0 --fresh-aug --workers 4 --seed 301 --tag dinov2b_new_s301
