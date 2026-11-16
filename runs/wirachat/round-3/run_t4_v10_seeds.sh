#!/bin/bash
# Task 4 v10: more seeds of the two member types that gained most (equal-weight ensembles of strong members helped: V31a). Waits for DINOv2-Large (v9) to finish.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_dinov2l_fresh_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft9_dinov2l_fresh_log.txt; do sleep 60; done
$PY -W ignore t4_finetune2.py --arch timm:vit_base_patch14_dinov2.lvd142m --h 224 --w 448 --bs 16 --epochs 10 --lr 4e-5 --all --scales 1.0 --fresh-aug --seed 3 --tag dinov2b_fresh_s3 > t4_ft10_dinov2b_fresh_s3_log.txt 2>&1
$PY -W ignore t4_finetune2.py --arch timm:convnext_base.fb_in22k_ft_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --all --fresh-aug --seed 5 --tag cnxtb22k_fresh_s5 > t4_ft10_cnxtb22k_fresh_s5_log.txt 2>&1
