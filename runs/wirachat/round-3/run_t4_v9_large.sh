#!/bin/bash
# Task 4 v9: DINOv2-Large (304M) fresh-augmentation fine-tune with gradient checkpointing. Waits for the v8 hi-res run to release the GPU first.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
until [ -f t4_cnxtb22k_hires_test_prob.npy ] || grep -q -E "Traceback|out of memory" t4_ft8_cnxtb22k_hires_log.txt; do sleep 60; done
$PY -W ignore t4_finetune2.py --arch timm:vit_large_patch14_dinov2.lvd142m --h 224 --w 448 --bs 8 --epochs 8 --lr 2e-5 --all --scales 1.0 --fresh-aug --gc --seed 21 --tag dinov2l_fresh > t4_ft9_dinov2l_fresh_log.txt 2>&1
