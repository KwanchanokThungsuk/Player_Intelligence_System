#!/bin/bash
# Task 4 v6: the small ConvNeXts retrained with --fresh-aug (new random augmentations every epoch)
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
$PY -W ignore t4_finetune2.py --arch convnext_small --h 224 --w 448 --bs 24 --epochs 12 --all --fresh-aug --tag cnxts_fresh > t4_ft6_cnxts_fresh_log.txt 2>&1
$PY -W ignore t4_finetune2.py --arch convnext_tiny --h 224 --w 448 --bs 32 --epochs 12 --all --fresh-aug --tag cnxt_fresh > t4_ft6_cnxt_fresh_log.txt 2>&1
