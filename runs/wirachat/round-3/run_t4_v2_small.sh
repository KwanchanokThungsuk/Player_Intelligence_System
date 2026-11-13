#!/bin/bash
cd /c/Users/Wirachat/Desktop/CPE342-KAGGLE
.venv/Scripts/python -W ignore runs/wirachat/round-3/t4_finetune2.py --arch convnext_small --h 224 --w 448 --bs 24 --epochs 12 --all --tag cnxts_hr_all > runs/wirachat/round-3/t4_ft2_cnxts_hr_all_log.txt 2>&1
