#!/bin/bash
# Third large backbone for Task 4: ConvNeXt-Base with CLIP-LAION2B pretraining (different pretraining => diverse member)
cd "$(dirname "$0")"
../../../.venv/Scripts/python.exe t4_finetune2.py --arch timm:convnext_base.clip_laion2b_augreg_ft_in12k_in1k --h 224 --w 448 --bs 16 --epochs 10 --lr 1e-4 --all --tag cnxtbclip_all > t4_ft4_cnxtbclip_log.txt 2>&1
