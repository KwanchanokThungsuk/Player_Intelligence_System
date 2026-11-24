# Task 4: game title detection (5 classes, macro F1)
**Method.** End-to-end fine-tuning of pretrained backbones at 224x448 with strong augmentation (fresh per epoch), 3-scale + flip test-time augmentation, probability-average ensemble, then a class-prior (Saerens EM) reweighting. Final members: 6 DINOv2-L with the new recipe (layer-wise lr decay 0.9, EMA 0.999, CutMix/MixUp), 1 DINOv2-B new recipe, 10 DINOv2-L baseline seeds, 6 ConvNeXt-B / DINOv2-B seeds, ConvNeXt-S/T. The EM step estimates the test class shares from the predicted probabilities, without labels (report, section 1.2).

| Step | Script |
|---|---|
| training and prediction (all backbones, `--fresh-aug`, `--llrd --ema --mix`) | [`round-3/t4_finetune2.py`](../../round-3/t4_finetune2.py) |
| exact command of every run | `round-3/run_t4_v*.sh`, e.g. [`run_t4_v25_large_newrecipe.sh`](../../round-3/run_t4_v25_large_newrecipe.sh), [`run_t4_v29_newrecipe_more2.sh`](../../round-3/run_t4_v29_newrecipe_more2.sh) |
| re-predict from saved weights | [`round-3/t4_predict_saved.py`](../../round-3/t4_predict_saved.py) |
| class-prior (EM) step | [`round-3/t4_prior_adjust.py`](../../round-3/t4_prior_adjust.py) |
| ensemble builder (maps predictions by image id into the key) | [`round-3/build_round3_t4_variant.py`](../../round-3/build_round3_t4_variant.py) |
| format and sanity check of a submission | [`round-3/check_submission.py`](../../round-3/check_submission.py) |
| diagnostics (style shift, adversarial validation, CLIP) | [`round-3/t4_diag_style.py`](../../round-3/t4_diag_style.py), [`round-3/adv_valid.py`](../../round-3/adv_valid.py), [`round-3/t4_clip.py`](../../round-3/t4_clip.py) |
| early work (embeddings, first fine-tune, held-out clusters) | [`round-2/t4_embed.py`](../../round-2/t4_embed.py), [`round-2/t4_finetune.py`](../../round-2/t4_finetune.py), [`round-2/t4_cv.py`](../../round-2/t4_cv.py) |
| final build commands | [`round-3/build_final_v57.sh`](../../round-3/build_final_v57.sh) |

Details: report section 5. Weights and prediction arrays are not committed.
