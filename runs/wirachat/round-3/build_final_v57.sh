#!/bin/bash
# Final candidates V57a (EM prior) and V57b (same weights, no EM = clean hedge). Run from this folder after all models are trained.
# Order: Task 3/5 update (V14b) -> Task 4 ensemble -> Task 5 NaN-stability flags.  Weights: others x2 (28) + ten plain DINOv2-L seeds x7 (70) + six new-recipe DINOv2-L (seeds 201-206) x33 (198) + all-image DINOv2-B new recipe x20 = 69% new-recipe weight.
cd "$(dirname "$0")"
export CPE342_DATA="C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main"; PY=../../../.venv/Scripts/python.exe
$PY t3_twostage_fe.py                      # Task 3 two-stage with identity features (OOF 0.78995) -> data2/t3_twostage_fe_*.npy
$PY build_round3_t3t5_update.py            # V14b = V14 + Task 3 blend (OOF 0.79071) + two Task 5 flags (rows 8038, 14925)
export T4_BASE=submission_v14b_t3fe_t5tail.csv
F=t4_cnxtb22k_fresh_test_prob.npy; F5=t4_cnxtb22k_fresh_s5_test_prob.npy; H=t4_cnxtb22k_hires_test_prob.npy; DF=t4_dinov2b_fresh_test_prob.npy; D7=t4_dinov2b_fresh_s7_test_prob.npy; D3=t4_dinov2b_fresh_s3_test_prob.npy; S=t4_cnxts_fresh_test_prob.npy; T=t4_cnxt_fresh_test_prob.npy
OTH="$F $F $F $F $F5 $F5 $F5 $F5 $H $H $H $H $DF $DF $DF $DF $D7 $D7 $D7 $D7 $D3 $D3 $D3 $D3 $S $S $T $T"
PL=""; for s in "" _s33 _s44 _s55 _s88 _s99 _s111 _s122 _s133 _s144; do f=t4_dinov2l_fresh${s}_test_prob.npy; PL="$PL $f $f $f $f $f $f $f"; done
NL=""; for i in $(seq 1 33); do for s in 201 202 203 204 205 206; do NL="$NL t4_dinov2l_new_s${s}_test_prob.npy"; done; done
BL=""; for i in $(seq 1 20); do BL="$BL t4_dinov2b_new_s301_test_prob.npy"; done
# V57a: EM class-prior step (test-derived, disclose in the report)
$PY t4_prior_adjust.py _v57a $OTH $PL $NL $BL | tail -n 1
$PY build_round3_t4_variant.py v57a_t4only t4_prior_v57a_em_test_prob.npy | tail -n 1
$PY build_round3_t5_nanstab.py v57a_t4only v57a_new6x33_ball20_t3fe_em 0.09; rm -f submission_v57a_t4only.csv
# V57b: no EM (same members and weights via file:weight)
M="$F:4 $F5:4 $H:4 $DF:4 $D7:4 $D3:4 $S:2 $T:2"
for s in "" _s33 _s44 _s55 _s88 _s99 _s111 _s122 _s133 _s144; do M="$M t4_dinov2l_fresh${s}_test_prob.npy:7"; done
for s in 201 202 203 204 205 206; do M="$M t4_dinov2l_new_s${s}_test_prob.npy:33"; done; M="$M t4_dinov2b_new_s301_test_prob.npy:20"
$PY build_round3_t4_variant.py v57b_t4only $M | tail -n 1
$PY build_round3_t5_nanstab.py v57b_t4only v57b_new6x33_ball20_t3fe_clean 0.09; rm -f submission_v57b_t4only.csv
