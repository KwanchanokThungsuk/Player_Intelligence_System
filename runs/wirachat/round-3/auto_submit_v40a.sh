#!/bin/bash
# Waits for the fifth plain DINOv2-Large seed, builds V40a (five plain L seeds x6 each + the V36b recipe), checks it, and makes ONE Kaggle submission
# (explicitly authorised by the user, one file only, no retries). Output goes to auto_submit_v40a_log.txt (no credentials are printed).
cd "$(dirname "$0")"
exec > auto_submit_v40a_log.txt 2>&1
export CPE342_DATA="C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main"
export KAGGLE_CONFIG_DIR="C:/Users/Wirachat/Desktop/CPE342-KAGGLE"
PY=../../../.venv/Scripts/python.exe
KG=../../../.venv/Scripts/kaggle.exe
COMP=player-intelligence-system-cpe-342-ml
echo "waiting for seed 5 ... $(date)"
until [ -f t4_dinov2l_fresh_s88_test_prob.npy ] || grep -q -E "Traceback|out of memory|Killed" t4_ft16_dinov2l_fresh_s88_log.txt; do sleep 60; done
if [ ! -f t4_dinov2l_fresh_s88_test_prob.npy ]; then echo "seed 5 FAILED - nothing built, nothing submitted"; tail -n 3 t4_ft16_dinov2l_fresh_s88_log.txt; exit 1; fi
sleep 10; echo "seed 5 done $(date)"
F=t4_cnxtb22k_fresh_test_prob.npy; F5=t4_cnxtb22k_fresh_s5_test_prob.npy; H=t4_cnxtb22k_hires_test_prob.npy; DF=t4_dinov2b_fresh_test_prob.npy; D7=t4_dinov2b_fresh_s7_test_prob.npy; D3=t4_dinov2b_fresh_s3_test_prob.npy; S=t4_cnxts_fresh_test_prob.npy; T=t4_cnxt_fresh_test_prob.npy
L=t4_dinov2l_fresh_test_prob.npy; L2=t4_dinov2l_fresh_s33_test_prob.npy; L3=t4_dinov2l_fresh_s44_test_prob.npy; L4=t4_dinov2l_fresh_s55_test_prob.npy; L5=t4_dinov2l_fresh_s88_test_prob.npy
OTH="$F $F $F5 $F5 $H $H $DF $DF $D7 $D7 $D3 $D3 $S $T"
LS6="$L $L $L $L $L $L $L2 $L2 $L2 $L2 $L2 $L2 $L3 $L3 $L3 $L3 $L3 $L3 $L4 $L4 $L4 $L4 $L4 $L4 $L5 $L5 $L5 $L5 $L5 $L5"
$PY t4_prior_adjust.py _40a $OTH $LS6 | tail -n 1 || { echo "prior adjust failed"; exit 1; }
$PY build_round3_t4_variant.py v40a_t4only t4_prior_40a_em_test_prob.npy | tail -n 1 || { echo "build failed"; exit 1; }
$PY build_round3_t5_nanstab.py v40a_t4only v40a_L5x6_em 0.09 | tail -n 1 || { echo "T5 build failed"; exit 1; }
rm -f submission_v40a_t4only.csv
if ! $PY check_submission.py submission_v40a_L5x6_em.csv submission_v36b_L4x6_em.csv 20 1500; then echo "sanity check failed - NOT submitting"; exit 1; fi
echo "submitting submission_v40a_L5x6_em.csv at $(date)"
$KG competitions submit $COMP -f "$(pwd)/submission_v40a_L5x6_em.csv" -m "v40a: five plain DINOv2-L seeds x6 + ConvNeXt/DINOv2-B members, EM prior, T5 flags" --wait 1200
echo "submit exit code $? at $(date)"
$KG competitions submissions $COMP --page-size 3
echo "done"
