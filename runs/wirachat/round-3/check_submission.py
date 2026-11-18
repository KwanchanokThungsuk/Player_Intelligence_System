"""Sanity-check a built submission against a reference submission before it is uploaded. Exit code 0 = OK, 1 = do not upload.
Usage: python check_submission.py <candidate.csv> <reference.csv> [min_task4_diff max_task4_diff]"""
import sys
import pandas as pd

cand, ref = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (20, 1500)
problems = []
if list(cand.columns) != list(ref.columns): problems.append(f"columns differ: {list(cand.columns)} vs {list(ref.columns)}")
if len(cand) != len(ref): problems.append(f"row count {len(cand)} vs {len(ref)}")
if not problems:
    if not (cand.iloc[:, 0] == ref.iloc[:, 0]).all(): problems.append("id column differs")
    if cand.isna().any().any(): problems.append("NaN values present")
    others = [c for c in cand.columns[1:] if c != "task4"]
    for c in others:
        if not (cand[c] == ref[c]).all(): problems.append(f"{c} differs from the reference (should be identical)")
    if not set(cand["task4"].unique()) <= {0, 1, 2, 3, 4}: problems.append(f"task4 values {sorted(cand['task4'].unique())}")
    d = int((cand["task4"] != ref["task4"]).sum())
    if not lo <= d <= hi: problems.append(f"task4 rows differing from reference = {d}, outside [{lo}, {hi}]")
    print(f"rows {len(cand)} | task4 rows differing from reference: {d} | task4 class shares {cand['task4'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
print("PROBLEMS: " + "; ".join(problems) if problems else "CHECK OK")
sys.exit(1 if problems else 0)
