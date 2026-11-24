# Task 5: account anomaly detection (unsupervised, plain F3)
**Method.** No labels exist, only `test.csv`. (1) Period-4 change group (825 rows): robust-z change score, clean gap between S 150 and 221. (2) Five persistent low-value groups, each separated by an empty gap (behavior_stability 246; ip_hash_entropy/login_time_variance/session_duration_std 207; communication_frequency 263; activity_level 265): 1,806 flags. (3) +49 rows with missing `behavior_stability` scored by a gradient-boosting classifier (flag if P > 0.09, because a flag helps iff precision > F3/10). (4) +2 tail rows (login-location jump and onset score far above the tail). Final: 1,857 flags. **Result:** F3 0.883 measured for the 1,806 flags (diagnostic upload); true anomalies estimated at about 2,070.

| Step | Script |
|---|---|
| EDA, group discovery (first data release) | [`round-1/task5_eda.py`](../../round-1/task5_eda.py), [`round-1/task5_groups.py`](../../round-1/task5_groups.py), [`round-2/task5_groups2.py`](../../round-2/task5_groups2.py) |
| gap scans and searches for more groups | [`round-2/task5_gapscan.py`](../../round-2/task5_gapscan.py), [`task5_gapscan2.py`](../../round-2/task5_gapscan2.py), [`task5_gapscan3.py`](../../round-2/task5_gapscan3.py), [`round-2/task5_remaining.py`](../../round-2/task5_remaining.py), [`round-3/t5_gmm.py`](../../round-3/t5_gmm.py) |
| groups on the corrected data (1,806 flags) | [`round-3/t5_new.py`](../../round-3/t5_new.py), [`round-3/t5_new2.py`](../../round-3/t5_new2.py) |
| missing-value classifier (+49) | [`round-3/t5_nanstab.py`](../../round-3/t5_nanstab.py), [`round-3/build_round3_t5_nanstab.py`](../../round-3/build_round3_t5_nanstab.py) |
| tail rows (+2) | [`round-3/build_round3_t3t5_update.py`](../../round-3/build_round3_t3t5_update.py) |
| ablation used to infer the number of anomalies | [`round-3/build_round3_ablate_t5.py`](../../round-3/build_round3_ablate_t5.py) |

Details: report section 6.
