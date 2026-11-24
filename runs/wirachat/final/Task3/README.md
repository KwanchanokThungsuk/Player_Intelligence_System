# Task 3: monthly spending (regression, NMAE)
**Method.** Blend `0.55 x two-stage(identity features) + 0.25 x two-stage(raw) + 0.20 x L1 LightGBM(identity)`, clipped to [0, 300000]. The two-stage model predicts the conditional median: LightGBM P(spend = 0), LightGBM log-spend for positive rows, a per-segment lognormal sigma. Identity features: `historical_spending = total_transactions x avg_transaction_value` and a recovered `segment`. **Result:** OOF NMAE score 0.7907.

| Step | Script |
|---|---|
| L1 LightGBM baselines and tuning | [`round-1/task3_cv.py`](../../round-1/task3_cv.py), [`round-1/task3_tune.py`](../../round-1/task3_tune.py) |
| exact identities and diagnostics | [`round-2/t123_identities.py`](../../round-2/t123_identities.py), [`round-1/task3_diag.py`](../../round-1/task3_diag.py), [`round-2/t3_resid.py`](../../round-2/t3_resid.py) |
| two-stage model (raw features) | [`round-2/t3_twostage.py`](../../round-2/t3_twostage.py) |
| L1 model with identity features | [`round-2/t3_fe2.py`](../../round-2/t3_fe2.py) |
| two-stage model with identity features | [`round-3/t3_twostage_fe.py`](../../round-3/t3_twostage_fe.py) |
| final blend into the submission | [`round-3/build_round3_t3t5_update.py`](../../round-3/build_round3_t3t5_update.py) |
| neural-net comparison (worse) | [`round-3/t3_nn.py`](../../round-3/t3_nn.py) |

Details: report section 4.
