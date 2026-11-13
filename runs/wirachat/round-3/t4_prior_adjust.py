"""Round 3 - Task 4 prior (class-share) adjustment of the V18 ensemble. Uses only the model probabilities on the test images (no labels, no score feedback).
em   : Saerens EM estimate of the test class priors (training loss was class-balanced -> start from uniform), then reweight probabilities by pi/0.2.
half : logit bias that moves the predicted shares half-way to uniform.   Writes prob files in round-3 (OLD test.csv row order); build the CSVs with build_round3_t4_variant.py."""
from pathlib import Path
import numpy as np
R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"
P = np.mean([np.load(R2 / "t4_cnxt_test_prob.npy"), np.load(R2 / "t4_cnxt_all_test_prob.npy"), np.load(R2 / "t4_cnxts_all_test_prob.npy"), np.load(R3 / "t4_cnxt_hr_test_prob.npy")], 0)
base = P.argmax(1); share = lambda a: np.round(np.bincount(a, minlength=5) / len(a), 3)
pi = np.ones(5) / 5
for _ in range(200): Q = P * (pi / 0.2); Q /= Q.sum(1, keepdims=True); pi = Q.mean(0)
Pem = P * (pi / 0.2); Pem /= Pem.sum(1, keepdims=True)
t = (np.bincount(base, minlength=5) / len(P) + 0.2) / 2; b = np.zeros(5)
for _ in range(500): b += 0.1 * (np.log(t) - np.log(np.bincount((np.log(P + 1e-9) + b).argmax(1), minlength=5) / len(P)))
Ph = np.exp(np.log(P + 1e-9) + b); Ph /= Ph.sum(1, keepdims=True)
def em_alpha(a):
    Q = P * (pi / 0.2) ** a; return Q / Q.sum(1, keepdims=True)
for nm, Q in (("em", Pem), ("half", Ph), ("em2", em_alpha(2.0)), ("em3", em_alpha(3.0))):
    np.save(R3 / f"t4_prior_{nm}_test_prob.npy", Q); print(nm, "shares", share(Q.argmax(1)), "rows changed vs V18 ensemble", round(float((Q.argmax(1) != base).mean()), 4))
print("base shares", share(base), "| EM priors", np.round(pi, 3))
