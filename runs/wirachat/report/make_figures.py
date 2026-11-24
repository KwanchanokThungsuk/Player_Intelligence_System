"""Figures of the technical report. Data: results written by the final notebooks (../final/outputs) and the Kaggle public scores recorded in ../round-3/notes.md.
    CPE342_DATA=<kaggle download folder> python make_figures.py        -> figures/*.png"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "final" / "outputs"; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = Path(os.environ.get("CPE342_DATA", ""))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9990"      # categorical slots 1 and 2 of the reference palette, plus a neutral
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False})

def save(name): plt.savefig(FIG / name); plt.close(); print("wrote", name)

# ---------------------------------------------------------------- Figure 1: score history
hist = [("baseline", 0.50879), ("round 1", 0.54617), ("V1", 0.56346), ("V3", 0.65561), ("V10", 0.69578), ("V14", 0.81542), ("V18", 0.81835), ("V20", 0.81973), ("V22c", 0.82473),
        ("V24a", 0.82772), ("V27c", 0.83310), ("V32a", 0.83558), ("V36a", 0.83735), ("V45a", 0.83820), ("V50b", 0.83960), ("V53c", 0.84113), ("V57a", 0.84167)]
notes = {"baseline": "provided baseline", "V1": "Task 1 threshold", "V3": "Task 5 groups", "V14": "host fixes Task 4 key", "V24a": "augmentation bug fixed",
         "V32a": "DINOv2-Large", "V50b": "new fine-tuning recipe", "V57a": "final"}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(len(hist)); s = np.array([h[1] for h in hist])
ax[0].plot(x, s, color=BLUE, lw=1.6, marker="o", ms=3.5)
for i, (n, v) in enumerate(hist):
    if n in ("baseline", "V3", "V14", "V57a"): ax[0].annotate(f"{n}\n{v:.3f}", (i, v), textcoords="offset points", xytext=(0, -24 if n != "V57a" else 6), ha="center" if n != "V57a" else "right", fontsize=7.5, color=INK2)
ax[0].set_ylim(0.45, 0.94); ax[0].set_xticks([]); ax[0].set_ylabel("Kaggle public score"); ax[0].set_title("(a) all submissions that changed the best score", loc="left", fontsize=8.5, color=INK2)
z = slice(5, None)
ax[1].plot(x[z], s[z], color=BLUE, lw=1.6, marker="o", ms=3.5)
for i, (n, v) in enumerate(hist):
    if i >= 5 and n in notes: ax[1].annotate(f"{n} {v:.4f}\n{notes[n]}", (i, v), textcoords="offset points", xytext=(6, -22) if n != "V57a" else (-4, 8), ha="left" if n != "V57a" else "right", fontsize=7, color=INK2)
ax[1].set_xticks(x[z]); ax[1].set_xticklabels([h[0] for h in hist][5:], rotation=60, fontsize=7); ax[1].set_ylim(0.81, 0.846); ax[1].set_title("(b) after the dataset correction", loc="left", fontsize=8.5, color=INK2)
plt.tight_layout(); save("fig_score_history.png")

# ---------------------------------------------------------------- Figure 2: Task 1 F2 versus threshold
c = np.load(OUT / "task1_f2_curve.npy")
fig, ax = plt.subplots(figsize=(4.6, 2.6)); ax.plot(c[:, 0], c[:, 1], color=BLUE, lw=1.6)
ax.axvline(0.18, color=ORANGE, lw=1, ls="--"); ax.text(0.195, 0.47, "threshold 0.18" + chr(10) + "OOF F2 0.837", fontsize=8, color=INK2, va="center")
ax.axvline(0.40, color=GRAY, lw=1, ls=":"); ax.annotate("0.40 (round 1,\nmacro-F2 threshold)", (0.40, np.interp(0.40, c[:, 0], c[:, 1])), textcoords="offset points", xytext=(6, 8), fontsize=8, color=INK2)
ax.set_xlabel("decision threshold on the OOF probability"); ax.set_ylabel("OOF F2 (cheater class)"); ax.set_xlim(0, 0.9); plt.tight_layout(); save("fig_task1_threshold.png")

# ---------------------------------------------------------------- Figure 3: Task 3 target and where the error sits
tr = pd.read_csv(DATA / "03_regression" / "train.csv"); y = tr.spending_30d.values
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
pos = y[y > 0]; ax[0].hist(np.log10(pos), bins=60, color=BLUE, alpha=0.9)
ax[0].set_xlabel("spending_30d, THB (positive values, log10 scale)"); ax[0].set_ylabel("players"); ax[0].set_xticks([0, 1, 2, 3, 4, 5]); ax[0].set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
ax[0].annotate(f"{np.mean(y == 0):.0%} of players spend exactly 0\n(not shown)", (0.02, 0.9), xycoords="axes fraction", fontsize=8, color=INK2, va="top")
ax[0].set_title("(a) the target is zero-inflated and heavy-tailed", loc="left", fontsize=8.5, color=INK2)
share = {"Casual": 0.010, "Grinder": 0.058, "Social": 0.075, "Whale": 0.796, "segment\nmissing": 0.060}       # from task3.ipynb (absolute-error share of the blend, OOF)
cols = [BLUE if k != "Whale" else ORANGE for k in share]
ax[1].bar(list(share), list(share.values()), color=cols, width=0.6)
for i, v in enumerate(share.values()): ax[1].text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
ax[1].set_ylim(0, 0.92); ax[1].set_ylabel("share of the absolute error"); ax[1].tick_params(axis="x", labelsize=7.5); ax[1].grid(axis="x", visible=False)
ax[1].set_title("(b) the error sits with the whales", loc="left", fontsize=8.5, color=INK2)
plt.tight_layout(); save("fig_task3_target_error.png")

# ---------------------------------------------------------------- Figure 4: Task 5 gap in the period-4 change score
v = np.load(OUT / "task5_sorted_S.npy"); v = v[v > 0]
fig, ax = plt.subplots(figsize=(5.2, 2.7)); ax.plot(np.arange(len(v)), v, color=BLUE, lw=1.4); ax.set_yscale("log")
cut = 182; ax.axhline(cut, color=ORANGE, lw=1, ls="--")
n_a = int((v > cut).sum()); ax.annotate(f"gap between 150 and 221\nthreshold {cut}: {n_a} rows above (group A)", (len(v) * 0.38, cut), textcoords="offset points", xytext=(0, 14), fontsize=8, color=INK2)
ax.set_xlabel("players, sorted by the period-4 change score"); ax.set_ylabel("change score S (log scale)"); plt.tight_layout(); save("fig_task5_gap.png")

# ---------------------------------------------------------------- Figure 5: Task 4 progress and the class-prior step
steps = [("V27c", 0.83310, 0, "4 fresh" + chr(10) + "models", (14, -4), "left"), ("V31a", 0.83419, 0, "", None, ""), ("V32a", 0.83558, 0, "+ DINOv2-L", (0, 8), "center"), ("V36a", 0.83735, 0, "4 L seeds", (-4, 8), "right"),
         ("V40a", 0.83762, 0, "", None, ""), ("V43a", 0.83803, 0, "", None, ""), ("V45a", 0.83820, 0, "10 L seeds", (0, 9), "center"), ("V46a", 0.83816, 0, "", None, ""),
         ("V50b", 0.83960, 1, "1", (0, -12), "center"), ("V51b", 0.84073, 1, "2", (0, -12), "center"), ("V53a", 0.84106, 1, "4", (0, -12), "center"), ("V53c", 0.84113, 1, "4+B", (0, 8), "center"), ("V55a", 0.84109, 1, "5+B", (0, -12), "center")]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
xs = np.arange(len(steps)); ys = [t[1] for t in steps]; cc = [ORANGE if t[2] else BLUE for t in steps]
ax[0].plot(xs, ys, color=GRAY, lw=1); ax[0].scatter(xs, ys, c=cc, s=22, zorder=3)
for i, t in enumerate(steps):
    if t[3]: ax[0].annotate(t[3], (i, t[1]), textcoords="offset points", xytext=t[4], ha=t[5], fontsize=7, color=INK2)
ax[0].set_xticks(xs); ax[0].set_xticklabels([t[0] for t in steps], rotation=60, fontsize=7); ax[0].set_ylabel("Kaggle public score"); ax[0].set_ylim(0.8325, 0.8425)
ax[0].scatter([], [], c=BLUE, s=22, label="baseline recipe"); ax[0].scatter([], [], c=ORANGE, s=22, label="new recipe (number = DINOv2-L models, B = DINOv2-B)"); ax[0].legend(loc="upper left", fontsize=7)
ax[0].set_title("(a) Task 4 members added (Tasks 1-3, 5 fixed)", loc="left", fontsize=8.5, color=INK2)
pairs = [("V31", 0.83248, 0.83419), ("V41", 0.83615, 0.83782), ("V57", 0.83980, 0.84167)]; yp = np.arange(3)[::-1]
for yy, p in zip(yp, pairs): ax[1].plot([p[1], p[2]], [yy, yy], color=GRAY, lw=1.2, zorder=1); ax[1].text((p[1] + p[2]) / 2, yy + 0.16, f"+{p[2] - p[1]:.4f}", ha="center", fontsize=7.5, color=INK2)
ax[1].scatter([p[1] for p in pairs], yp, color=GRAY, s=30, zorder=3, label="without class-prior step"); ax[1].scatter([p[2] for p in pairs], yp, color=BLUE, s=30, zorder=3, label="with class-prior step")
ax[1].set_yticks(yp); ax[1].set_yticklabels([p[0] for p in pairs]); ax[1].set_ylim(-0.6, 2.8); ax[1].set_xlim(0.8315, 0.8440); ax[1].set_xlabel("Kaggle public score"); ax[1].legend(loc="lower left", fontsize=7); ax[1].grid(axis="y", visible=False)
ax[1].tick_params(axis="x", labelsize=7)
ax[1].set_title("(b) same members with and without EM", loc="left", fontsize=8.5, color=INK2)
plt.tight_layout(); save("fig_task4_progress.png")
