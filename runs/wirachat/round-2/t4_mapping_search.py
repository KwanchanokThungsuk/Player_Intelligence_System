"""Round 2 - Task 4: find the integer encoding of the class labels from score feedback ("how many classes match").
Truth encoding pi is unknown; V4 (alphabetical = identity) matched about one class. Each guess g returns k = #classes where g == pi.
This only identifies the 5 class codes; it never touches individual row labels."""
import itertools, collections, json
from pathlib import Path

CLASSES = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
PERMS = list(itertools.permutations(range(5)))      # perm[i] = integer given to CLASSES[i]
fx = lambda g, p: sum(a == b for a, b in zip(g, p))
IDENT = tuple(range(5))
S1 = [p for p in PERMS if fx(IDENT, p) == 1]        # consistent with V4 matching exactly one class


def best_guess(S, tried):
    """pick the permutation that minimises (worst remaining set size, expected remaining size)"""
    best = None
    for g in PERMS:
        if g in tried: continue
        d = collections.Counter(fx(g, p) for p in S)
        key = (max(d.values()), sum(v * v for v in d.values()) / len(S))
        if best is None or key < best[0]: best = (key, g, d)
    return best[1], best[2]


def tree(S, tried, depth=0, lines=None, label="start"):
    lines = [] if lines is None else lines
    pad = "  " * depth
    if len(S) == 1:
        lines.append(f"{pad}{label}: ENCODING FOUND -> {dict(zip(CLASSES, S[0]))}")
        return lines
    if len(S) == 0:
        lines.append(f"{pad}{label}: no candidate left (assumptions wrong)")
        return lines
    g, d = best_guess(S, tried)
    lines.append(f"{pad}{label}: {len(S)} candidates -> upload guess {dict(zip(CLASSES, g))}")
    for k in sorted(d):
        sub = [p for p in S if fx(g, p) == k]
        tree(sub, tried | {g}, depth + 1, lines, f"if {k} classes match ({len(sub)} left)")
    return lines


if __name__ == "__main__":
    lines = tree(S1, {IDENT})
    out = Path(__file__).resolve().parent / "t4_mapping_plan.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    g1, d1 = best_guess(S1, {IDENT})
    json.dump({"first_guess": dict(zip(CLASSES, g1))}, open(Path(__file__).resolve().parent / "t4_first_guess.json", "w"))
