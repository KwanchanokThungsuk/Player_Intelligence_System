"""Round 3 - compare the NEW Task 4 Drive zip (2026-11-12) with the OLD one (read-only, no extraction).
Usage: python -I zip_compare.py <old.zip> <new.zip>"""
import sys, zipfile, io, collections
import numpy as np, pandas as pd
old, new = zipfile.ZipFile(sys.argv[1]), zipfile.ZipFile(sys.argv[2])
norm = lambda n: n.replace('task4_image_classification/', '', 1)
oi = {norm(i.filename): i for i in old.infolist() if not i.filename.startswith('__MACOSX') and norm(i.filename)}
ni = {norm(i.filename): i for i in new.infolist() if not i.filename.startswith('__MACOSX') and norm(i.filename)}
olds = {norm(i.filename): i.filename for i in old.infolist()}; news = {norm(i.filename): i.filename for i in new.infolist()}
print(f"entries: old {len(oi):,} | new {len(ni):,} | only in old {len(set(oi) - set(ni))} | only in new {len(set(ni) - set(oi))}")
extra = sorted(set(ni) - set(oi))[:8]; print("examples only in new:", extra)
print("non-image entries in new:", [n for n in ni if not n.lower().endswith(('.jpg', '.jpeg', '.png')) and not n.endswith('/')][:15])
common = [n for n in oi if n in ni and n.lower().endswith('.jpg')]
same = sum(oi[n].CRC == ni[n].CRC for n in common)
print(f"images present in both: {len(common):,} | byte-identical (CRC): {same:,} | different: {len(common) - same:,}")
for split in ("test", "train"):
    o = [n for n in oi if n.startswith(f"{split}/") and n.endswith(".jpg")]; nn = [n for n in ni if n.startswith(f"{split}/") and n.endswith(".jpg")]
    print(f"[{split}] images old {len(o):,} new {len(nn):,} | same names: {set(o) == set(nn)} | new entry order == sorted: {[x.split('/')[-1] for x in nn] == sorted(x.split('/')[-1] for x in nn)}")
tot_new = sum(i.file_size for i in new.infolist()); tot_old = sum(i.file_size for i in old.infolist())
print(f"uncompressed total: old {tot_old:,} new {tot_new:,} | compressed: old {sum(i.compress_size for i in old.infolist()):,} new {sum(i.compress_size for i in new.infolist()):,}")
print("compression types old:", dict(collections.Counter(i.compress_type for i in old.infolist())), "new:", dict(collections.Counter(i.compress_type for i in new.infolist())))
for name in ("test.csv", "train.csv"):
    a, b = pd.read_csv(io.BytesIO(old.read(olds[name]))), pd.read_csv(io.BytesIO(new.read(news[name])))
    ai, bi = a.id.astype(str).tolist(), b.id.astype(str).tolist()
    print(f"\n{name}: old {a.shape} new {b.shape} | columns old {list(a.columns)} new {list(b.columns)}")
    print(f"   same id set: {set(ai) == set(bi)} | identical order: {ai == bi} | rows in the same position: {np.mean(np.array(ai[:min(len(ai), len(bi))]) == np.array(bi[:min(len(ai), len(bi))])):.4f}")
    if set(ai) == set(bi) and ai != bi:
        pos = {s: k for k, s in enumerate(ai)}; order = np.array([pos[s] for s in bi])
        print(f"   correlation between old and new row positions: {np.corrcoef(np.arange(len(bi)), order)[0, 1]:.4f} | new first ids {bi[:4]} | old first ids {ai[:4]}")
    if name == "train.csv":
        m = a.merge(b, on="id", suffixes=("_o", "_n")); print("   train labels identical for common ids:", float((m["class_o"] == m["class_n"]).mean()))
    else:
        print("   class column non-null: old", int(a['class'].notna().sum()), "new", int(b['class'].notna().sum()))
