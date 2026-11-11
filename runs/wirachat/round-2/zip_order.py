"""Round 2 - read the entry list of the Task 4 Drive zip (no extraction) and compare its order with test.csv / train.csv.
Usage: python -I zip_order.py <path-to-zip>"""
import sys, zipfile, zlib, collections
from pathlib import Path
import numpy as np, pandas as pd

zpath = Path(sys.argv[1]); LOCAL = Path(sys.argv[2])            # LOCAL = .../cpe342-kaggle-main/04_image_classification
zf = zipfile.ZipFile(zpath)
infos = zf.infolist()
print(f"zip: {zpath.name} | {zpath.stat().st_size:,} bytes | {len(infos):,} entries | compression types: {dict(collections.Counter(i.compress_type for i in infos))}")
print("first 6 entries:", [i.filename for i in infos[:6]])
top = collections.Counter(i.filename.split('/')[0] + '/' + (i.filename.split('/')[1] if '/' in i.filename else '') for i in infos)
print("top-level folders:", dict(top.most_common(8)))
non_img = [i.filename for i in infos if not i.filename.lower().endswith(('.jpg', '.jpeg', '.png')) and not i.filename.endswith('/')]
print("non-image files in the zip:", non_img[:20])


def stem(n): return Path(n).stem


for split in ("test", "train"):
    ents = [i for i in infos if f"/{split}/" in i.filename and i.filename.lower().endswith(".jpg")]
    ids = [stem(i.filename) for i in ents]
    csv = pd.read_csv(LOCAL / f"{split}.csv").id.astype(str).tolist()
    same_set = set(ids) == set(csv)
    pos_csv = {s: k for k, s in enumerate(csv)}
    order = np.array([pos_csv[s] for s in ids if s in pos_csv])
    corr = float(np.corrcoef(np.arange(len(order)), order)[0, 1]) if len(order) > 2 else float('nan')
    print(f"\n[{split}] zip images {len(ids):,} vs {split}.csv {len(csv):,} | same id set: {same_set}")
    print(f"   zip order == {split}.csv order: {ids == csv} | == sorted(ids): {ids == sorted(ids)} | == reversed sorted: {ids == sorted(ids, reverse=True)}")
    print(f"   correlation between zip position and {split}.csv row: {corr:.4f} | first 5 zip ids: {ids[:5]} | first 5 csv ids: {csv[:5]}")
    adj = np.mean(np.array(ids[1:]) > np.array(ids[:-1])) if len(ids) > 1 else float('nan')
    print(f"   fraction of ascending adjacent pairs in zip order: {adj:.3f}")
    # modification times stored in the zip (the host's file creation order may show here)
    mt = [i.date_time for i in ents]
    print(f"   zip timestamps: first {mt[0]} last {mt[-1]} | monotone non-decreasing: {mt == sorted(mt)} | distinct values: {len(set(mt))}")

# csv files inside the zip vs the local copies
for i in infos:
    if i.filename.lower().endswith(".csv"):
        data = zf.read(i.filename); local = LOCAL / Path(i.filename).name
        print(f"\ncsv in zip: {i.filename} ({len(data):,} bytes) | local copy bytes equal: {local.exists() and local.read_bytes() == data} | local size {local.stat().st_size if local.exists() else 'n/a'}")

# CRC check of local images against the zip (no extraction): sample 2000 test + 2000 train
rng = np.random.default_rng(0); bad = 0; n = 0
crc = {stem(i.filename): (i.CRC, "/test/" in i.filename) for i in infos if i.filename.lower().endswith(".jpg")}
for split in ("test", "train"):
    ids = list(pd.read_csv(LOCAL / f"{split}.csv").id.astype(str))
    for s in rng.choice(ids, 2000, replace=False):
        n += 1; c = zlib.crc32((LOCAL / split / f"{s}.jpg").read_bytes()) & 0xFFFFFFFF
        if s not in crc or crc[s][0] != c: bad += 1
print(f"\nCRC32 check, local images vs zip entries: {n - bad}/{n} identical")
