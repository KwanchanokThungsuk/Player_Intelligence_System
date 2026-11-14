"""Round 3 - Task 4: CLIP (openai/clip-vit-base-patch16) image embeddings for train+test (whole image squashed to 224x224, plus horizontal flip),
zero-shot logits from game-name prompts, saved to round-3/data3 (not committed). Pretrained models are allowed for Task 4 only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPTokenizer
warnings.filterwarnings("ignore")
R3 = Path(__file__).resolve().parent; D = R3.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
NAME = sys.argv[1] if len(sys.argv) > 1 else "openai/clip-vit-base-patch16"; TAG = NAME.split("/")[-1]
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
def feat(o): return o if torch.is_tensor(o) else o.pooler_output
class DS(Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        with Image.open(self.p[i]) as im: im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
        return (torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
if __name__ == "__main__":
    tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
    paths = [D / "train" / f"{i}.jpg" for i in tr.id.astype(str)] + [D / "test" / f"{i}.jpg" for i in te.id.astype(str)]
    model = CLIPModel.from_pretrained(NAME, torch_dtype=torch.float16).cuda().eval(); tok = CLIPTokenizer.from_pretrained(NAME)
    prompts = {"7-11k": ["a screenshot of the mobile game Seven Knights", "a screenshot of an anime mobile RPG with character cards and menus"],
               "fiveN": ["a screenshot of GTA V FiveM roleplay", "a screenshot of Grand Theft Auto 5 online roleplay server"],
               "freefried": ["a screenshot of Garena Free Fire", "a screenshot of a battle royale mobile game Free Fire"],
               "pube": ["a screenshot of PUBG Mobile", "a screenshot of PlayerUnknown's Battlegrounds"],
               "roblock": ["a screenshot of Roblox", "a screenshot of a blocky Roblox game"]}
    with torch.no_grad():
        T = []
        for c in ["7-11k", "fiveN", "freefried", "pube", "roblock"]:
            t = tok(prompts[c], padding=True, return_tensors="pt").to("cuda"); e = feat(model.get_text_features(**t)).float(); e = e / e.norm(dim=-1, keepdim=True); T.append(e.mean(0))
        T = torch.stack(T); T = T / T.norm(dim=-1, keepdim=True)
        print("text emb", tuple(T.shape), flush=True)
        E, Ef = [], []
        for x in DataLoader(DS(paths), batch_size=128, num_workers=8):
            x = x.cuda().half()
            for v, out in ((x, E), (torch.flip(x, dims=[3]), Ef)):
                f = feat(model.get_image_features(pixel_values=v)).float(); out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    E, Ef = np.concatenate(E), np.concatenate(Ef); n = len(tr)
    np.save(R3 / "data3" / f"clip_{TAG}_emb.npy", E); np.save(R3 / "data3" / f"clip_{TAG}_embflip.npy", Ef); np.save(R3 / "data3" / f"clip_{TAG}_text.npy", T.cpu().numpy())
    classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values
    zs = 100 * E @ T.cpu().numpy().T; from sklearn.metrics import f1_score
    print(f"{TAG}: zero-shot on TRAIN: acc {(zs[:n].argmax(1) == y).mean():.3f} macroF1 {f1_score(y, zs[:n].argmax(1), average='macro'):.3f} | test zero-shot class shares {np.round(np.bincount(zs[n:].argmax(1), minlength=5) / len(te), 3).tolist()}")
