"""Task 4 ResNet18 training and inference entry point.

Importing this module does not train. Use ``python -m src.task4.train``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from src.common import add_data_arguments, resolve_data_root

SEED = 342
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
IMAGE_SIZE = (224, 224)


def find_dataset(data_root=None, download=False) -> Path:
    root = resolve_data_root(data_root, download)
    candidates = sorted({p.parent for p in root.rglob("train.csv")
                         if (p.parent / "test.csv").is_file()
                         and (p.parent / "train").is_dir()
                         and (p.parent / "test").is_dir()})
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one Task 4 dataset, found: {candidates}")
    return candidates[0]


def validate_dataset(task_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    if not {"id", "class"}.issubset(train.columns) or "id" not in test.columns:
        raise ValueError(f"Unexpected Task 4 columns: train={train.columns.tolist()}, test={test.columns.tolist()}")
    if train.id.isna().any() or train.id.duplicated().any() or test.id.isna().any() or test.id.duplicated().any():
        raise ValueError("Task 4 IDs must be non-null and unique")
    classes = sorted(train["class"].astype(str).unique().tolist())
    mapping = {label: index for index, label in enumerate(classes)}
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    train_paths = {p.stem for p in (task_dir / "train").iterdir() if p.suffix.lower() in extensions}
    test_paths = {p.stem for p in (task_dir / "test").iterdir() if p.suffix.lower() in extensions}
    missing_train = set(train.id.astype(str)) - train_paths
    missing_test = set(test.id.astype(str)) - test_paths
    if missing_train or missing_test:
        raise FileNotFoundError(f"Missing images: train={len(missing_train)}, test={len(missing_test)}")
    return train, test, mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Task 4 ResNet18 and export predictions.")
    add_data_arguments(parser)
    parser.add_argument("--setup-only", action="store_true", help="Validate paths/schema/dependencies without training")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args(argv)
    task_dir = find_dataset(args.data_root, args.download)
    train, test, class_to_idx = validate_dataset(task_dir)
    print(f"Dataset: {task_dir}")
    print(f"Train/test shapes: {train.shape}, {test.shape}")
    print(f"Class mapping: {class_to_idx}")
    if args.setup_only:
        print("Setup-only complete; training was not started.")
        return

    try:
        import torch
        from PIL import Image
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError("Install torch, torchvision, pillow, pandas, and scikit-learn before training") from exc

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_indices, valid_indices = train_test_split(np.arange(len(train)), test_size=.20, random_state=SEED, stratify=train["class"])
    image_ext = {p.stem: p for p in (task_dir / "train").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    test_ext = {p.stem: p for p in (task_dir / "test").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}

    class GameImages(Dataset):
        def __init__(self, frame, paths, transform, labelled=True): self.frame=frame.reset_index(drop=True); self.paths=paths; self.transform=transform; self.labelled=labelled
        def __len__(self): return len(self.frame)
        def __getitem__(self, index):
            row=self.frame.iloc[index]
            with Image.open(self.paths[str(row.id)]) as image: image=image.convert("RGB")
            value=self.transform(image)
            return (value, class_to_idx[str(row["class"])]) if self.labelled else (value, str(row.id))

    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
    eval_tf=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.ToTensor(),normalize])
    train_loader=DataLoader(GameImages(train.iloc[train_indices],image_ext,train_tf),batch_size=args.batch_size,shuffle=True,num_workers=0)
    valid_loader=DataLoader(GameImages(train.iloc[valid_indices],image_ext,eval_tf),batch_size=128,shuffle=False,num_workers=0)
    test_loader=DataLoader(GameImages(test,test_ext,eval_tf,labelled=False),batch_size=128,shuffle=False,num_workers=0)
    try: model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception as exc: print(f"Pretrained weights unavailable; using random initialization: {exc}"); model=models.resnet18(weights=None)
    model.fc=nn.Linear(model.fc.in_features,len(class_to_idx)); model=model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate); criterion=nn.CrossEntropyLoss(); best=-1.
    root=Path(__file__).resolve().parents[1]; checkpoint=root.parent/"artifacts/task4_best_model.pt"
    for epoch in range(args.epochs):
        model.train()
        for images,labels in train_loader: optimizer.zero_grad(set_to_none=True); criterion(model(images.to(device)),labels.to(device)).backward(); optimizer.step()
        model.eval(); true=[]; pred=[]
        with torch.no_grad():
            for images,labels in valid_loader: pred.extend(model(images.to(device)).argmax(1).cpu().numpy()); true.extend(labels.numpy())
        score=f1_score(true,pred,average="macro"); print(f"epoch={epoch+1} valid_macro_f1={score:.4f}")
        if score>best: best=score; torch.save({"model_state_dict":model.state_dict(),"class_to_idx":class_to_idx,"best_macro_f1":best,"epoch":epoch+1},checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state_dict"]); model.eval(); ids=[]; predictions=[]
    with torch.no_grad():
        for images,batch_ids in test_loader: predictions.extend(model(images.to(device)).argmax(1).cpu().tolist()); ids.extend(list(batch_ids))
    output=pd.DataFrame({"id":ids,"task4":[{v:k for k,v in class_to_idx.items()}[int(i)] for i in predictions]})
    if output.id.tolist()!=test.id.astype(str).tolist(): raise ValueError("Task 4 output order does not match test.csv")
    output.to_csv(root.parent/"predictions/task4_test_predictions.csv",index=False); (root.parent/"artifacts/task4_class_to_idx.json").write_text(json.dumps(class_to_idx,indent=2)); print("Task 4 outputs written.")


if __name__ == "__main__":
    main()
