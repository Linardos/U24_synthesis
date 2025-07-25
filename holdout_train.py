#!/usr/bin/env python
# train_holdout_monai.py  – single hold-out split with NiftiDataset
# -------------------------------------------------------------------------

import os, sys, json, yaml, random, shutil, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
from tqdm import tqdm

import torchvision.transforms as T
from data_loaders import NiftiDataset, make_balanced_loader
from models import get_model

# ─────────────── configuration ────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

seed           = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["seed"]
real_fraction  = cfg.get("real_percentage", 1.0)
val_split      = cfg.get("val_split", 0.1)

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

root_dir   = cfg["root_dir"]
real_path  = os.path.join(root_dir, cfg["data_dir"])
synth_path = os.path.join(root_dir, cfg["synth_data_dir"])

# ─────────────── experiment folder & (maybe) resume ────────────────────────
dataset_tag = Path(root_dir).parts[-2]           # -> 'EMBED' / 'CMMD'
exp_dir = Path("experiments") / (
    f'{cfg["experiment_number"]:03d}_{dataset_tag}_holdout_{cfg["model_name"]}_'
    f'{cfg["experiment_name"]}_seed{seed}_Augs{cfg["augmentations"]}_real_perc{real_fraction}'
)

resume = exp_dir.exists()        # ── key flag
if resume:
    print(f"Resuming experiment {exp_dir}")
else:
    exp_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy("config.yaml", exp_dir / "config.yaml")
    shutil.copy("models.py", exp_dir / "models.py")
    shutil.copy("holdout_train.py", exp_dir / "holdout_train.py")
    shutil.copy("data_loaders.py", exp_dir / "data_loaders.py")
    print(f"Initiating experiment {exp_dir}")

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
    
# --- always‑on steps -----------------------------------------------------
base = [T.Lambda(min_max_normalization)]

# --- augmentation bundles -----------------------------------------------
basic = [
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
]

geometric = [
    T.RandomRotation(degrees=8, fill=0),
    T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
]

intensity = [
    T.RandomApply([T.GaussianBlur(3,(0.1,0.8))],p=0.2),
    T.RandomApply([T.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.3),
]

occlusion = [
    T.RandomErasing(p=0.25, scale=(0.01, 0.05),
                    ratio=(0.3, 3.3), value='random'),
]

# --- choose bundle(s) ----------------------------------------------------

if cfg['augmentations'] == 'basic':
    aug_list = basic
elif cfg['augmentations'] == 'geometric':
    aug_list = basic + geometric
elif cfg['augmentations'] == 'intensity':
    aug_list = basic + geometric + intensity
elif cfg['augmentations'] == 'all':
    aug_list = basic + geometric + intensity + occlusion
else:                      # 'none' or anything unrecognised
    aug_list = []

# --- end‑of‑pipeline normalisation --------------------------------------
norm = [T.Normalize(mean=[0.5], std=[0.5])]

train_tf = T.Compose(base + aug_list + norm)

eval_tf = T.Compose([
    T.Lambda(min_max_normalization),
    T.Normalize(mean=[0.5], std=[0.5]),
])

# Load the dataset with transformations applied

print(f"Loading data from:\n  • {real_path}\n  • {synth_path}")
real_ds  = NiftiDataset(real_path,  train_tf)
synth_ds = NiftiDataset(synth_path, train_tf)   # augments OK on synth
val_ds   = NiftiDataset(real_path,  eval_tf)

def labels(ds):
    y = []
    for _, lab in DataLoader(ds, batch_size=1024, num_workers=8):
        y.extend(lab.numpy())
    return np.asarray(y)

real_lbl  = labels(real_ds)
synth_lbl = labels(synth_ds)

classes = np.sort(np.unique(real_lbl))
print(f"Loaded  real={len(real_ds)}  synth={len(synth_ds)}  classes={classes}")

# ─────────────── train / val split (real only) ────────────────────────────
tr_idx, val_idx = train_test_split(
    np.arange(len(real_ds)),
    test_size=val_split,
    stratify=real_lbl,
    random_state=seed,
)

rng = np.random.default_rng(seed)

# ── build mixed train subset ──────────────────────────────────────────────
synth_pool_by_cls = {c: np.where(synth_lbl == c)[0] for c in classes}

keep_real, add_synth = [], []
for c in classes:
    cls_idx = tr_idx[real_lbl[tr_idx] == c]
    rng.shuffle(cls_idx)

    n_keep = int(round(len(cls_idx) * real_fraction))
    n_need = len(cls_idx) - n_keep

    keep_real.extend(cls_idx[:n_keep])

    if n_need and len(synth_pool_by_cls[c]):
        add_synth.extend(rng.choice(synth_pool_by_cls[c], size=n_need, replace=False))
    elif n_need:
        print(f"⚠️  no synthetic samples for class {c}")

train_set = ConcatDataset([
    Subset(real_ds,  keep_real),
    Subset(synth_ds, add_synth)
])
val_set   = Subset(val_ds, val_idx)   # 100 % real
# this list powers the on-the-fly balanced loader
train_samples = (
    [real_ds.samples[i]  for i in keep_real] +
    [synth_ds.samples[i] for i in add_synth]
)

json.dump(
    {"train_real": list(map(int, keep_real)),
     "train_synth": list(map(int, add_synth)),
     "val_real":   list(map(int, val_idx))},
    open(exp_dir / "indices.json", "w"), indent=2
)

if not cfg['dynamic_balanced_sampling']:
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

val_loader = DataLoader(val_set,
                        batch_size=cfg["batch_size"],
                        shuffle=False)


# ─────────────── model & optimisation ─────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = get_model(cfg["model_name"], pretrained=True).to(device)
backbone_params = []
head_params     = []
for n,p in model.named_parameters():
    (head_params if n.startswith('backbone.fc') else backbone_params).append(p)

optim = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg["learning_rate"]*0.1}, # lower LR for backbone
        {'params': head_params,     'lr': cfg["learning_rate"]}
])

# optim     = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
# )
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


sched = CosineAnnealingLR(optim, T_max=cfg["num_epochs"])

# ─────────────── (optional) load checkpoint ────────────────────────────────
start_epoch, best_auc, patience = 1, -np.inf, 0
ckpt_path = exp_dir / "last.pt"
if resume and ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optim.load_state_dict(ckpt["optim_state"])
    sched.load_state_dict(ckpt["sched_state"])
    best_auc  = ckpt["best_auc"]
    patience  = ckpt["patience"]
    start_epoch = ckpt["epoch"] + 1
    print(f"→ Loaded checkpoint from epoch {ckpt['epoch']}  (best AUC {best_auc:.4f})")

# ─────────────── CSV logger (append if resuming) ───────────────────────────
log_mode = "a" if resume else "w"
log_f = open(exp_dir / "logs.csv", log_mode, newline="")
writer = csv.writer(log_f)
header = ["epoch","phase","loss","overall_acc","AUC","balanced_acc"] + \
         [f"ACC_class{c}" for c in classes]
if not resume:               # only write header on a fresh run
    writer.writerow(header)

# ─────────────── training loop ─────────────────────────────────────────────
for epoch in range(start_epoch, cfg["num_epochs"] + 1):

    # -- pick loader --------------------------------------------------
    if cfg['dynamic_balanced_sampling']:
        train_loader = make_balanced_loader(
            samples   = train_samples,
            batch_size= cfg["batch_size"],
            transform = train_tf,
            ratio     = cfg["majority_to_minority_ratio"],       # keep 1:1; tweak via cfg if you like
            num_workers = 8
        )
    # else: train_loader already defined above

    print(f"\nEpoch {epoch}/{cfg['num_epochs']}")
    for phase, loader in [("train", train_loader), ("val", val_loader)]:
        model.train(phase == "train")

        total, correct, run_loss = 0, 0, 0.0
        all_y, all_p = [], []

        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)

            with torch.set_grad_enabled(phase == "train"):
                logits = model(x)
                loss   = criterion(logits, y)

                if phase == "train":
                    optim.zero_grad(); loss.backward(); optim.step()

            probs  = torch.softmax(logits, 1).detach()
            preds  = probs.argmax(1)
            run_loss += loss.item() * x.size(0)
            correct  += (preds == y).sum().item()
            total    += x.size(0)
            all_y.extend(y.cpu().numpy()); all_p.extend(probs.cpu().numpy())

        # ── metrics ────────────────────────────────────────────────
        all_y  = np.asarray(all_y); all_p = np.asarray(all_p)
        preds  = all_p.argmax(1)

        cm = confusion_matrix(all_y, preds, labels=classes)
        tp = np.diag(cm); fn = cm.sum(1) - tp
        per_cls_acc = tp / (tp + fn + 1e-12)

        auc = (roc_auc_score(all_y, all_p[:,1])
               if len(classes) == 2 else
               roc_auc_score(all_y, all_p, multi_class='ovr'))

        row = [epoch, phase,
               run_loss / total,
               correct / total,
               auc,
               balanced_accuracy_score(all_y, preds),
               *per_cls_acc]
        writer.writerow(row)

        print(f"{phase:5s}  loss={row[2]:.4f}  acc={row[3]:.4f}  acc_malignant={per_cls_acc[1]:.4f}  acc_benign={per_cls_acc[0]:.4f}   AUC={row[4]:.4f}")

        if phase == "val":
            if auc > best_auc + 0.001:
                best_auc, patience = auc, 0
                torch.save({
                    "epoch":      epoch,
                    "model_state":model.state_dict(),
                    "optim_state":optim.state_dict(),
                    "sched_state":sched.state_dict(),
                    "best_auc":   best_auc,
                    "patience":   patience,
                }, ckpt_path)
                print("✓ checkpointed (AUC ↑)")
            else:
                patience += 1

    sched.step()
    if patience >= cfg["early_stopping_patience"]:
        print("Early-stopping: no ΔAUC for", patience, "epochs."); break
