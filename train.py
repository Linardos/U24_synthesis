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
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    average_precision_score, f1_score
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
val_split      = cfg.get("val_split", 0.1)   # unused now (kept for backward compat)

random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# NEW: separate roots
train_root = cfg["train_root_dir"]   # e.g. .../EMBED_binary_clean/train
val_root   = cfg["val_root_dir"]     # e.g. .../EMBED_binary_clean/val

real_path_train  = os.path.join(train_root, cfg["data_dir"])
synth_path_train = os.path.join(train_root, cfg["synth_data_dir"])
real_path_val    = os.path.join(val_root)

# ─────────────── experiment folder & (maybe) resume ────────────────────────
dataset_tag = Path(train_root).parts[-2]     # unchanged logic
synth_mult = cfg.get("synthetic_multiplier", 1.0)

fine_tune      = cfg.get("fine_tune", False)
fine_tune_ckpt = cfg.get("fine_tune_ckpt", "")
if fine_tune:
    real_fraction = 1.0
    synth_mult    = 0.0
    fine_tune_tag = "_fTune_" + fine_tune_ckpt[:4] + fine_tune_ckpt[-15:]
else:
    fine_tune_tag = ""

exp_dir = Path("experiments") / (
    f'{cfg["experiment_number"]:03d}_{dataset_tag}_holdout_{cfg["model_name"]}_'
    f'{cfg["experiment_name"]}_seed{seed}_Augs{cfg["augmentations"]}_'
    f'real{real_fraction}_syn{synth_mult}{fine_tune_tag}'
)

resume = exp_dir.exists()
if resume:
    print(f"Resuming experiment {exp_dir}")
else:
    exp_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy("config.yaml",     exp_dir / "config.yaml")
    shutil.copy("models.py",       exp_dir / "models.py")
    shutil.copy("holdout_train.py",exp_dir / "holdout_train.py")
    shutil.copy("data_loaders.py", exp_dir / "data_loaders.py")
    print(f"Initiating experiment {exp_dir}")

# --- transforms (unchanged) ----------------------------------------------
def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val + 1e-12)

base = [T.Lambda(min_max_normalization)]
basic = [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.3)]
geometric = [T.RandomRotation(degrees=8, fill=0),
             T.RandomAffine(degrees=0, translate=(0.02,0.02), scale=(0.95,1.05))]
intensity = [T.RandomApply([T.GaussianBlur(3,(0.1,0.8))],p=0.2),
             T.RandomApply([T.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.3)]
occlusion = [T.RandomErasing(p=0.25, scale=(0.01,0.05), ratio=(0.3,3.3), value='random')]

if cfg['augmentations'] == 'basic':
    aug_list = basic
elif cfg['augmentations'] == 'geometric':
    aug_list = basic + geometric
elif cfg['augmentations'] == 'intensity':
    aug_list = basic + geometric + intensity
elif cfg['augmentations'] == 'all':
    aug_list = basic + geometric + intensity + occlusion
else:
    aug_list = []

norm = [T.Normalize(mean=[0.5], std=[0.5])]
train_tf = T.Compose(base + aug_list + norm)
eval_tf  = T.Compose([T.Lambda(min_max_normalization),
                      T.Normalize(mean=[0.5], std=[0.5])])

# ─────────────── Load datasets (no internal split) ────────────────────────
print("Loading data from:")
print(f"  • train real : {real_path_train}")
print(f"  • train synth: {synth_path_train}")
print(f"  • val real   : {real_path_val}")

real_train_ds = NiftiDataset(real_path_train,  train_tf)
synth_ds      = NiftiDataset(synth_path_train, train_tf)   # augments OK on synth
val_ds        = NiftiDataset(real_path_val,    eval_tf)    # val is 100% real

# labels from train sets
def labels(ds):
    y = []
    for _, lab in DataLoader(ds, batch_size=1024, num_workers=8):
        y.extend(lab.numpy())
    return np.asarray(y)

real_lbl_train  = labels(real_train_ds)
synth_lbl_train = labels(synth_ds) if len(synth_ds) else np.array([], dtype=int)

classes = np.sort(np.unique(real_lbl_train))
print(f"Loaded  real_train={len(real_train_ds)}  synth_train={len(synth_ds)}  classes={classes}")
print(f"Validation size   ={len(val_ds)}")

rng = np.random.default_rng(seed)

# ─────────────── Build training set (class-wise keep + synth mix) ─────────
if cfg['sanity_check']:
    n_sanity   = max(1, int(round(len(real_train_ds) * 0.05)))
    sanity_idx = rng.choice(np.arange(len(real_train_ds)),
                            size=min(n_sanity, len(real_train_ds)),
                            replace=False)
    print(f"🧪  Sanity-check mode ON — over-fitting {len(sanity_idx)} samples")

    train_set     = Subset(real_train_ds, sanity_idx)
    val_set       = Subset(real_train_ds, sanity_idx)   # identical copy
    train_samples = [real_train_ds.samples[i] for i in sanity_idx]

    keep_real, add_synth = sanity_idx.tolist(), []
    cfg['dynamic_balanced_sampling'] = False
else:
    # collect pools by class
    synth_pool_by_cls = {c: np.where(synth_lbl_train == c)[0] for c in classes}

    keep_real, add_synth = [], []
    tr_idx_all = np.arange(len(real_train_ds))

    for c in classes:
        cls_idx = tr_idx_all[real_lbl_train[tr_idx_all] == c]
        rng.shuffle(cls_idx)

        # ❶ how many real to keep
        n_real_keep = int(round(len(cls_idx) * real_fraction))
        keep_real.extend(cls_idx[:n_real_keep])

        # ❷ how many synthetic to add
        base = n_real_keep if real_fraction > 0 else len(cls_idx)
        n_syn_needed = int(round(base * synth_mult))

        pool = synth_pool_by_cls[c]
        if n_syn_needed and len(pool):
            add_synth.extend(
                rng.choice(pool, size=min(n_syn_needed, len(pool)), replace=False)
            )
        elif n_syn_needed:
            print(f"⚠️  not enough synthetic samples for class {c} "
                  f"(requested {n_syn_needed}, found {len(pool)})")

    train_set = ConcatDataset([
        Subset(real_train_ds, keep_real),
        Subset(synth_ds,      add_synth)
    ])
    val_set   = val_ds  # <── your predefined validation folder

    train_samples = (
        [real_train_ds.samples[i] for i in keep_real] +
        [synth_ds.samples[i]      for i in add_synth]
    )

    json.dump(
        {"train_real": list(map(int, keep_real)),
         "train_synth": list(map(int, add_synth)),
         "val_from_dir": real_path_val},
        open(exp_dir / "indices.json", "w"), indent=2
    )

print(f"┌──── train sampling summary ────")
print(f"│ kept real      : {len(keep_real):5d}")
print(f"│ added synthetic: {len(add_synth):5d}")
print(f"│ total train    : {len(keep_real)+len(add_synth):5d}")
print(f"└──────────────────────────")

# loaders (unchanged)
if not cfg['dynamic_balanced_sampling']:
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)

val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

# ─────────────── model & optimisation ─────────────────────────────────────

def toggle_backbone(model, train_backbone: bool = True): 
    """
    Enable/disable grads for everything except the classification head.
    Works with timm ConvNeXt / ResNet style (model.backbone).
    Use only for fine-tuning
    """
    for n, p in model.named_parameters():
        if n.startswith("backbone"):
            p.requires_grad = train_backbone


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


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


sched = CosineAnnealingLR(optim, T_max=cfg["num_epochs"])

# ─────────────── load checkpoint (resume if continuing interrupted run, or fine-tune from a previous experiment) ────────────────────────────────
start_epoch, best_metric, patience = 1, -np.inf, 0
early_name = cfg.get("early_stopping_metric", "balanced_acc").lower()
ckpt_path = exp_dir / "last.pt"
if fine_tune and fine_tune_ckpt:                         # ← NEW  ❶
    ckpt_path_to_load = Path("experiments") / fine_tune_ckpt / "last.pt"
    print(f"🔄  fine-tune – loading weights from {ckpt_path_to_load}")
elif resume and (exp_dir / "last.pt").exists():          # ← existing logic
    ckpt_path_to_load = exp_dir / "last.pt"
else:
    ckpt_path_to_load = None

if ckpt_path_to_load:
    ckpt = torch.load(ckpt_path_to_load, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # fresh optimiser / scheduler if we are *fine-tuning*
    if not fine_tune:            # normal resume
        optim.load_state_dict(ckpt["optim_state"])
        sched.load_state_dict(ckpt["sched_state"])
        best_metric = ckpt.get("best_metric", ckpt.get("best_auc", -np.inf))
        patience  = ckpt["patience"]
        start_epoch = ckpt["epoch"] + 1
        print(f"→ resumed from epoch {ckpt['epoch']}  (best {early_name} {best_metric:.4f})")
    else:                        # fine-tune run
        print("→ weights loaded, optimiser/scheduler re-initialised for fine-tuning")

# ─────────────── CSV logger (append if resuming) ───────────────────────────
log_path   = exp_dir / "logs.csv"
file_exists = log_path.exists()
log_f      = open(log_path, "a" if file_exists else "w", newline="")
writer     = csv.writer(log_f)

header = ["epoch","phase","loss","overall_acc","AUC","balanced_acc"] + \
         [f"ACC_class{c}" for c in classes]

# write the header only if the file is new or empty
if not file_exists or os.path.getsize(log_path) == 0:
    writer.writerow(header)
# ─────────────── training loop ─────────────────────────────────────────────
freeze_epochs = cfg.get("freeze_backbone_epochs", 0)
for epoch in range(start_epoch, cfg["num_epochs"] + 1):
    # Toggle backbone training state exactly once when epoch == 1 and when it unfreezes
    if epoch == 1 and freeze_epochs > 0:
        toggle_backbone(model, train_backbone=False)
        print(f"🥶  Backbone frozen for the next {freeze_epochs} epoch(s)")
    if epoch == freeze_epochs + 1 and freeze_epochs > 0:
        toggle_backbone(model, train_backbone=True)
        print("🧊→🔥  Backbone unfrozen – fine-tuning whole network")

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
        balanced_acc = balanced_accuracy_score(all_y, preds)

        row = [epoch, phase,
               run_loss / total,
               correct / total,
               auc,
               balanced_accuracy_score(all_y, preds),
               *per_cls_acc]
        writer.writerow(row)

        print(f"{phase:5s}  loss={row[2]:.4f}  acc={row[3]:.4f}  acc_malignant={per_cls_acc[1]:.4f}  acc_benign={per_cls_acc[0]:.4f}   AUC={row[4]:.4f}")


        if phase == "val":
            
            metric_map = {
                "auc": auc,
                "balanced_acc": balanced_acc,
                "val_loss": -row[2],  # negate because lower loss is better
            }
            current = metric_map.get(early_name, balanced_acc)  # default to BA
            if current > best_metric + 0.001:
                best_metric, patience = current, 0
                torch.save({
                    "epoch":      epoch,
                    "model_state":model.state_dict(),
                    "optim_state":optim.state_dict(),
                    "sched_state":sched.state_dict(),
                    "best_metric":   best_metric,
                    "patience":   patience,
                }, ckpt_path)
                print(f"✓ checkpointed ({early_name} ↑)")
            else:
                patience += 1

    sched.step()
    if patience >= cfg["early_stopping_patience"]:
        print("Early-stopping: no ΔAUC for", patience, "epochs."); break
