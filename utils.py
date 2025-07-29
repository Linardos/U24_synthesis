# utils_cv_helpers.py
# ---------------------------------------------------------------------
# helpers shared by hold-out and k-fold training
# ---------------------------------------------------------------------
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
from tqdm import tqdm

from data_loaders import make_balanced_loader


# ---------------------------------------------------------------------
# 1) BUILD TRAIN / VAL LOADERS
# ---------------------------------------------------------------------
def make_loaders(train_set,
                 val_set,
                 cfg,
                 train_tf,
                 train_samples=None):
    """
    Returns (train_loader, val_loader).

    If cfg["dynamic_balanced_sampling"] is True we *must* get the
    original `train_samples` list; if it is None we rebuild it
    on-the-fly from `train_set`.
    """
    if cfg["dynamic_balanced_sampling"]:
        # rebuild only if the caller didn't supply it
        if train_samples is None:
            train_samples = [s for s in train_set.datasets[0].samples] + \
                            [s for s in train_set.datasets[1].samples]
        train_loader = make_balanced_loader(
            samples     = train_samples,
            batch_size  = cfg["batch_size"],
            transform   = train_tf,
            ratio       = cfg["majority_to_minority_ratio"],
            num_workers = 8,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = cfg["batch_size"],
            shuffle    = True,
            num_workers= 8,
            pin_memory = True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size = cfg["batch_size"],
        shuffle    = False,
        num_workers= 8,
        pin_memory = True,
    )
    return train_loader, val_loader



# ---------------------------------------------------------------------
# 2) BUILD OPTIMISER + SCHEDULER
# ---------------------------------------------------------------------
def make_optim(model: nn.Module, cfg: dict):
    """
    AdamW with 10-× lower LR on feature extractor,
    CosineAnnealingLR for the schedule (same as hold-out run).
    """
    backbone, head = [], []
    for n, p in model.named_parameters():
        (head if n.startswith(("backbone.fc", "head")) else backbone).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": cfg["learning_rate"] * 0.1},
            {"params": head,     "lr": cfg["learning_rate"]},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    return optimizer, scheduler


# ---------------------------------------------------------------------
# 3) ONE EPOCH (train **or** eval) – returns metric list ready for CSV
# ---------------------------------------------------------------------
@torch.inference_mode()
def _forward_pass(model, x, y, criterion):
    logits = model(x)
    loss   = criterion(logits, y)
    probs  = torch.softmax(logits, 1)
    return loss, probs


def run_epoch(model, loader, is_train, optim,
              criterion, device, classes):
    """
    Mirrors the metric block inside holdout_train.py.
    Returns:
        [loss, overall_acc, AUC, balanced_acc, *per_class_acc]
    """
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    total, correct, run_loss = 0, 0, 0.0
    all_y, all_p = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        # forward / backward ------------------------------------------------
        if is_train:
            optim.zero_grad()
            loss, probs = _forward_pass(model, x, y, criterion)
            loss.backward()
            optim.step()
        else:
            loss, probs = _forward_pass(model, x, y, criterion)

        # accumulate --------------------------------------------------------
        preds = probs.argmax(1)
        run_loss += loss.item() * x.size(0)
        correct  += (preds == y).sum().item()
        total    += x.size(0)
        all_y.extend(y.cpu().numpy())
        all_p.extend(probs.cpu().numpy())

    # ---------------- metrics ---------------------------------------------
    all_y  = np.asarray(all_y)
    all_p  = np.asarray(all_p)
    preds  = all_p.argmax(1)

    cm = confusion_matrix(all_y, preds, labels=classes)
    tp = np.diag(cm)
    fn = cm.sum(1) - tp
    per_cls_acc = tp / (tp + fn + 1e-12)

    auc = (roc_auc_score(all_y, all_p[:, 1])
           if len(classes) == 2 else
           roc_auc_score(all_y, all_p, multi_class="ovr"))

    overall_acc  = correct / total
    bal_acc      = balanced_accuracy_score(all_y, preds)

    return [
        run_loss / total,
        overall_acc,
        auc,
        bal_acc,
        *per_cls_acc
    ]
