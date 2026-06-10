#!/usr/bin/env python3
"""
Self-contained Hybrid Training Script
Models: CBAM EfficientNet-B0 and CBAM ResNet50 classifier

What this does:
 - Builds both models inside this single script (no cross-file imports)
 - Trains with staged unfreezing (Head / Partial / Full) and modern tricks
 - Calibrates each model with temperature scaling on the validation split
 - Evaluates on the test set and computes a Hybrid fusion (0.7 EfficientNetB0 / 0.3 ResNet50)
 - Saves: CSV logs, text logs, confusion matrices (counts/normalized) PNGs, and a combined JSON report

Assumed data structure:
  archive/Training/<class>/*.jpg|png|jpeg
  archive/Testing/<class>/*.jpg|png|jpeg

Python 3.10+ recommended.
"""

from __future__ import annotations

import os
import re
import json
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import math
import random
import warnings
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torch import amp as torch_amp

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        average_precision_score,
        matthews_corrcoef,
        cohen_kappa_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed; advanced metrics limited.")


# ==============================
# Config
# ==============================


@dataclass
class StageCfg:
    name: str
    epochs: int
    lr: float
    target_acc: float
    use_mix: bool


@dataclass
class TrainCfg:
    seed: int = 42
    deterministic: bool = False
    classes: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # EfficientNet params
    efficientnetb0_variant: str = "b0"
    efficientnetb0_cbam_indices: Tuple[int, ...] = (2, 4, 6)
    efficientnetb0_dropout: float = 0.5
    efficientnetb0_stages: Tuple[StageCfg, ...] = (
        StageCfg("Head-Only", 3, 2e-3, 97.0, False),
        StageCfg("Partial", 6, 1e-3, 98.0, False),
        StageCfg("Full", 12, 7e-4, 99.0, True),
    )
    # ResNet50 params
    resnet50_cbam_on: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
    resnet50_dropout_head: float = 0.5
    resnet50_stages: Tuple[StageCfg, ...] = (
        StageCfg("Head-Only", 3, 1e-3, 97.0, False),
        StageCfg("Partial", 6, 9e-4, 98.0, True),
        StageCfg("Full", 12, 7e-4, 99.0, True),
    )
    # Training knobs (shared)
    weight_decay: float = 1e-4
    label_smoothing_head: float = 0.1
    label_smoothing_rest: float = 0.05
    mix_prob: float = 0.6
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    max_grad_norm: float = 0.6
    warmup_ratio: float = 0.1
    cosine_min_lr_ratio: float = 0.07
    ema_decay: float = 0.9995
    ema_warmup: int = 100
    use_ema: bool = True
    early_stop_patience: int = 5
    batch_size_auto: bool = True
    stratified_val_ratio: float = 0.15
    # Hybrid fusion weights (EfficientNetB0, ResNet50)
    hybrid_weights: Tuple[float, float] = (0.5, 0.5)
    # IO
    logs_dir: str = "training_logs"
    models_dir: str = "models"
    # Data root override (folder containing Training/ and Testing/). If None, auto-discover.
    data_root: str | None = None


CFG = TrainCfg()


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


set_seed(CFG.seed, CFG.deterministic)


# ==============================
# Logger
# ==============================


class SimpleLogger:
    def __init__(self, out_dir: str, prefix: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{prefix}_{self.session_id}"
        self.log_path = os.path.join(out_dir, f"{self.session_name}.log")
        self.csv_path = os.path.join(out_dir, f"{self.session_name}_metrics.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,stage,train_loss,train_acc,val_loss,val_acc,lr,epoch_time_s,gpu_mem_gb,bs,train_val_gap\n")
        self.log(f"SESSION START: {self.session_name}")

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_epoch(self, epoch, stage, train_loss, train_acc, val_loss, val_acc, lr, epoch_time, gpu_gb, bs, gap):
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{stage},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f},{lr:.6e},{epoch_time:.2f},{gpu_gb:.3f},{bs},{gap:.2f}\n")


# ==============================
# Data
# ==============================


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for c in classes:
            cdir = os.path.join(root_dir, c)
            if os.path.isdir(cdir):
                for name in os.listdir(cdir):
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cdir, name), self.class_to_idx[c]))
        if not self.samples:
            warnings.warn(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            from PIL import Image as PILImage

            img = PILImage.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


def build_datasets(classes: Tuple[str, ...], val_ratio: float):
    def find_dataset_base() -> Tuple[str, str]:
        # 1) Explicit override via env var
        env_base = os.environ.get("BRAIN_TUMOR_DATA_DIR")
        if env_base:
            tr = os.path.join(env_base, "Training")
            te = os.path.join(env_base, "Testing")
            if os.path.isdir(tr) and os.path.isdir(te):
                return tr, te
        # 1b) Config override
        if CFG.data_root:
            tr = os.path.join(CFG.data_root, "Training")
            te = os.path.join(CFG.data_root, "Testing")
            if os.path.isdir(tr) and os.path.isdir(te):
                return tr, te
        # 2) Build candidate roots around CWD and script location, including parent dirs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates_roots = []
        # Current working directory
        candidates_roots.append(os.path.join(os.getcwd(), "archive"))
        candidates_roots.append(os.path.join(os.getcwd(), "archive (2)"))
        candidates_roots.append(os.path.join(os.getcwd(), "dataset"))
        # Script dir
        candidates_roots.append(os.path.join(script_dir, "archive"))
        candidates_roots.append(os.path.join(script_dir, "archive (2)"))
        candidates_roots.append(os.path.join(script_dir, "dataset"))
        # Parent of script dir (repo root when running from training_scripts)
        parent = os.path.dirname(script_dir)
        candidates_roots.append(os.path.join(parent, "archive"))
        candidates_roots.append(os.path.join(parent, "archive (2)"))
        candidates_roots.append(os.path.join(parent, "dataset"))
        # Grandparent (defensive)
        grand = os.path.dirname(parent)
        candidates_roots.append(os.path.join(grand, "archive"))
        candidates_roots.append(os.path.join(grand, "archive (2)"))
        candidates_roots.append(os.path.join(grand, "dataset"))

        tried = []
        for base in candidates_roots:
            tr = os.path.join(base, "Training")
            te = os.path.join(base, "Testing")
            tried.append(base)
            if os.path.isdir(tr) and os.path.isdir(te):
                return tr, te
        # Not found
        raise RuntimeError(
            "Datasets not found. Tried the following bases for 'Training' and 'Testing':\n" +
            "\n".join(f" - {p}" for p in tried) +
            "\nTip: place data under 'dataset/Training' and 'dataset/Testing' at the repo root, or set --data-root / BRAIN_TUMOR_DATA_DIR to that folder."
        )

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2), value="random"),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr, te = find_dataset_base()
    train_ds = BrainTumorDataset(tr, list(classes), train_tf)
    test_ds = BrainTumorDataset(te, list(classes), test_tf)

    # Stratified split
    labels = np.array([lbl for _, lbl in train_ds.samples])
    indices = np.arange(len(labels))
    rng = np.random.default_rng(CFG.seed)
    val_indices, train_indices = [], []
    for c in range(len(classes)):
        cls_idx = indices[labels == c]
        rng.shuffle(cls_idx)
        k = int(len(cls_idx) * val_ratio)
        val_indices.extend(cls_idx[:k])
        train_indices.extend(cls_idx[k:])
    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)
    return train_subset, val_subset, test_ds


def auto_batch_size(enable_auto=True):
    if not enable_auto:
        return 32
    if not torch.cuda.is_available():
        return 16
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if mem_gb >= 24:
        return 128
    if mem_gb >= 12:
        return 64
    if mem_gb >= 8:
        return 32
    if mem_gb >= 4:
        return 16
    return 8


# ==============================
# CBAM Modules
# ==============================


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, max(1, in_ch // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_ch // reduction), in_ch),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        w = self.sig(a + m).view(x.size(0), -1, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        cat = torch.cat([avg, mx], 1)
        w = self.sig(self.conv(cat))
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ==============================
# Models
# ==============================


class CBAMEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5, cbam_indices=(2, 4, 6), use_cbam=True, variant="b0"):
        super().__init__()
        if variant == "b1":
            efficientnet_backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
        elif variant == "b2":
            efficientnet_backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V2)
        else:
            efficientnet_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = efficientnet_backbone.features
        self.use_cbam = use_cbam
        self.cbam_layers = nn.ModuleDict()
        # Discover out_channels per block
        block_channels = {}
        for idx, block in enumerate(self.features):
            out_ch = None
            for layer in reversed(block):
                if hasattr(layer, "out_channels"):
                    out_ch = layer.out_channels
                    break
            if out_ch is not None:
                block_channels[idx] = out_ch
        if use_cbam:
            for idx in cbam_indices:
                if idx in block_channels:
                    self.cbam_layers[f"cbam{idx}"] = CBAMBlock(block_channels[idx])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        for i, blk in enumerate(self.features):
            x = blk(x)
            if self.use_cbam and f"cbam{i}" in self.cbam_layers:
                x = self.cbam_layers[f"cbam{i}"](x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):  # type: ignore
        if self.training and input.size(0) == 1:
            was_training = self.training
            super().train(False)
            with torch.no_grad():
                out = super().forward(input)
            super().train(was_training)
            return out
        return super().forward(input)


class CBAMResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, use_cbam=True, cbam_on=("layer1", "layer2", "layer3", "layer4"), dropout_head=0.5):
        super().__init__()
        self.use_cbam = use_cbam
        self.cbam_on = set(cbam_on)
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        feat_dim = 2048
        # CBAM modules per layer
        self.cbam_modules = nn.ModuleDict()
        if self.use_cbam:
            for lname, ch in [("layer1", 256), ("layer2", 512), ("layer3", 1024), ("layer4", 2048)]:
                if lname in self.cbam_on:
                    self.cbam_modules[lname] = CBAMBlock(ch)
        # Multi-scale head
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), SafeBatchNorm1d(1024), nn.Dropout(dropout_head),
            nn.Linear(1024, 512), nn.ReLU(), SafeBatchNorm1d(512), nn.Dropout(0.4),
        )
        self.scale1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), SafeBatchNorm1d(256), nn.Dropout(0.3))
        self.scale2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), SafeBatchNorm1d(128), nn.Dropout(0.25))
        self.scale3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), SafeBatchNorm1d(64), nn.Dropout(0.2))
        merged = 256 + 128 + 64
        self.classifier = nn.Sequential(
            nn.Linear(merged, 512), nn.ReLU(), SafeBatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), SafeBatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), SafeBatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self._init_new()

    def _init_new(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_cbam(self, lname: str, x):
        if self.use_cbam and lname in self.cbam_modules:
            x = self.cbam_modules[lname](x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self, lname)(x)
            x = self._apply_cbam(lname, x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        att = self.attention(x)
        s1 = self.scale1(att)
        s2 = self.scale2(att)
        s3 = self.scale3(att)
        merged = torch.cat([s1, s2, s3], dim=1)
        return self.classifier(merged)


# ==============================
# Training helpers
# ==============================


class HybridModel(nn.Module):
    """
    End-to-end hybrid that fuses EfficientNet-B0 and a ResNet50 CBAM classifier
    by averaging their logits equally (0.5, 0.5). Gradients flow into both
    sub-models, enabling joint training with staged unfreezing.
    """

    def __init__(self, efficientnetb0: CBAMEfficientNetB0, resnet50: CBAMResNet50Classifier):
        super().__init__()
        self.efficientnetb0 = efficientnetb0
        self.resnet50 = resnet50

    def forward(self, x):
        efficientnetb0_logits = self.efficientnetb0(x)
        resnet50_logits = self.resnet50(x)
        return 0.5 * (efficientnetb0_logits + resnet50_logits)


# ==============================
# Robust metrics helpers (multiclass)
# ==============================

def macro_pr_auc_ovr(y_true_np: np.ndarray, probs_np: np.ndarray) -> float:
    """Compute macro-averaged PR-AUC across one-vs-rest classes."""
    n_classes = probs_np.shape[1]
    aps = []
    for c in range(n_classes):
        y_bin = (y_true_np == c).astype(int)
        try:
            ap = average_precision_score(y_bin, probs_np[:, c])
            if np.isfinite(ap):
                aps.append(ap)
        except Exception:
            continue
    return float(np.mean(aps)) if aps else float("nan")


def multiclass_brier(y_true_np: np.ndarray, probs_np: np.ndarray) -> float:
    """Mean over samples of sum_c (p_c - 1[y=c])^2."""
    n_classes = probs_np.shape[1]
    onehot = np.eye(n_classes)[y_true_np]
    se = (probs_np - onehot) ** 2
    return float(np.mean(np.sum(se, axis=1)))


def roc_auc_macro_ovr_safe(y_true_np: np.ndarray, probs_np: np.ndarray):
    """Robust macro ROC-AUC (OvR): try sklearn multiclass first, then per-class fallback."""
    try:
        return float(roc_auc_score(y_true_np, probs_np, average="macro", multi_class="ovr"))
    except Exception:
        # Fallback: per-class OvR manually averaged over classes that have both pos and neg
        aucs = []
        n_classes = probs_np.shape[1]
        for c in range(n_classes):
            y_bin = (y_true_np == c).astype(int)
            # Need both positive and negative samples for AUC
            if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
                continue
            try:
                aucs.append(roc_auc_score(y_bin, probs_np[:, c]))
            except Exception:
                continue
        return float(np.mean(aucs)) if aucs else None


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def apply_cutmix(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    box_area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - box_area / (x.size(2) * x.size(3))
    return x_cut, y, y[idx], lam_adj


class EMA:
    def __init__(self, model: nn.Module, decay=0.9995, warmup=100):
        self.decay = decay
        self.warmup = warmup
        self.shadow = {}
        self.num_updates = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        if self.num_updates < self.warmup:
            d = 1 - (1 - self.decay) * (self.num_updates / self.warmup)
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def apply(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        if hasattr(self, "backup"):
            for n, p in model.named_parameters():
                if n in self.backup:
                    p.data.copy_(self.backup[n])
            self.backup = {}


class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr_ratio=0.05, base_lr=None):
        self.opt = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        self.base_lrs = [pg["lr"] if base_lr is None else base_lr for pg in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, pg in enumerate(self.opt.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * self.current_step / max(1, self.warmup_steps)
            else:
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * cosine
            pg["lr"] = lr

    def get_lr(self):
        return [pg["lr"] for pg in self.opt.param_groups]


# ==============================
# Train/Eval/Calibration
# ==============================


def set_trainable_efficientnetb0(model: nn.Module, stage: str):
    if stage == "Head-Only":
        for n, p in model.named_parameters():
            p.requires_grad = ("classifier" in n)
    elif stage == "Partial":
        for n, p in model.named_parameters():
            if any(f"features.{i}" in n for i in [5, 6, 7]) or ("classifier" in n) or ("cbam" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True


def set_trainable_resnet50(model: nn.Module, stage: str):
    head_keys = ["attention", "scale", "classifier", "cbam_modules"]
    if stage == "Head-Only":
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in head_keys)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    elif stage == "Partial":
        for n, p in model.named_parameters():
            if any(k in n for k in ["layer4", *head_keys]):
                p.requires_grad = True
            else:
                p.requires_grad = False
        for m in model.layer4.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.train()


def set_trainable_hybrid(model: "HybridModel", stage: str):
    """Apply staged trainability to both sub-models consistently."""
    set_trainable_efficientnetb0(model.efficientnetb0, stage)
    set_trainable_resnet50(model.resnet50, stage)


def train_staged(model: nn.Module, train_loader, val_loader, device: str, stages: Tuple[StageCfg, ...], logger: SimpleLogger, save_prefix: str):
    scaler = torch_amp.GradScaler(enabled=torch.cuda.is_available())
    ema = EMA(model, decay=CFG.ema_decay, warmup=CFG.ema_warmup) if CFG.use_ema else None

    best_global_acc = 0.0
    best_global_state = None

    def bs(loader):
        return getattr(loader, "batch_size", 0)

    for si, stage in enumerate(stages):
        if isinstance(model, CBAMEfficientNetB0):
            set_trainable_efficientnetb0(model, stage.name)
        elif isinstance(model, CBAMResNet50Classifier):
            set_trainable_resnet50(model, stage.name)
        elif isinstance(model, HybridModel):
            set_trainable_hybrid(model, stage.name)
        else:
            # Fallback: make all params trainable
            for p in model.parameters():
                p.requires_grad = True
        smoothing = CFG.label_smoothing_head if stage.name == "Head-Only" else CFG.label_smoothing_rest

        # Merge per-backbone schedules when training HybridModel
        if isinstance(model, HybridModel):
            # Match ResNet50 stage by index (expects aligned names: Head-Only/Partial/Full)
            resnet50_stage = CFG.resnet50_stages[si] if si < len(CFG.resnet50_stages) else CFG.resnet50_stages[-1]
            efficientnetb0_lr = stage.lr
            resnet50_lr = resnet50_stage.lr
            stage_epochs = max(stage.epochs, resnet50_stage.epochs)
            target_acc = max(stage.target_acc, resnet50_stage.target_acc)
            use_mix_flag = stage.use_mix or resnet50_stage.use_mix
            efficientnetb0_params = [p for p in model.efficientnetb0.parameters() if p.requires_grad]
            resnet50_params = [p for p in model.resnet50.parameters() if p.requires_grad]
            param_groups = []
            if efficientnetb0_params:
                param_groups.append({"params": efficientnetb0_params, "lr": efficientnetb0_lr})
            if resnet50_params:
                param_groups.append({"params": resnet50_params, "lr": resnet50_lr})
            optimizer = optim.AdamW(param_groups, weight_decay=CFG.weight_decay)
            steps_per_epoch = max(1, len(train_loader))
            total_steps = steps_per_epoch * stage_epochs
            trainable_params = efficientnetb0_params + resnet50_params
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=stage.lr, weight_decay=CFG.weight_decay)
            steps_per_epoch = max(1, len(train_loader))
            total_steps = steps_per_epoch * stage.epochs
            target_acc = stage.target_acc
            use_mix_flag = stage.use_mix
            trainable_params = params
        warmup_steps = int(CFG.warmup_ratio * total_steps)
        scheduler = WarmupCosine(optimizer, total_steps, warmup_steps, min_lr_ratio=CFG.cosine_min_lr_ratio)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        ce_no_smooth = nn.CrossEntropyLoss(label_smoothing=0.0)

        stage_best = 0.0
        stage_best_state = None
        no_improve = 0

        # Determine loop epochs
        num_epochs = stage_epochs if isinstance(model, HybridModel) else stage.epochs
        for ep in range(1, num_epochs + 1):
            model.train()
            t0 = time.time()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                use_mix = use_mix_flag and (random.random() < CFG.mix_prob)
                with torch_amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    if use_mix:
                        if random.random() < 0.5:
                            xm, ya, yb, lam = apply_mixup(x, y, CFG.mixup_alpha)
                        else:
                            xm, ya, yb, lam = apply_cutmix(x, y, CFG.cutmix_alpha)
                        logits = model(xm)
                        loss = lam * ce_no_smooth(logits, ya) + (1 - lam) * ce_no_smooth(logits, yb)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                if CFG.max_grad_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, CFG.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if ema:
                    ema.update(model)
                epoch_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = epoch_loss / max(1, len(train_loader))
            train_acc = 100.0 * correct / max(1, total)

            # Validation
            model.eval()
            val_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv = xv.to(device)
                    yv = yv.to(device)
                    with torch_amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        v_logits = model(xv)
                        v_loss = criterion(v_logits, yv)
                    if torch.isfinite(v_loss):
                        val_loss += v_loss.item()
                    v_preds = v_logits.argmax(1)
                    v_correct += (v_preds == yv).sum().item()
                    v_total += yv.size(0)
            val_loss = val_loss / max(1, len(val_loader))
            val_acc = 100.0 * v_correct / max(1, v_total)
            gap = train_acc - val_acc
            ep_time = time.time() - t0
            gpu_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            logger.log_epoch(ep, stage.name, train_loss, train_acc, val_loss, val_acc, scheduler.get_lr()[0], ep_time, gpu_gb, bs(train_loader), gap)

            if val_acc > stage_best:
                stage_best = val_acc
                stage_best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
                logger.log(f"[{save_prefix}:{stage.name}] New stage best: {val_acc:.2f}%")
            else:
                no_improve += 1

            if val_acc > best_global_acc:
                best_global_acc = val_acc
                best_global_state = {k: v.cpu() for k, v in model.state_dict().items()}
                logger.log(f"[GLOBAL {save_prefix}] New best val acc: {val_acc:.2f}% ({stage.name})")

            # Stage target check
            stage_target = target_acc if isinstance(model, HybridModel) else stage.target_acc
            if val_acc >= stage_target and gap < 10.0:
                logger.log(f"[{save_prefix}:{stage.name}] Target {stage_target:.2f}% reached. Ending stage early.")
                break
            if no_improve >= CFG.early_stop_patience:
                logger.log(f"[{save_prefix}:{stage.name}] Early stopping (no improvement {no_improve}).")
                break

        # Save stage-best
        if stage_best_state is not None:
            os.makedirs(CFG.models_dir, exist_ok=True)
            out_path = os.path.join(CFG.models_dir, f"model_stage_best_{save_prefix}_{stage.name}_{logger.session_id}.pth")
            torch.save(stage_best_state, out_path)
            logger.log(f"Saved stage-best weights -> {out_path}")

    return {"state_dict": best_global_state, "best_val_acc": best_global_acc}


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: str):
    model.eval()
    ys, preds, logits = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch_amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            logit = model(x)
        p = logit.argmax(1)
        ys.append(y.cpu())
        preds.append(p.cpu())
        logits.append(logit.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    y_logits = torch.cat(logits).numpy()
    return y_true, y_pred, y_logits


def temperature_search(logits: torch.Tensor, labels: torch.Tensor, t_min=0.3, t_max=1.5, t_step=0.05):
    best_t, best_nll = 1.0, float("inf")
    for T in np.arange(t_min, t_max + 1e-9, t_step):
        logp = torch.log_softmax(logits / float(T), dim=-1)
        nll = -logp[torch.arange(labels.size(0)), labels].mean().item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(T)
    return best_t, best_nll


def softmax_probs(logits: torch.Tensor):
    return torch.softmax(logits, dim=1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_all_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    results = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))) if SKLEARN_AVAILABLE else None
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if (SKLEARN_AVAILABLE and cm is not None) else None
    if SKLEARN_AVAILABLE:
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        macro_f1 = f1.mean()
        weighted_f1 = (f1 * support).sum() / support.sum()
        balanced_acc = rec.mean()
        try:
            roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(num_classes)))
        except Exception:
            roc_auc_macro = None
        pr_auc_per_class = []
        for c in range(num_classes):
            try:
                pr_auc = average_precision_score((y_true == c).astype(int), y_prob[:, c])
            except Exception:
                pr_auc = None
            pr_auc_per_class.append(pr_auc)
        pr_auc_macro = np.nanmean([v for v in pr_auc_per_class if v is not None]) if pr_auc_per_class else None
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = None
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = None
    else:
        macro_f1 = weighted_f1 = balanced_acc = roc_auc_macro = pr_auc_macro = mcc = kappa = None

    ece = expected_calibration_error(y_prob, y_true, n_bins=15)
    brier = brier_score(y_prob, y_true, num_classes)

    per_class = []
    if SKLEARN_AVAILABLE and cm is not None:
        for i, cname in enumerate(class_names):
            per_class.append(
                {
                    "class": cname,
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
            )

    results.update(
        {
            "confusion_matrix_raw": None if cm is None else cm.tolist(),
            "confusion_matrix_normalized": None if cm_norm is None else cm_norm.tolist(),
            "per_class": per_class,
            "macro_f1": None if macro_f1 is None else float(macro_f1),
            "weighted_f1": None if weighted_f1 is None else float(weighted_f1),
            "balanced_accuracy": None if balanced_acc is None else float(balanced_acc),
            "roc_auc_macro_ovr": None if roc_auc_macro is None else float(roc_auc_macro),
            "pr_auc_macro": None if pr_auc_macro is None else float(pr_auc_macro),
            "mcc": None if mcc is None else float(mcc),
            "cohen_kappa": None if kappa is None else float(kappa),
            "ece": ece,
            "brier_score": brier,
        }
    )
    return results


def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix", fmt="d", cmap="Blues"):
    if cm is None:
        return
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ==============================
# Unified Evaluation Helpers
# ==============================

def _build_model(kind: str, num_classes: int):
    kind_l = kind.lower()
    if kind_l == "efficientnet":
        return CBAMEfficientNetB0(
            num_classes=num_classes,
            dropout=CFG.efficientnetb0_dropout,
            cbam_indices=CFG.efficientnetb0_cbam_indices,
            use_cbam=True,
            variant=CFG.efficientnetb0_variant,
        )
    if kind_l == "resnet50":
        return CBAMResNet50Classifier(
            num_classes=num_classes,
            use_cbam=True,
            cbam_on=CFG.resnet50_cbam_on,
            dropout_head=CFG.resnet50_dropout_head,
        )
    if kind_l == "hybrid":
        efficientnetb0_model = CBAMEfficientNetB0(
            num_classes=num_classes,
            dropout=CFG.efficientnetb0_dropout,
            cbam_indices=CFG.efficientnetb0_cbam_indices,
            use_cbam=True,
            variant=CFG.efficientnetb0_variant,
        )
        resnet50_model = CBAMResNet50Classifier(
            num_classes=num_classes,
            use_cbam=True,
            cbam_on=CFG.resnet50_cbam_on,
            dropout_head=CFG.resnet50_dropout_head,
        )
        return HybridModel(efficientnetb0_model, resnet50_model)
    raise ValueError(f"Unknown model kind: {kind}")


def _auto_discover_checkpoint(kind: str):
    """Return absolute path to latest discovered checkpoint for given kind."""
    mdir = CFG.models_dir
    if not os.path.isdir(mdir):
        return None
    cand = []
    low_kind = kind.lower()
    for fn in os.listdir(mdir):
        path = os.path.join(mdir, fn)
        low = fn.lower()
        if low_kind == "efficientnet":
            if low == "efficientnet_full.pth" or (fn.startswith("model_stage_best_EfficientNet_") and fn.endswith(".pth")):
                cand.append(path)
        elif low_kind == "resnet50":
            if low == "resnet50_full.pth" or (fn.startswith("model_stage_best_ResNet50_") and fn.endswith(".pth")):
                cand.append(path)
        elif low_kind == "hybrid":
            if (fn.startswith("model_stage_best_Hybrid_Full_") and fn.endswith(".pth")) or low == "hybrid_full.pth":
                cand.append(path)
            elif (fn.startswith("model_stage_best_Hybrid_") and fn.endswith(".pth")) or (low.startswith("hybrid_") and low.endswith(".pth")):
                cand.append(path)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _maybe_load_temperature_from_report(report_path: str):
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Try common locations
        if isinstance(data, dict):
            if "temperature" in data and isinstance(data["temperature"], (int, float)):
                return float(data["temperature"])
            # Nested metrics structure (calibration block) fallback
            calib = data.get("calibration") if isinstance(data.get("calibration"), dict) else None
            if calib and "temperature" in calib:
                return float(calib["temperature"])
        return None
    except Exception:
        return None


def _temperature_strategy(kind: str, strategy: str, val_logits: np.ndarray, val_labels: np.ndarray, explicit_report: str | None):
    strategy = strategy.lower()
    if strategy == "none":
        return 1.0, {"source": "none"}
    if strategy == "report":
        if not explicit_report:
            raise SystemExit("--temp-source report requires --temp-report path")
        T = _maybe_load_temperature_from_report(explicit_report)
        if T is None:
            raise SystemExit("Temperature not found in provided report")
        return T, {"source": "report", "report_path": explicit_report}
    if strategy == "auto":
        # Heuristic: search logs dir for latest report referencing kind
        logs = CFG.logs_dir
        best_match = None
        if os.path.isdir(logs):
            rpt_files = [os.path.join(logs, f) for f in os.listdir(logs) if f.lower().endswith(".json")]
            rpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for rp in rpt_files:
                try:
                    with open(rp, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    text_blob = json.dumps(d).lower()
                    if kind.lower() in text_blob:
                        T = None
                        if "temperature" in d and isinstance(d["temperature"], (int, float)):
                            T = float(d["temperature"])
                        elif isinstance(d.get("calibration"), dict) and isinstance(d["calibration"].get("temperature"), (int, float)):
                            T = float(d["calibration"]["temperature"])
                        if T is not None:
                            return T, {"source": "auto_report", "report_path": rp}
                except Exception:
                    continue
        # Fallback to recompute
        strategy = "recompute"
    if strategy == "recompute":
        T, best_nll = temperature_search(torch.from_numpy(val_logits), torch.from_numpy(val_labels))
        return float(T), {"source": "recompute", "val_nll": float(best_nll)}
    raise SystemExit(f"Unknown --temp-source strategy: {strategy}")


def _metrics_block(y_true: np.ndarray, logits: np.ndarray, T: float | None, class_names: list[str]):
    probs_raw = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    metrics_raw = compute_all_metrics(y_true, probs_raw.argmax(1), probs_raw, len(class_names), class_names)
    # Enrich with robust helpers
    try:
        metrics_raw["pr_auc_macro"] = macro_pr_auc_ovr(y_true, probs_raw)
        metrics_raw["brier_score"] = multiclass_brier(y_true, probs_raw)
        metrics_raw["roc_auc_macro_ovr"] = roc_auc_macro_ovr_safe(y_true, probs_raw)
    except Exception:
        pass
    out = {"raw": metrics_raw}
    if T is not None and abs(T - 1.0) > 1e-6:
        probs_cal = torch.softmax(torch.from_numpy(logits) / T, dim=-1).numpy()
        metrics_cal = compute_all_metrics(y_true, probs_cal.argmax(1), probs_cal, len(class_names), class_names)
        try:
            metrics_cal["pr_auc_macro"] = macro_pr_auc_ovr(y_true, probs_cal)
            metrics_cal["brier_score"] = multiclass_brier(y_true, probs_cal)
            metrics_cal["roc_auc_macro_ovr"] = roc_auc_macro_ovr_safe(y_true, probs_cal)
        except Exception:
            pass
        out["calibrated"] = metrics_cal
    return out


def _export_curves(y_true: np.ndarray, logits: np.ndarray, T: float | None):
    # Deprecated full export path replaced below by summary mode toggle inside unified_evaluate.
    # Left here as a thin wrapper for backward compatibility (will be overridden logically).
    return {}


def _export_curves_summary_or_full(y_true: np.ndarray, probs: np.ndarray, full: bool):
    curves: dict[str, Any] = {}
    if not SKLEARN_AVAILABLE:
        return curves
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    num_classes = probs.shape[1]
    # Always compute AUC summaries
    per_class_pr_auc: dict[str, float | None] = {}
    roc_macro_values = []
    # For full mode, store coordinate arrays; otherwise just AUCs
    if full:
        fpr_grid = np.linspace(0, 1, 200)
        tpr_accum = []
        recall_grid = np.linspace(0, 1, 200)
        pr_accum = []
        pr_curves = {}
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
            roc_auc_c = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc_c = np.array([0,1]), np.array([0,1]), None
        try:
            prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
            pr_auc_c = average_precision_score(y_bin, probs[:, c])
        except Exception:
            prec, rec, pr_auc_c = np.array([1,1]), np.array([0,1]), None
        per_class_pr_auc[str(c)] = None if pr_auc_c is None else float(pr_auc_c)
        if full:
            # Interpolate ROC
            if len(np.unique(fpr)) > 1:
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
            else:
                tpr_interp = np.linspace(0,1,200)
            tpr_accum.append(tpr_interp)
            # Interpolate PR (reverse recall for monotonicity if needed)
            r_seq = rec
            p_seq = prec
            if len(np.unique(r_seq)) > 1:
                pr_interp = np.interp(recall_grid, r_seq[::-1], p_seq[::-1], left=1.0, right=p_seq[-1])
                pr_accum.append(pr_interp)
            pr_curves[str(c)] = {
                "precision": p_seq.tolist(),
                "recall": r_seq.tolist(),
                "pr_auc": None if pr_auc_c is None else float(pr_auc_c),
            }
        if roc_auc_c is not None:
            roc_macro_values.append(roc_auc_c)
    # Macro summaries
    curves["roc_macro_auc_mean"] = float(np.mean(roc_macro_values)) if roc_macro_values else None
    curves["pr_per_class_auc"] = per_class_pr_auc
    # Macro PR AUC (already computed elsewhere but convenient)
    try:
        curves["pr_macro_auc"] = float(macro_pr_auc_ovr(y_true, probs))
    except Exception:
        curves["pr_macro_auc"] = None
    if full:
        curves["roc_macro"] = {
            "fpr": fpr_grid.tolist(),
            "tpr_mean": np.mean(tpr_accum, axis=0).tolist() if tpr_accum else None,
        }
        curves["pr_macro"] = {
            "recall": recall_grid.tolist(),
            "precision_mean": np.mean(pr_accum, axis=0).tolist() if pr_accum else None,
        }
        curves["pr_per_class_curves"] = pr_curves
    return curves


def unified_evaluate(kind: str, checkpoint: str | None, temp_source: str, temp_report: str | None, save_probs: bool, export_curves: bool, deterministic: bool, full_curves: bool):
    import time, hashlib

    def normalize_checkpoint_state(model_kind: str, raw_state: Any):
        state = raw_state.get("state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
        if not isinstance(state, dict):
            return state
        renamed = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if model_kind == "hybrid":
                if new_key.startswith("eff."):
                    new_key = "efficientnetb0." + new_key[len("eff."):]
                elif new_key.startswith("tdn."):
                    new_key = "resnet50." + new_key[len("tdn."):]
            renamed[new_key] = value
        return renamed

    device = CFG.device
    num_classes = len(CFG.classes)
    print(f"[UnifiedEval] Starting kind={kind} on device={device}")
    # Determinism
    seed_used = None
    if deterministic:
        seed_used = 42
        random.seed(seed_used)
        np.random.seed(seed_used)
        torch.manual_seed(seed_used)
        torch.cuda.manual_seed_all(seed_used)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data (fresh build for unified eval)
    _train_ds, val_ds, test_ds = build_datasets(CFG.classes, CFG.stratified_val_ratio)
    print(f"[UnifiedEval] Dataset ready: val={len(val_ds)} test={len(test_ds)}")
    try:
        bs = auto_batch_size() if CFG.batch_size_auto else 64
    except NameError:
        bs = 64
    nw = min(8, os.cpu_count() or 4)
    print(f"[UnifiedEval] DataLoader config: batch_size={bs} workers={nw}")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    # Model & checkpoint
    model = _build_model(kind, num_classes).to(device)
    ckpt_path = checkpoint or _auto_discover_checkpoint(kind)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint found for kind={kind}; supply --checkpoint")
    print(f"[UnifiedEval] Loading checkpoint: {ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    state = normalize_checkpoint_state(kind, state)
    load_result = model.load_state_dict(state, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        missing_preview = ", ".join(missing_keys[:5])
        unexpected_preview = ", ".join(unexpected_keys[:5])
        raise RuntimeError(
            f"Checkpoint load mismatch for kind={kind}. "
            f"missing={len(missing_keys)} [{missing_preview}] "
            f"unexpected={len(unexpected_keys)} [{unexpected_preview}]"
        )
    model.eval()
    print("[UnifiedEval] Model loaded; running validation pass for temperature")

    # Validation logits (for temperature)
    val_labels, _, val_logits = run_eval(model, val_loader, device)
    T, T_meta = _temperature_strategy(kind, temp_source, val_logits, val_labels, temp_report)
    print(f"[UnifiedEval] Temperature ready: T={float(T):.4f} source={T_meta.get('source', 'unknown')}")

    # Test logits
    print("[UnifiedEval] Running test pass")
    y_true, _, test_logits = run_eval(model, test_loader, device)
    metrics = _metrics_block(y_true, test_logits, T if temp_source != "none" else None, CFG.classes)
    print("[UnifiedEval] Metrics computed")

    # Accuracy convenience fields
    raw_probs = torch.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    raw_pred = raw_probs.argmax(1)
    acc_raw = float((raw_pred == y_true).mean() * 100.0)
    acc_cal = None
    if temp_source != "none" and (T_meta.get("source") != "none"):
        cal_probs = torch.softmax(torch.from_numpy(test_logits) / T, dim=-1).numpy()
        cal_pred = cal_probs.argmax(1)
        acc_cal = float((cal_pred == y_true).mean() * 100.0)

    # Curves
    if export_curves:
        # Choose calibrated probs if available, else raw
        use_probs = torch.softmax(torch.from_numpy(test_logits) / (T if (temp_source != "none") else 1.0), dim=-1).numpy()
        curves = _export_curves_summary_or_full(y_true, use_probs, full_curves)
    else:
        curves = {}

    # Confusion matrices (use calibrated if available else raw)
    chosen_probs = cal_probs if (acc_cal is not None) else raw_probs  # type: ignore[name-defined]
    cm = confusion_matrix(y_true, chosen_probs.argmax(1), labels=list(range(num_classes))) if SKLEARN_AVAILABLE else None
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if (cm is not None) else None
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(CFG.logs_dir, f"cm_{kind}_unified_{sid}.png")
    cmn_path = os.path.join(CFG.logs_dir, f"cm_{kind}_unified_norm_{sid}.png")
    save_confusion_matrix(cm, CFG.classes, cm_path, title=f"{kind.title()} CM (Unified)")
    save_confusion_matrix(cm_norm, CFG.classes, cmn_path, title=f"{kind.title()} CM (Unified Norm)", fmt=".2f", cmap="Greens")
    print("[UnifiedEval] Confusion matrices saved")

    # Provenance
    ckpt_stat = os.stat(ckpt_path)
    sha256 = None
    try:
        h = hashlib.sha256()
        with open(ckpt_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        sha256 = h.hexdigest()
    except Exception:
        pass
    provenance = {
        "model_kind": kind,
        "checkpoint": {
            "path": ckpt_path,
            "mtime": ckpt_stat.st_mtime,
            "size_bytes": ckpt_stat.st_size,
            "sha256": sha256,
        },
        "temperature": {"value": float(T), **T_meta} if temp_source != "none" else {"value": 1.0, "source": "none"},
        "seed": seed_used,
        "deterministic": deterministic,
        "device": str(CFG.device),
        "class_mapping": CFG.classes,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(CFG.logs_dir, exist_ok=True)
    artifacts = {"confusion_matrix": cm_path, "confusion_matrix_normalized": cmn_path}
    should_save_probs = save_probs or kind == "hybrid"
    if should_save_probs:
        probs_targets = [os.path.join(CFG.logs_dir, f"probs_{kind}_unified_{sid}.npz")]
        if kind == "hybrid":
            probs_targets.insert(0, os.path.join(CFG.logs_dir, f"hybrid_{sid}.npz"))
        try:
            for probs_path in probs_targets:
                np.savez_compressed(probs_path, test_logits=test_logits, y_true=y_true, temperature=T)
            artifacts["probabilities"] = probs_targets[0]
            print(f"[UnifiedEval] Saved probability artifacts -> {probs_targets[0]}")
        except Exception:
            pass

    report = {
        "mode": "unified_eval",
        "model_kind": kind,
        "session_id": sid,
        "provenance": provenance,
        "metrics": metrics,
        "accuracy_raw": acc_raw,
        "accuracy_calibrated": acc_cal,
        "curves": curves,
        "artifacts": artifacts,
    }
    report_targets = [os.path.join(CFG.logs_dir, f"unified_eval_{kind}_{sid}.json")]
    if kind == "hybrid":
        report_targets.insert(0, os.path.join(CFG.logs_dir, f"hybrid_{sid}.json"))
    for out_path in report_targets:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(f"[UnifiedEval] Saved {kind} report -> {report_targets[0]}")
    return report_targets[0]


# ==============================
# Main
# ==============================


def main():
    # CLI args for data overrides and quick verification
    parser = argparse.ArgumentParser(description="Integrated hybrid trainer (EfficientNetB0 + ResNet50 CBAM)")
    parser.add_argument("--data-root", type=str, default=None, help="Path to folder containing Training/ and Testing")
    parser.add_argument("--verify-data-only", action="store_true", help="Only verify dataset paths and counts, then exit")
    parser.add_argument("--separate-models", action="store_true", help="Train EfficientNet and ResNet50 separately and fuse post-hoc (legacy behavior)")
    parser.add_argument("--eval-hybrid-only", action="store_true", help="Evaluation-only: load saved Hybrid weights and compute metrics without training")
    parser.add_argument("--hyb-weights", type=str, default=None, help="Path to saved Hybrid weights (.pth) for eval-only mode")
    parser.add_argument("--eval-efficientnet-only", action="store_true", help="Evaluation-only: load EfficientNet weights and compute metrics without training")
    parser.add_argument("--eval-resnet50-only", action="store_true", help="Evaluation-only: load ResNet50 weights and compute metrics without training")
    parser.add_argument("--eval-resnet50-only-legacy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--efficientnetb0-weights", type=str, default=None, help="Path to EfficientNetB0 weights (.pth) for eval-only mode")
    parser.add_argument("--resnet50-weights", type=str, default=None, help="Path to ResNet50 weights (.pth) for eval-only mode")
    parser.add_argument("--resnet50-weights-legacy", type=str, default=None, help=argparse.SUPPRESS)
    # Unified evaluation flags
    parser.add_argument("--unified-eval", type=str, default=None, help="Unified evaluation for a model kind: efficientnet|resnet50|hybrid|all")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (overrides auto-discovery) for unified eval when single kind")
    parser.add_argument("--temp-source", type=str, default="auto", help="Temperature strategy: auto|recompute|none|report")
    parser.add_argument("--temp-report", type=str, default=None, help="Report JSON to pull temperature from when --temp-source=report")
    parser.add_argument("--no-save-probs", action="store_true", help="Do not save test logits & metadata to NPZ during unified eval")
    parser.add_argument("--export-curves", action="store_true", help="Export ROC/PR curves inside unified eval JSON")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic seeds during unified eval")
    parser.add_argument("--full-curves", action="store_true", help="Store full curve coordinate arrays (default: only AUC summaries)")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.data_root:
        CFG.data_root = args.data_root

    device = CFG.device
    os.makedirs(CFG.logs_dir, exist_ok=True)
    os.makedirs(CFG.models_dir, exist_ok=True)
    print(f"Device: {device}")

    # Data
    train_ds, val_ds, test_ds = build_datasets(CFG.classes, CFG.stratified_val_ratio)
    # Print which dataset paths got used (train/test bases for clarity)
    if isinstance(train_ds, Subset) and len(train_ds) > 0:
        try:
            sample_idx = train_ds.indices[0] if hasattr(train_ds, 'indices') else 0
            train_base_hint = os.path.dirname(os.path.dirname(train_ds.dataset.samples[sample_idx][0]))
            print(f"Using TRAIN dataset base: {train_base_hint}")
        except Exception:
            pass
    try:
        if hasattr(test_ds, 'samples') and len(getattr(test_ds, 'samples', [])):
            test_base_hint = os.path.dirname(os.path.dirname(test_ds.samples[0][0]))
            print(f"Using TEST dataset base:  {test_base_hint}")
    except Exception:
        pass
    # Verification-only mode: report counts and exit
    if args.verify_data_only:
        tr_count = len(train_ds)
        val_count = len(val_ds)
        te_count = len(test_ds)
        print(f"Train/Val/Test sizes: {tr_count}/{val_count}/{te_count}")
        # Show class names
        print("Classes:", ", ".join(CFG.classes))
        return
    bs = auto_batch_size(CFG.batch_size_auto)
    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    # Deprecated individual eval-only flags (redirect to unified)
    if args.eval_hybrid_only:
        print("[Deprecated] --eval-hybrid-only -> use --unified-eval hybrid instead (routing now).")
        unified_evaluate(
            kind="hybrid",
            checkpoint=args.hyb_weights,
            temp_source="auto",
            temp_report=None,
            save_probs=True,
            export_curves=False,
            deterministic=False,
            full_curves=False,
        )
        return
    if args.eval_efficientnet_only:
        print("[Deprecated] --eval-efficientnet-only -> use --unified-eval efficientnet instead (routing now).")
        unified_evaluate(
            kind="efficientnet",
            checkpoint=args.efficientnetb0_weights,
            temp_source="auto",
            temp_report=None,
            save_probs=True,
            export_curves=False,
            deterministic=False,
            full_curves=False,
        )
        return
    if args.eval_resnet50_only or args.eval_resnet50_only_legacy:
        print("[Deprecated] --eval-resnet50-only -> use --unified-eval resnet50 instead (routing now).")
        unified_evaluate(
            kind="resnet50",
            checkpoint=args.resnet50_weights or args.resnet50_weights_legacy,
            temp_source="auto",
            temp_report=None,
            save_probs=True,
            export_curves=False,
            deterministic=False,
            full_curves=False,
        )
        return

    # Unified eval explicit usage
    if args.unified_eval:
        kinds = []
        if args.unified_eval.lower() == "all":
            kinds = ["efficientnet", "resnet50", "hybrid"]
        else:
            kinds = [args.unified_eval.lower()]
        summary = {}
        for k in kinds:
            path = unified_evaluate(
                kind=k,
                checkpoint=args.checkpoint if len(kinds) == 1 else None,
                temp_source=args.temp_source,
                temp_report=args.temp_report,
                save_probs=not args.no_save_probs,
                export_curves=args.export_curves,
                deterministic=args.deterministic,
                full_curves=args.full_curves,
            )
            summary[k] = path
        if len(kinds) > 1:
            sid = datetime.now().strftime("%Y%m%d_%H%M%S")
            combo_path = os.path.join(CFG.logs_dir, f"unified_eval_summary_{sid}.json")
            with open(combo_path, "w", encoding="utf-8") as f:
                json.dump({"generated_reports": summary}, f, indent=2)
            print(f"[UnifiedEval] Summary -> {combo_path}")
        return

    if not args.separate_models:
        # Build backbones
        hybrid_logger = SimpleLogger(CFG.logs_dir, prefix="integrated_hybrid_e2e")
        hybrid_logger.log(f"Config: {asdict(CFG)}")
        efficientnetb0_model = CBAMEfficientNetB0(
            num_classes=len(CFG.classes),
            dropout=CFG.efficientnetb0_dropout,
            cbam_indices=CFG.efficientnetb0_cbam_indices,
            use_cbam=True,
            variant=CFG.efficientnetb0_variant,
        ).to(device)
        resnet50_model = CBAMResNet50Classifier(
            num_classes=len(CFG.classes),
            use_cbam=True,
            cbam_on=CFG.resnet50_cbam_on,
            dropout_head=CFG.resnet50_dropout_head,
        ).to(device)
        hybrid_model = HybridModel(efficientnetb0_model, resnet50_model).to(device)
        # Staged training: take a conservative blend of stages (use the slower of the two where they differ)
        # We'll reuse EfficientNet stages as default schedule for the hybrid
        hyb = train_staged(hybrid_model, train_loader, val_loader, device, CFG.efficientnetb0_stages, hybrid_logger, save_prefix="Hybrid")
        hybrid_model.load_state_dict(hyb["state_dict"])  # best-val snapshot

        # Calibration on validation split
        hyb_val_true, _, hyb_val_logits = run_eval(hybrid_model, DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw), device)
        hyb_T, hyb_nll = temperature_search(torch.from_numpy(hyb_val_logits), torch.from_numpy(hyb_val_true))
        hybrid_logger.log(f"Temperature (HybridE2E): T={hyb_T:.3f}, NLL={hyb_nll:.6f}")

        # Test evaluation
        hyb_test_true, _, hyb_test_logits = run_eval(hybrid_model, test_loader, device)
        y_true = hyb_test_true
        hyb_probs = torch.softmax(torch.from_numpy(hyb_test_logits) / hyb_T, dim=-1).numpy()
        hyb_pred = hyb_probs.argmax(1)
        hyb_acc = (hyb_pred == y_true).mean() * 100.0

        # Metrics
        num_classes = len(CFG.classes)
        hyb_metrics = compute_all_metrics(y_true, hyb_pred, hyb_probs, num_classes, CFG.classes)

        # Save CMs
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_confusion_matrix(
            np.array(hyb_metrics.get("confusion_matrix_raw")) if hyb_metrics.get("confusion_matrix_raw") is not None else None,
            CFG.classes,
            os.path.join(CFG.logs_dir, f"cm_hybrid_e2e_{sid}.png"),
            title="Hybrid E2E CM",
            fmt="d",
            cmap="Blues",
        )
        save_confusion_matrix(
            np.array(hyb_metrics.get("confusion_matrix_normalized")) if hyb_metrics.get("confusion_matrix_normalized") is not None else None,
            CFG.classes,
            os.path.join(CFG.logs_dir, f"cm_hybrid_e2e_norm_{sid}.png"),
            title="Hybrid E2E CM (Norm)",
            fmt=".2f",
            cmap="Greens",
        )

        # Report
        report = {
            "session_id": sid,
            "config": asdict(CFG),
            "mode": "hybrid_end2end_equal_logit",
            "temperature": hyb_T,
            "accuracy": hyb_acc,
            "metrics": {"hybrid_e2e": hyb_metrics},
            "figures": {
                "cm_hybrid_e2e": f"{CFG.logs_dir}/cm_hybrid_e2e_{sid}.png",
                "cm_hybrid_e2e_norm": f"{CFG.logs_dir}/cm_hybrid_e2e_norm_{sid}.png",
            },
        }
        out_path = os.path.join(CFG.logs_dir, f"hybrid_e2e_report_{sid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {out_path}")
        return

    # Separate models path: train separately then fuse post-hoc (legacy behavior)
    # EfficientNet
    efficientnetb0_logger = SimpleLogger(CFG.logs_dir, prefix="integrated_efficientnetb0")
    efficientnetb0_logger.log(f"Config: {asdict(CFG)}")
    efficientnetb0_model = CBAMEfficientNetB0(
        num_classes=len(CFG.classes),
        dropout=CFG.efficientnetb0_dropout,
        cbam_indices=CFG.efficientnetb0_cbam_indices,
        use_cbam=True,
        variant=CFG.efficientnetb0_variant,
    ).to(device)
    efficientnetb0_result = train_staged(efficientnetb0_model, train_loader, val_loader, device, CFG.efficientnetb0_stages, efficientnetb0_logger, save_prefix="EfficientNetB0")
    efficientnetb0_model.load_state_dict(efficientnetb0_result["state_dict"])

    # ResNet50 + CBAM
    resnet50_logger = SimpleLogger(CFG.logs_dir, prefix="integrated_resnet50")
    resnet50_model = CBAMResNet50Classifier(
        num_classes=len(CFG.classes),
        use_cbam=True,
        cbam_on=CFG.resnet50_cbam_on,
        dropout_head=CFG.resnet50_dropout_head,
    ).to(device)
    resnet50_result = train_staged(resnet50_model, train_loader, val_loader, device, CFG.resnet50_stages, resnet50_logger, save_prefix="ResNet50")
    resnet50_model.load_state_dict(resnet50_result["state_dict"])

    # Calibration (validation split)
    efficientnetb0_val_true, _, efficientnetb0_val_logits = run_eval(efficientnetb0_model, DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw), device)
    resnet50_val_true, _, resnet50_val_logits = run_eval(resnet50_model, DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw), device)
    efficientnetb0_temperature, efficientnetb0_nll = temperature_search(torch.from_numpy(efficientnetb0_val_logits), torch.from_numpy(efficientnetb0_val_true))
    resnet50_temperature, resnet50_nll = temperature_search(torch.from_numpy(resnet50_val_logits), torch.from_numpy(resnet50_val_true))
    efficientnetb0_logger.log(f"Temperature (EfficientNetB0): T={efficientnetb0_temperature:.3f}, NLL={efficientnetb0_nll:.6f}")
    resnet50_logger.log(f"Temperature (ResNet50): T={resnet50_temperature:.3f}, NLL={resnet50_nll:.6f}")

    # Test evaluation
    efficientnetb0_test_true, _, efficientnetb0_test_logits = run_eval(efficientnetb0_model, test_loader, device)
    resnet50_test_true, _, resnet50_test_logits = run_eval(resnet50_model, test_loader, device)
    assert (efficientnetb0_test_true == resnet50_test_true).all(), "Test set mismatch between models"
    y_true = efficientnetb0_test_true
    efficientnetb0_probs = torch.softmax(torch.from_numpy(efficientnetb0_test_logits) / efficientnetb0_temperature, dim=-1).numpy()
    resnet50_probs = torch.softmax(torch.from_numpy(resnet50_test_logits) / resnet50_temperature, dim=-1).numpy()
    efficientnetb0_pred = efficientnetb0_probs.argmax(1)
    resnet50_pred = resnet50_probs.argmax(1)
    efficientnetb0_acc = (efficientnetb0_pred == y_true).mean() * 100.0
    resnet50_acc = (resnet50_pred == y_true).mean() * 100.0

    # Hybrid (post-hoc)
    hybrid_weight_efficientnetb0, hybrid_weight_resnet50 = CFG.hybrid_weights
    hybrid_probs = hybrid_weight_efficientnetb0 * efficientnetb0_probs + hybrid_weight_resnet50 * resnet50_probs
    hybrid_pred = hybrid_probs.argmax(1)
    hybrid_acc = (hybrid_pred == y_true).mean() * 100.0

    # Metrics
    num_classes = len(CFG.classes)
    efficientnetb0_metrics = compute_all_metrics(y_true, efficientnetb0_pred, efficientnetb0_probs, num_classes, CFG.classes)
    resnet50_metrics = compute_all_metrics(y_true, resnet50_pred, resnet50_probs, num_classes, CFG.classes)
    hyb_metrics = compute_all_metrics(y_true, hybrid_pred, hybrid_probs, num_classes, CFG.classes)

    # Save CMs
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_confusion_matrix(
        np.array(efficientnetb0_metrics.get("confusion_matrix_raw")) if efficientnetb0_metrics.get("confusion_matrix_raw") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_efficientnetb0_{sid}.png"),
        title="EfficientNetB0 CM",
        fmt="d",
        cmap="Blues",
    )
    save_confusion_matrix(
        np.array(efficientnetb0_metrics.get("confusion_matrix_normalized")) if efficientnetb0_metrics.get("confusion_matrix_normalized") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_efficientnetb0_norm_{sid}.png"),
        title="EfficientNetB0 CM (Norm)",
        fmt=".2f",
        cmap="Greens",
    )
    save_confusion_matrix(
        np.array(resnet50_metrics.get("confusion_matrix_raw")) if resnet50_metrics.get("confusion_matrix_raw") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_resnet50_{sid}.png"),
        title="ResNet50 CM",
        fmt="d",
        cmap="Blues",
    )
    save_confusion_matrix(
        np.array(resnet50_metrics.get("confusion_matrix_normalized")) if resnet50_metrics.get("confusion_matrix_normalized") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_resnet50_norm_{sid}.png"),
        title="ResNet50 CM (Norm)",
        fmt=".2f",
        cmap="Greens",
    )
    save_confusion_matrix(
        np.array(hyb_metrics.get("confusion_matrix_raw")) if hyb_metrics.get("confusion_matrix_raw") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_hybrid_{sid}.png"),
        title="Hybrid CM",
        fmt="d",
        cmap="Blues",
    )
    save_confusion_matrix(
        np.array(hyb_metrics.get("confusion_matrix_normalized")) if hyb_metrics.get("confusion_matrix_normalized") is not None else None,
        CFG.classes,
        os.path.join(CFG.logs_dir, f"cm_hybrid_norm_{sid}.png"),
        title="Hybrid CM (Norm)",
        fmt=".2f",
        cmap="Greens",
    )

    # Report
    report = {
        "session_id": sid,
        "config": asdict(CFG),
        "temperatures": {"efficientnetb0": efficientnetb0_temperature, "resnet50": resnet50_temperature},
        "accuracies": {"efficientnetb0": efficientnetb0_acc, "resnet50": resnet50_acc, "hybrid": hybrid_acc},
        "metrics": {"efficientnetb0": efficientnetb0_metrics, "resnet50": resnet50_metrics, "hybrid": hyb_metrics},
        "figures": {
            "cm_efficientnetb0": f"{CFG.logs_dir}/cm_efficientnetb0_{sid}.png",
            "cm_efficientnetb0_norm": f"{CFG.logs_dir}/cm_efficientnetb0_norm_{sid}.png",
            "cm_resnet50": f"{CFG.logs_dir}/cm_resnet50_{sid}.png",
            "cm_resnet50_norm": f"{CFG.logs_dir}/cm_resnet50_norm_{sid}.png",
            "cm_hybrid": f"{CFG.logs_dir}/cm_hybrid_{sid}.png",
            "cm_hybrid_norm": f"{CFG.logs_dir}/cm_hybrid_norm_{sid}.png",
        },
    }
    out_path = os.path.join(CFG.logs_dir, f"hybrid_integrated_report_{sid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    main()
