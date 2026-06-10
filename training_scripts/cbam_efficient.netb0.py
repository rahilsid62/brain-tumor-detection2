#!/usr/bin/env python3
"""
Full Featured Staged Training Pipeline for Brain Tumor Classification
Model: EfficientNet-B0 + Optional CBAM
Features Implemented:
  - Staged fine-tuning (Head / Partial / Full)
  - Mixup + CutMix (with correct area-adjusted lambda)
  - Label smoothing (auto-disabled for mixed targets)
  - Warmup + Cosine LR (per-step scheduling)
  - Exponential Moving Average (EMA) of weights
  - Gradient Checkpointing (optional) to reduce memory
  - Stratified train/val split (image-level)
  - Advanced logging (CSV + JSON final report)
  - Per-class & aggregate metrics:
        * Confusion Matrix (raw + normalized)
        * Precision / Recall / F1 (per-class, macro, weighted)
        * Balanced Accuracy
        * ROC-AUC (OvR macro & per-class)
        * PR-AUC (macro & per-class)
        * MCC, Cohen's Kappa
        * Calibration: ECE, Brier Score
  - Test-Time Augmentation (configurable)
  - Early stopping per stage + target accuracy short-circuit
  - Pause / Save / Resume (Ctrl+C interactive); (basic mid-epoch safety checkpoint)
  - Reproducibility seed setup (configurable deterministic option)
  - Graceful handling of non-finite losses
  - GPU memory usage logging

Requirements (install if missing):
  pip install scikit-learn

NOTE:
  This script assumes directory structure:
      archive/Training/<class>/*.jpg
      archive/Testing/<class>/*.jpg
    (and optionally 'archive (2)' variant)
"""

import os
import time
import csv
import random
import warnings
import signal
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp as torch_amp
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from torchvision import models, transforms

try:
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score, average_precision_score,
                                 matthews_corrcoef, cohen_kappa_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Metrics beyond accuracy will be skipped.")

# ==============================
# Configuration
# ==============================

@dataclass
class Config:
    seed: int = 42
    deterministic: bool = False
    num_classes: int = 4
    classes: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    efficientnet_variant: str = "b0"  # Options: "b0", "b1", "b2"
    use_kfold: bool = False
    kfold_splits: int = 5

    # Training stages
    stages: List[Tuple[str, int, float, float, bool]] = (
        ("Head-Only", 3, 2e-3, 97.0, False),
        ("Partial",   6, 1e-3, 98.0, False),
        ("Full",     14, 7e-4, 99.0, True),
    )

    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    mix_prob: float = 0.6
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    max_grad_norm: float = 0.5

    warmup_ratio: float = 0.08  # fraction of steps in a stage for warmup
    cosine_min_lr_ratio: float = 0.05  # final_lr = base_lr * this ratio

    ema_decay: float = 0.9995
    ema_warmup: int = 100  # steps to ramp EMA
    use_ema: bool = True

    use_cbam: bool = True
    cbam_indices: Tuple[int, ...] = (2, 4, 6)
    dropout: float = 0.5

    gradient_checkpointing: bool = False
    checkpoint_block_group: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2),
        (3, 4),
        (5, 6, 7),
    )  # groups of feature indices to checkpoint

    stratified_val_ratio: float = 0.15
    batch_size_auto: bool = True

    early_stop_patience: int = 5

    tta_enabled: bool = True
    tta_transforms: int = 4  # number of augment variants + the original
    tta_flip: bool = True

    pause_checkpoint_dir: str = "checkpoints"
    models_dir: str = "models"
    logs_dir: str = "training_logs"

    compute_full_metrics_on_val_every: int = 0  # 0 = only final test

    bootstrap_ci: bool = False
    bootstrap_iterations: int = 1000
    ci_alpha: float = 0.05  # 95% CI

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()

# ==============================
# Reproducibility
# ==============================

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

class ResearchLogger:
    def __init__(self, out_dir: str, session_prefix: str = "efficientnet_full"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{session_prefix}_session_{self.session_id}"
        self.main_log = os.path.join(self.out_dir, f"{self.session_name}_main.log")
        self.metrics_csv = os.path.join(self.out_dir, f"{self.session_name}_metrics.csv")
        self.final_report = os.path.join(self.out_dir, f"{self.session_name}_final_report.json")

        with open(self.metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch","stage","train_loss","train_acc","val_loss","val_acc",
                "lr","epoch_time_s","gpu_mem_gb","batch_size","train_val_gap",
                "stage_best_acc","global_best_acc"
            ])
        self.log("="*100)
        self.log(f"SESSION START: {self.session_name}")
        self.log("="*100)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.main_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_epoch(self, epoch, stage, train_loss, train_acc, val_loss, val_acc,
                  lr, epoch_time, gpu_mem_gb, bs, gap, stage_best, global_best):
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, stage, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_loss:.6f}", f"{val_acc:.2f}",
                f"{lr:.6e}", f"{epoch_time:.2f}", f"{gpu_mem_gb:.3f}", bs, f"{gap:.2f}",
                f"{stage_best:.2f}", f"{global_best:.2f}"
            ])
        self.log(
            f"[{stage}] Ep{epoch:03d} TL={train_loss:.4f} TA={train_acc:.2f}% | "
            f"VL={val_loss:.4f} VA={val_acc:.2f}% | LR={lr:.2e} Time={epoch_time:.1f}s "
            f"GPU={gpu_mem_gb:.2f}GB Gap={gap:.2f}% (StageBest={stage_best:.2f} / GlobalBest={global_best:.2f})"
        )

    def finalize(self, report: Dict):
        with open(self.final_report, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"Final report saved -> {self.final_report}")

# ==============================
# Dataset
# ==============================

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for c in classes:
            cdir = os.path.join(root_dir, c)
            if os.path.isdir(cdir):
                for name in os.listdir(cdir):
                    if name.lower().endswith((".jpg",".jpeg",".png")):
                        self.samples.append((os.path.join(cdir, name), self.class_to_idx[c]))
        if not self.samples:
            warnings.warn(f"No images found in {root_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
            img = Image.new('RGB', (224,224), (0,0,0))
        if self.transform:
            img = self.transform(img)
        return img, label

def build_datasets(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3,0.3,0.2,0.1),
        transforms.GaussianBlur(3, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.01,0.3), value='random'),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_sets, test_sets = [], []
    # Print class distribution for balancing
    print("Class distribution in training sets:")
    base = 'archive'
    total_train_images = 0
    if os.path.isdir(base):
        tr = os.path.join(base, "Training")
        te = os.path.join(base, "Testing")
        if os.path.isdir(tr):
            ds = BrainTumorDataset(tr, list(cfg.classes), train_tf)
            train_sets.append(ds)
            # Print class counts
            counts = [0]*cfg.num_classes
            for _, lbl in ds.samples:
                counts[lbl] += 1
            total_train_images = sum(counts)
            print(f"  {base}/Training: {dict(zip(cfg.classes, counts))}")
            print(f"Total training images: {total_train_images}")
        if os.path.isdir(te):
            test_sets.append(BrainTumorDataset(te, list(cfg.classes), test_tf))
    if not train_sets or not test_sets:
        raise RuntimeError("No training/testing datasets found in 'archive' or 'archive (2)'.")
    concat_train = ConcatDataset(train_sets)
    concat_test = ConcatDataset(test_sets)

    # Stratified split on image-level labels
    labels = []
    for ds in train_sets:
        labels.extend([lbl for _, lbl in ds.samples])
    labels = np.array(labels)
    indices = np.arange(len(labels))
    rng = np.random.default_rng(cfg.seed)
    val_indices = []
    train_indices = []
    for cls_i in range(cfg.num_classes):
        cls_idx = indices[labels == cls_i]
        rng.shuffle(cls_idx)
        k = int(len(cls_idx) * cfg.stratified_val_ratio)
        val_indices.extend(cls_idx[:k])
        train_indices.extend(cls_idx[k:])
    train_subset = Subset(concat_train, train_indices)
    val_subset = Subset(concat_train, val_indices)
    return train_subset, val_subset, concat_test

# ==============================
# Device helpers
# ==============================

def auto_batch_size():
    if not CFG.batch_size_auto:
        return 32
    if not torch.cuda.is_available():
        return 16
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
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
# CBAM
# ==============================

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, in_ch//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//reduction, in_ch)
        )
        self.sig = nn.Sigmoid()
    def forward(self,x):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        w = self.sig(a+m).view(x.size(0), -1, 1, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,k,padding=k//2,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        avg = x.mean(1, keepdim=True)
        mx,_ = x.max(1, keepdim=True)
        cat = torch.cat([avg,mx],1)
        w = self.sig(self.conv(cat))
        return x * w

class CBAMBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ca=ChannelAttention(c)
        self.sa=SpatialAttention()
    def forward(self,x):
        return self.sa(self.ca(x))

# ==============================
# Model
# ==============================

class CBAMEfficientNetB0(nn.Module):

    def __init__(self, num_classes=4, dropout=0.5, cbam_indices=(2,4,6), use_cbam=True,
                 gradient_checkpointing=False, checkpoint_groups=(), variant="b0"):
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
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_groups = checkpoint_groups

        # Dynamically determine output channels for each block
        block_channels = {}
        for idx, block in enumerate(self.features):
            # Each block is a Sequential, get out_channels from the last Conv2d in the block
            out_ch = None
            for layer in reversed(block):
                if hasattr(layer, 'out_channels'):
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

    def forward_features_standard(self, x):
        for i, blk in enumerate(self.features):
            x = blk(x)
            if self.use_cbam and f"cbam{i}" in self.cbam_layers:
                x = self.cbam_layers[f"cbam{i}"](x)
        return x

    def forward_features_checkpointed(self, x):
        # Group certain block indices into checkpoint calls
        module_list = list(self.features)
        used = set()
        for group in self.checkpoint_groups:
            def run_group(*inputs):
                z = inputs[0]
                for gi in group:
                    blk = module_list[gi]
                    z = blk(z)
                    if self.use_cbam and f"cbam{gi}" in self.cbam_layers:
                        z = self.cbam_layers[f"cbam{gi}"](z)
                return z
            x = torch.utils.checkpoint.checkpoint(run_group, x)
            used.update(group)
        # Remaining blocks
        for i, blk in enumerate(module_list):
            if i in used:
                continue
            x = blk(x)
            if self.use_cbam and f"cbam{i}" in self.cbam_layers:
                x = self.cbam_layers[f"cbam{i}"](x)
        return x

    def forward(self, x):
        if self.gradient_checkpointing:
            x = self.forward_features_checkpointed(x)
        else:
            x = self.forward_features_standard(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# ==============================
# Mixup & CutMix
# ==============================

def rand_bbox(size, lam):
    # size: (B,C,H,W)
    H = size[2]
    W = size[3]
    cut_ratio = math.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W)
    y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W)
    y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def apply_mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

def apply_cutmix(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # Adjust lambda based on the actual area
    box_area = (x2 - x1) * (y2 - y1)
    lam_adj = 1. - box_area / (x.size(2) * x.size(3))
    return x_cut, y, y[idx], lam_adj

# ==============================
# EMA
# ==============================

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
            # ramp up
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
        for n, p in model.named_parameters():
            if hasattr(self, 'backup') and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# ==============================
# Learning Rate Scheduler (Warmup + Cosine)
# ==============================

class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr_ratio=0.05, base_lr=None):
        self.opt = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
        self.base_lrs = [pg['lr'] if base_lr is None else base_lr for pg in optimizer.param_groups]

    def step(self):
        self.current_step += 1
        for i, pg in enumerate(self.opt.param_groups):
            base_lr = self.base_lrs[i]
            if self.current_step <= self.warmup_steps:
                lr = base_lr * self.current_step / max(1, self.warmup_steps)
            else:
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * cosine
            pg['lr'] = lr

    def get_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]

# Removed interactive PauseController to eliminate console prompt feature
class DummyPauseController:
    def wait_if_paused(self):
        return
    @property
    def should_exit(self):
        return False

pause_controller = DummyPauseController()

def save_checkpoint(path: str, state: Dict):
    torch.save(state, path)

# ==============================
# Metrics & Calibration
# ==============================

def softmax_probs(logits: torch.Tensor):
    return torch.softmax(logits, dim=1)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=15):
    # probs shape: (N, C), labels: (N,)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum()/len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return float(np.mean(np.sum((probs - one_hot)**2, axis=1)))

def compute_all_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    """
    Returns dict of metrics. y_prob shape: (N,C).
    """
    results = {}
    if not SKLEARN_AVAILABLE:
        return results

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    results["cm"] = cm
    results["cm_norm"] = cm_norm

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum()
    balanced_acc = rec.mean()

    try:
        roc_auc_per = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=list(range(num_classes)))
        roc_auc_macro = roc_auc_per  # sklearn returns macro by default for multiclass
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

    ece = expected_calibration_error(y_prob, y_true, n_bins=15)
    brier = brier_score(y_prob, y_true, num_classes)

    per_class = []
    for i, cname in enumerate(class_names):
        per_class.append({
            "class": cname,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "pr_auc": None if pr_auc_per_class[i] is None else float(pr_auc_per_class[i])
        })

    results.update({
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class": per_class,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc_macro_ovr": None if roc_auc_macro is None else float(roc_auc_macro),
        "pr_auc_macro": None if pr_auc_macro is None else float(pr_auc_macro),
        "mcc": None if mcc is None else float(mcc),
        "cohen_kappa": None if kappa is None else float(kappa),
        "ece": ece,
        "brier_score": brier,
    })
    return results

# ==============================
# Trainer
# ==============================

class StageTrainer:
    def __init__(self, model: nn.Module, cfg: Config, logger: ResearchLogger,
                 train_loader, val_loader, test_loader):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = torch_amp.GradScaler(enabled=torch.cuda.is_available())
        self.global_best_acc = 0.0
        self.global_best_state = None
        self.global_best_epoch = None
        self.global_best_stage = None
        self.ema = EMA(self.model, decay=cfg.ema_decay, warmup=cfg.ema_warmup) if cfg.use_ema else None

    def set_trainable(self, stage: str):
        if stage == "Head-Only":
            for n,p in self.model.named_parameters():
                p.requires_grad = ("classifier" in n)
        elif stage == "Partial":
            # Unfreeze last few feature indices and classifier + CBAM
            for n,p in self.model.named_parameters():
                if any(f"features.{i}" in n for i in [5,6,7]) or "classifier" in n or "cbam" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for p in self.model.parameters():
                p.requires_grad = True

    def evaluate(self, loader, criterion=None, collect_probs=False):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.cfg.device)
                target = target.to(self.cfg.device)
                with torch_amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = self.model(data)
                    if criterion is not None:
                        loss = criterion(logits, target)
                    else:
                        loss = torch.zeros(1, device=data.device)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds == target).sum().item()
                total += target.size(0)
                if collect_probs:
                    y_true.append(target.cpu())
                    y_pred.append(preds.cpu())
                    y_prob.append(softmax_probs(logits).cpu())
        avg_loss = total_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        if collect_probs:
            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            y_prob = torch.cat(y_prob).numpy()
            return avg_loss, acc, (y_true, y_pred, y_prob)
        return avg_loss, acc

    def train_stage(self, stage_name: str, epochs: int, base_lr: float,
                    target_acc: float, use_mixup_cutmix: bool):
        self.set_trainable(stage_name)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=base_lr, weight_decay=self.cfg.weight_decay)

        # Steps for this stage
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        scheduler = WarmupCosine(optimizer, total_steps, warmup_steps,
                                 min_lr_ratio=self.cfg.cosine_min_lr_ratio)

        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        ce_no_smooth = nn.CrossEntropyLoss(label_smoothing=0.0)

        stage_best = 0.0
        stage_best_state = None
        no_improve = 0

        for ep in range(1, epochs+1):
            pause_controller.wait_if_paused()
            if pause_controller.should_exit:
                ckpt_path = f"{self.cfg.pause_checkpoint_dir}/exit_ckpt_{stage_name}_ep{ep}.pth"
                save_checkpoint(ckpt_path, {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": ep,
                    "stage": stage_name
                })
                self.logger.log(f"Saved emergency checkpoint: {ckpt_path}. Exiting.")
                return False

            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            t0 = time.time()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.cfg.device, non_blocking=True)
                target = target.to(self.cfg.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                use_mix = use_mixup_cutmix and (random.random() < self.cfg.mix_prob)
                with torch_amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    if use_mix:
                        if random.random() < 0.5:
                            data_mix, ya, yb, lam = apply_mixup(data, target, self.cfg.mixup_alpha)
                        else:
                            data_mix, ya, yb, lam = apply_cutmix(data, target, self.cfg.cutmix_alpha)
                        logits = self.model(data_mix)
                        loss = lam * ce_no_smooth(logits, ya) + (1 - lam) * ce_no_smooth(logits, yb)
                    else:
                        logits = self.model(data)
                        loss = criterion(logits, target)

                if not torch.isfinite(loss):
                    continue
                self.scaler.scale(loss).backward()
                if self.cfg.max_grad_norm:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()

                if self.ema:
                    self.ema.update(self.model)

                # Scheduler per step
                scheduler.step()

                epoch_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds == target).sum().item()
                total += target.size(0)

            train_loss = epoch_loss / max(1, len(self.train_loader))
            train_acc = 100.0 * correct / max(1, total)

            val_loss, val_acc = self.evaluate(self.val_loader, criterion)

            gap = train_acc - val_acc
            ep_time = time.time() - t0
            gpu_gb = (torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if val_acc > stage_best:
                stage_best = val_acc
                stage_best_state = {k: v.cpu() for k,v in self.model.state_dict().items()}
                no_improve = 0
                self.logger.log(f"[{stage_name}] 🎯 Stage new best: {val_acc:.2f}%")
            else:
                no_improve += 1

            if val_acc > self.global_best_acc:
                self.global_best_acc = val_acc
                self.global_best_state = {k: v.cpu() for k,v in self.model.state_dict().items()}
                self.global_best_epoch = ep
                self.global_best_stage = stage_name
                self.logger.log(f"[GLOBAL] 🌍 New best validation accuracy: {val_acc:.2f}%")

            current_lr = scheduler.get_lr()[0]
            self.logger.log_epoch(ep, stage_name, train_loss, train_acc, val_loss, val_acc,
                                  current_lr, ep_time, gpu_gb,
                                  getattr(self.train_loader, 'batch_size', 0),
                                  gap, stage_best, self.global_best_acc)

            if gap > 15.0:
                self.logger.log(f"[{stage_name}] ⚠️ Large train-val gap {gap:.2f}% detected.")

            if val_acc >= target_acc and gap < 10.0:
                self.logger.log(f"[{stage_name}] ✅ Target {target_acc:.2f}% reached (Val={val_acc:.2f}%). Ending stage early.")
                break

            if no_improve >= self.cfg.early_stop_patience:
                self.logger.log(f"[{stage_name}] ⏹️ Early stopping (no improvement {no_improve} epochs).")
                break

        # Save stage best
        if stage_best_state:
            os.makedirs(self.cfg.models_dir, exist_ok=True)
            path = f"{self.cfg.models_dir}/model_stage_best_{stage_name}_{self.logger.session_id}.pth"
            torch.save(stage_best_state, path)
            self.logger.log(f"Saved stage best weights -> {path}")
        return True

    def final_test(self):
        # Evaluate with EMA weights (if available)
        if self.ema:
            self.logger.log("Applying EMA weights for final test evaluation.")
            self.ema.apply(self.model)

        collect = self.evaluate(self.test_loader, criterion=None, collect_probs=True)
        _, test_acc, (y_true, y_pred, y_prob) = collect
        metrics = compute_all_metrics(y_true, y_pred, y_prob,
                                      self.cfg.num_classes,
                                      list(self.cfg.classes))

        # TTA (optional augmentation inference)
        if self.cfg.tta_enabled:
            self.logger.log("Running TTA inference...")
            tta_preds = []
            tta_probs = []
            y_true_tta = []
            self.model.eval()
            with torch.no_grad():
                for data, target in self.test_loader:
                    data = data.to(self.cfg.device)
                    target = target.to(self.cfg.device)
                    # original
                    with torch_amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        logits_accum = self.model(data)
                        count = 1
                        # Horizontal flip
                        if self.cfg.tta_flip:
                            logits_accum += self.model(torch.flip(data, dims=[3]))
                            count += 1
                        # Additional light transforms
                        extra_needed = max(0, self.cfg.tta_transforms - count)
                        for _ in range(extra_needed):
                            # simple random 90-degree rotation
                            rot_k = random.randint(0,3)
                            rot = torch.rot90(data, k=rot_k, dims=[2,3])
                            logits_accum += self.model(rot)
                            count += 1
                        logits_mean = logits_accum / count
                    preds = logits_mean.argmax(1)
                    tta_preds.append(preds.cpu())
                    tta_probs.append(softmax_probs(logits_mean).cpu())
                    y_true_tta.append(target.cpu())
            y_true_tta = torch.cat(y_true_tta).numpy()
            y_pred_tta = torch.cat(tta_preds).numpy()
            y_prob_tta = torch.cat(tta_probs).numpy()
            tta_acc = 100.0 * (y_pred_tta == y_true_tta).sum() / len(y_true_tta)
            tta_metrics = compute_all_metrics(y_true_tta, y_pred_tta, y_prob_tta,
                                              self.cfg.num_classes, list(self.cfg.classes))
        else:
            tta_acc = None
            tta_metrics = {}

        # Remove EMA weights (restore original) for potential further use
        if self.ema:
            self.ema.restore(self.model)

        return test_acc, metrics, tta_acc, tta_metrics

# ==============================
# Main
# ==============================

def main():
    device = CFG.device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    else:
        print("Using CPU")

    logger = ResearchLogger(CFG.logs_dir)
    logger.log(f"Config: {asdict(CFG)}")

    train_ds, val_ds, test_ds = build_datasets(CFG)
    bs = auto_batch_size()
    logger.log(f"Batch size selected: {bs}")
    nw = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw,
                             pin_memory=torch.cuda.is_available())

    model = CBAMEfficientNetB0(num_classes=CFG.num_classes,
                               dropout=CFG.dropout,
                               cbam_indices=CFG.cbam_indices,
                               use_cbam=CFG.use_cbam,
                               gradient_checkpointing=CFG.gradient_checkpointing,
                               checkpoint_groups=CFG.checkpoint_block_group,
                               variant=CFG.efficientnet_variant)


    # --- Checkpoint Resume Logic ---
    latest_ckpt = None
    ckpt_dir = CFG.pause_checkpoint_dir
    if os.path.isdir(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
            latest_ckpt = os.path.join(ckpt_dir, ckpt_files[0])

    start_stage_idx = 0
    start_epoch = 1
    optimizer = None
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=CFG.device)
        model.load_state_dict(checkpoint["model"])
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG.stages[0][2], weight_decay=CFG.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 1)
        stage_name = checkpoint.get("stage", CFG.stages[0][0])
        # Find stage index
        for idx, (sname, _, _, _, _) in enumerate(CFG.stages):
            if sname == stage_name:
                start_stage_idx = idx
                break

    trainer = StageTrainer(model, CFG, logger, train_loader, val_loader, test_loader)

    # Run Stages
    for stage_idx, (stage_name, epochs, lr, target, mix_flag) in enumerate(CFG.stages):
        if stage_idx < start_stage_idx:
            continue
        ep_start = start_epoch if stage_idx == start_stage_idx else 1
        for ep in range(ep_start, epochs+1):
            cont = trainer.train_stage(stage_name, 1, lr, target, mix_flag)
            if not cont:
                logger.log("Training interrupted before completing all stages.")
                break
        start_epoch = 1  # Only use resume epoch for first resumed stage

    # Load global best weights for final test
    if trainer.global_best_state:
        logger.log(f"Reloading global best weights ({trainer.global_best_acc:.2f}% from stage {trainer.global_best_stage}).")
        trainer.model.load_state_dict(trainer.global_best_state)

    test_acc, test_metrics, tta_acc, tta_metrics = trainer.final_test()

    # Save confusion matrix as PNG
    def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def save_confusion_matrix_norm(cm_norm, class_names, out_path, title="Normalized Confusion Matrix"):
        plt.figure(figsize=(7,6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    cm = test_metrics.get("cm")
    cm_norm = test_metrics.get("cm_norm")
    class_names = list(CFG.classes)
    logs_dir = CFG.logs_dir
    session_id = logger.session_id
    cm_path = None
    cm_norm_path = None
    if cm is not None:
        out_path = os.path.join(logs_dir, f"cm_{session_id}.png")
        save_confusion_matrix(cm, class_names, out_path)
        logger.log(f"Confusion matrix saved as image: {out_path}")
        cm_path = out_path
    if cm_norm is not None:
        out_path_norm = os.path.join(logs_dir, f"cm_norm_{session_id}.png")
        save_confusion_matrix_norm(cm_norm, class_names, out_path_norm)
        logger.log(f"Normalized confusion matrix saved as image: {out_path_norm}")
        cm_norm_path = out_path_norm

    # Export unified-style artifacts used by plotting scripts.
    _, _, (y_true_u, _y_pred_u, y_prob_u) = trainer.evaluate(test_loader, criterion=None, collect_probs=True)
    test_logits_u = np.log(np.clip(y_prob_u, 1e-12, 1.0)).astype(np.float32)
    probs_targets = [
        os.path.join(logs_dir, f"efficientnet_{session_id}.npz"),
        os.path.join(logs_dir, f"probs_efficientnetb0_unified_{session_id}.npz"),
    ]
    for probs_path in probs_targets:
        np.savez_compressed(
            probs_path,
            test_logits=test_logits_u,
            y_true=y_true_u,
            temperature=np.float32(1.0)
        )
    logger.log(f"Unified probabilities saved: {probs_targets[0]}")

    scalar_metrics_raw = {}
    for k, v in test_metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            scalar_metrics_raw[k] = float(v)

    unified_artifacts = {"probabilities": probs_targets[0]}
    if cm_path is not None:
        unified_artifacts["confusion_matrix"] = cm_path
    if cm_norm_path is not None:
        unified_artifacts["confusion_matrix_normalized"] = cm_norm_path

    unified_report = {
        "mode": "unified_eval",
        "model_kind": "efficientnetb0",
        "session_id": session_id,
        "provenance": {
            "temperature": {"value": 1.0, "source": "identity"},
            "class_mapping": class_names,
            "source": "training_run"
        },
        "metrics": {
            "raw": scalar_metrics_raw,
            "calibrated": None
        },
        "accuracy_raw": float(test_acc),
        "accuracy_calibrated": None,
        "curves": {},
        "artifacts": unified_artifacts
    }
    unified_report_targets = [
        os.path.join(logs_dir, f"efficientnet_{session_id}.json"),
        os.path.join(logs_dir, f"unified_eval_efficientnetb0_{session_id}.json"),
    ]
    for unified_report_path in unified_report_targets:
        with open(unified_report_path, "w", encoding="utf-8") as f:
            json.dump(unified_report, f, indent=2)
    logger.log(f"Unified report saved: {unified_report_targets[0]}")

    final_report = {
        "session_id": logger.session_id,
        "global_best_val_acc": trainer.global_best_acc,
        "global_best_stage": trainer.global_stage_best if hasattr(trainer, 'global_stage_best') else trainer.global_best_stage,
        "test_accuracy_no_tta": test_acc,
        "test_metrics_no_tta": test_metrics,
        # TTA removed for consistency
        "config": asdict(CFG),
        "sklearn_metrics_available": SKLEARN_AVAILABLE
    }
    import numpy as np
    def convert_ndarrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarrays(v) for v in obj]
        return obj
    final_report = convert_ndarrays(final_report)
    logger.finalize(final_report)

    logger.log("="*100)
    logger.log(f"FINAL: TestAcc (no TTA): {test_acc:.2f}% (TTA omitted)")
    logger.log("Detailed metrics saved in final_report.json")
    logger.log("="*100)

if __name__ == "__main__":
    main()