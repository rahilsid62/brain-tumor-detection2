"""ResNet50 classifier: ResNet50 backbone + CBAM + multi-scale fusion for 4-class tumor classification.

Provides a lightweight inference / fine-tuning architecture compatible with the
slim ResNet50 checkpoints you generated ( *_infer_fp16.pth ). If original
training checkpoints stored a dict with key 'model', you can load them with:

    import torch
    from training_scripts.cbam_resnet50 import ResNet50Classifier
    model = ResNet50Classifier(num_classes=4, use_cbam=True)
    state = torch.load('models/resnet50_full_infer_fp16.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

CBAM is inserted after selected residual layers (spatial & channel attention).
Multi-scale fusion uses pooled outputs from layer2, layer3, layer4 that are
channel-aligned via 1x1 convs then concatenated and projected to the classifier.

This file focuses on inference / light fine-tuning (no training loop here).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models
from typing import Sequence, Tuple

__all__ = [
    'ChannelAttention', 'SpatialAttention', 'CBAMBlock',
    'ResNet50Classifier'
]


class ChannelAttention(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        red = max(4, in_ch // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, red),
            nn.ReLU(inplace=True),
            nn.Linear(red, in_ch)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        w = self.sig(a + m).view(x.size(0), -1, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        cat = torch.cat([avg, mx], 1)
        w = self.sig(self.conv(cat))
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class ResNet50Classifier(nn.Module):
    """ResNet50 backbone with CBAM & multi-scale fusion.

    Multi-scale taps:
        - layer2 output ("mid")
        - layer3 output ("high")
        - layer4 output ("top")

    Each is globally pooled, channel-aligned (to embed_dim) and concatenated.
    Final head: Dropout + Linear -> num_classes.

    Args:
        num_classes: output classes (default 4)
        embed_dim: channel size after 1x1 alignment for each scale
        use_cbam: if True, inserts CBAM after chosen blocks
        cbam_layers: tuple specifying at which ResNet layer groups to apply CBAM;
            options drawn from {2,3,4} meaning after each of those group outputs
        dropout: dropout before classifier
        pretrained: use ImageNet pretrained weights for backbone
    """
    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 512,
        use_cbam: bool = True,
        cbam_layers: Tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract layers (keeping naming explicit for clarity)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 256 ch
        self.layer2 = backbone.layer2  # 512 ch
        self.layer3 = backbone.layer3  # 1024 ch
        self.layer4 = backbone.layer4  # 2048 ch

        self.use_cbam = use_cbam
        self.cbam_l2 = CBAMBlock(512) if use_cbam and 2 in cbam_layers else nn.Identity()
        self.cbam_l3 = CBAMBlock(1024) if use_cbam and 3 in cbam_layers else nn.Identity()
        self.cbam_l4 = CBAMBlock(2048) if use_cbam and 4 in cbam_layers else nn.Identity()

        # Channel alignment to embed_dim for each scale
        self.proj2 = nn.Conv2d(512, embed_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(1024, embed_dim, 1, bias=False)
        self.proj4 = nn.Conv2d(2048, embed_dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.bn4 = nn.BatchNorm2d(embed_dim)

        fused_dim = embed_dim * 3
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fused_dim, num_classes)

        # Init new layers (leave pretrained backbone intact)
        for m in [self.proj2, self.proj3, self.proj4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l2 = self.cbam_l2(l2)
        l3 = self.layer3(l2)
        l3 = self.cbam_l3(l3)
        l4 = self.layer4(l3)
        l4 = self.cbam_l4(l4)

        # Align & pool
        def pool_proj(t, proj, bn):
            t = proj(t)
            t = bn(t)
            t = torch.relu(t)
            t = torch.flatten(torch.adaptive_avg_pool2d(t, 1), 1)
            return t

        f2 = pool_proj(l2, self.proj2, self.bn2)
        f3 = pool_proj(l3, self.proj3, self.bn3)
        f4 = pool_proj(l4, self.proj4, self.bn4)

        fused = torch.cat([f2, f3, f4], dim=1)
        out = self.classifier(self.dropout(fused))
        return out

    def freeze_backbone(self, except_layers: Sequence[int] = (4,)):
        """Freeze backbone except specified stage indices (2/3/4)."""
        layer_map = {2: self.layer2, 3: self.layer3, 4: self.layer4}
        for i in (2, 3, 4):
            requires = i in except_layers
            for p in layer_map[i].parameters():
                p.requires_grad = requires


def build_resnet50_classifier(num_classes: int = 4, **kwargs) -> ResNet50Classifier:
    return ResNet50Classifier(num_classes=num_classes, **kwargs)


if __name__ == "__main__":  # simple sanity check
    model = ResNet50Classifier(num_classes=4, use_cbam=True).eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
    print("Output shape:", out.shape)
    assert out.shape == (2, 4)
    print("ResNet50 classifier sanity check passed.")

#!/usr/bin/env python3
"""
Integrated Advanced Training Pipeline
Model: ResNet50 backbone + CBAM + multi-scale head
Features (ported & extended from efficient.py):
  - Staged fine-tuning (Head / Partial / Full)
  - Mixup + CutMix (area-adjusted lambda)
  - Label smoothing (auto-disabled for mixed targets)
  - Warmup + Cosine LR (per-step scheduling)
  - Exponential Moving Average (EMA) of weights
  - (Optional) Gradient Checkpointing for ResNet layers
  - Stratified train/val split
  - Advanced logging (CSV + JSON report + confusion matrices)
  - Extensive metrics (confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, MCC, Kappa, ECE, Brier)
  - Test-Time Augmentation (flips + rotations)
  - Early stopping per stage + target accuracy short-circuit
  - Pause / Save / Resume (Ctrl+C interactive)
  - Reproducibility with optional deterministic mode
  - Graceful handling of non-finite losses

Directory assumptions:
  archive/Training/<class>/*.jpg
  archive/Testing/<class>/*.jpg

Usage:
    python cbam_resnet50.py

"""
import os
import time
import csv
import random
import warnings
import signal
import json
import math
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
    warnings.warn("scikit-learn not installed. Advanced metrics will be skipped.")

# ==============================
# Configuration
# ==============================

@dataclass
class Config:
    seed: int = 42
    deterministic: bool = False
    num_classes: int = 4
    classes: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    stages: List[Tuple[str, int, float, float, bool]] = (
        ("Head-Only", 3, 1e-3, 97.0, False),
        ("Partial",   6, 9e-4, 98.0, True),
        ("Full",     14, 7e-4, 99.0, True),
    )
    weight_decay: float = 1e-4
    # Stage-specific smoothing (head vs others)
    label_smoothing_head: float = 0.1
    label_smoothing_rest: float = 0.05
    mix_prob: float = 0.6
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    max_grad_norm: float = 0.6
    warmup_ratio: float = 0.2  # increased for smoother ramp
    cosine_min_lr_ratio: float = 0.1  # slightly higher floor to avoid over-decay
    ema_decay: float = 0.9995
    ema_warmup: int = 100
    use_ema: bool = True
    evaluate_both_ema: bool = True  # NEW: compare raw vs EMA at test
    use_cbam: bool = True
    cbam_on: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
    dropout_head: float = 0.5
    gradient_checkpointing: bool = False
    checkpoint_layers: Tuple[str, ...] = ("layer3", "layer4")
    stratified_val_ratio: float = 0.15
    batch_size_auto: bool = True
    early_stop_patience: int = 5
    tta_enabled: bool = True
    tta_transforms: int = 4
    tta_flip: bool = True
    pause_checkpoint_dir: str = "checkpoints"
    models_dir: str = "models"
    logs_dir: str = "training_logs"
    compute_full_metrics_on_val_every: int = 0
    bootstrap_ci: bool = False
    bootstrap_iterations: int = 1000
    ci_alpha: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    debug_nan: bool = True
    debug_max_batches: int = 0
    lr_epoch_mode: bool = True
    discriminative_lrs: bool = True
    lr_multipliers: Dict[str, float] = None
    debug_strict_nan: bool = True  # NEW: enable deep diagnostics
    nan_patience: int = 5          # NEW: how many consecutive non-finite batches before fallback
    stabilize_head_epoch1: bool = True  # disable mixup/cutmix + smoothing + lower LR on first epoch head stage
    head_epoch1_lr_scale: float = 0.3   # scale LR for first epoch head stabilization

    def __post_init__(self):
        if self.lr_multipliers is None:
            # Earlier layers smaller LR, head largest
            self.lr_multipliers = {
                'layer1': 0.25,
                'layer2': 0.5,
                'layer3': 0.75,
                'layer4': 1.0,
                'head': 1.5  # attention/scale/classifier
            }

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
    def __init__(self, out_dir: str, session_prefix: str = "cbam_resnet50"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{session_prefix}_session_{self.session_id}"
        self.main_log = os.path.join(self.out_dir, f"{self.session_name}_main.log")
        self.metrics_csv = os.path.join(self.out_dir, f"{self.session_name}_metrics.csv")
        self.final_report = os.path.join(self.out_dir, f"{self.session_name}_final_report.json")
        with open(self.metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
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
            csv.writer(f).writerow([
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
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
            from PIL import Image as PILImage
            img = PILImage.new('RGB', (224,224), (0,0,0))
        if self.transform:
            img = self.transform(img)
        return img, label

def build_datasets(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.2,0.2,0.15,0.05),
        transforms.GaussianBlur(3, sigma=(0.1,1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02,0.2), value='random'),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_sets, test_sets = [], []
    base = 'archive'
    if os.path.isdir(base):
        tr = os.path.join(base, "Training")
        te = os.path.join(base, "Testing")
        if os.path.isdir(tr):
            ds = BrainTumorDataset(tr, list(cfg.classes), train_tf)
            # Print distribution
            counts = [0]*cfg.num_classes
            for _, lbl in ds.samples:
                counts[lbl] += 1
            print(f"Class distribution ({base}/Training): {dict(zip(cfg.classes, counts))}")
            train_sets.append(ds)
        if os.path.isdir(te):
            test_sets.append(BrainTumorDataset(te, list(cfg.classes), test_tf))
    if not train_sets or not test_sets:
        raise RuntimeError("Datasets not found in 'archive'.")

    concat_train = ConcatDataset(train_sets)
    concat_test = ConcatDataset(test_sets)

    labels = []
    for ds in train_sets:
        labels.extend([lbl for _, lbl in ds.samples])
    labels = np.array(labels)
    indices = np.arange(len(labels))
    rng = np.random.default_rng(cfg.seed)
    val_indices, train_indices = [], []
    for cls_i in range(cfg.num_classes):
        cls_idx = indices[labels == cls_i]
        rng.shuffle(cls_idx)
        k = int(len(cls_idx) * cfg.stratified_val_ratio)
        val_indices.extend(cls_idx[:k])
        train_indices.extend(cls_idx[k:])
    train_subset = Subset(concat_train, train_indices)
    val_subset = Subset(concat_train, val_indices)
    
    print("Computing test class distribution...")
    from collections import Counter
    test_counter = Counter()
    # concat_test is ConcatDataset; iterate underlying datasets if possible
    for ds in test_sets:
        for _, lbl in ds.samples:
            test_counter[lbl] += 1
    if test_counter:
        mapped = {cfg.classes[k]: v for k, v in sorted(test_counter.items())}
        print(f"Test class counts: {mapped}")
    else:
        print("WARNING: No test samples counted.")

    return train_subset, val_subset, concat_test

# ==============================
# Helpers
# ==============================

def auto_batch_size():
    if not CFG.batch_size_auto:
        return 32
    if not torch.cuda.is_available():
        return 16
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if mem_gb >= 24: return 128
    if mem_gb >= 12: return 64
    if mem_gb >= 8:  return 32
    if mem_gb >= 4:  return 16
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
            nn.Linear(in_ch, max(1,in_ch//reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1,in_ch//reduction), in_ch)
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

class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that gracefully handles batch size 1 during training.
    If batch size == 1, falls back to using running statistics without updating them.
    This prevents ValueError: Expected more than 1 value per channel..."""
    def forward(self, input):  # type: ignore
        if self.training and input.size(0) == 1:
            # Temporarily switch to eval mode for this forward to use running stats
            was_training = self.training
            super().train(False)
            with torch.no_grad():
                out = super().forward(input)
            super().train(was_training)
            return out
        return super().forward(input)

# ==============================
# Model (ResNet50 + CBAM + Multi-Scale Head)
# ==============================

class CBAMResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, cbam_reduction=16, cbam_kernel=7,
                 use_cbam=True, cbam_on=("layer2","layer3","layer4"), dropout_head=0.5,
                 gradient_checkpointing=False, checkpoint_layers=("layer3","layer4")):
        super().__init__()
        self.use_cbam = use_cbam
        self.cbam_on = list(cbam_on)
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_layers = set(checkpoint_layers)
        self.dropout_head = dropout_head
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Extract layers
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
        if self.use_cbam:
            self.cbam_modules = nn.ModuleDict()
            for lname in self.cbam_on:
                ch = {'layer1':256,'layer2':512,'layer3':1024,'layer4':2048}.get(lname,None)
                if ch:
                    self.cbam_modules[lname] = CBAMBlock(ch)
        else:
            self.cbam_modules = nn.ModuleDict()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), SafeBatchNorm1d(1024), nn.Dropout(self.dropout_head),
            nn.Linear(1024, 512), nn.ReLU(), SafeBatchNorm1d(512), nn.Dropout(0.4)
        )
        self.scale1 = nn.Sequential(nn.Linear(512,256), nn.ReLU(), SafeBatchNorm1d(256), nn.Dropout(0.3))
        self.scale2 = nn.Sequential(nn.Linear(512,128), nn.ReLU(), SafeBatchNorm1d(128), nn.Dropout(0.25))
        self.scale3 = nn.Sequential(nn.Linear(512,64),  nn.ReLU(), SafeBatchNorm1d(64),  nn.Dropout(0.2))
        merged = 256+128+64
        self.classifier = nn.Sequential(
            nn.Linear(merged,512), nn.ReLU(), SafeBatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512,256), nn.ReLU(), SafeBatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256,128), nn.ReLU(), SafeBatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self._init_new()

    def _init_new(self):
        def init_module(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for mod in [self.cbam_modules, getattr(self,'scale1',None), getattr(self,'scale2',None), getattr(self,'scale3',None), getattr(self,'classifier',None)]:
            if mod is None: continue
            for m in mod.modules():
                if hasattr(m, 'weight') and not any(x in m.__class__.__name__.lower() for x in ['batchnorm']) and getattr(m, 'weight', None) is not None:
                    pass
                init_module(m)

    def _fwd_layer(self, layer_name, x):
        layer = getattr(self, layer_name)
        if self.gradient_checkpointing and layer_name in self.checkpoint_layers:
            def run(module, inp):
                return module(inp)
            x = torch.utils.checkpoint.checkpoint(run, layer, x)
        else:
            x = layer(x)
        if self.use_cbam and layer_name in self.cbam_modules:
            x = self.cbam_modules[layer_name](x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for lname in ["layer1","layer2","layer3","layer4"]:
            x = self._fwd_layer(lname, x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        att = self.attention(x)
        s1 = self.scale1(att)
        s2 = self.scale2(att)
        s3 = self.scale3(att)
        merged = torch.cat([s1,s2,s3], dim=1)
        return self.classifier(merged)

# ==============================
# Mixup & CutMix
# ==============================

def rand_bbox(size, lam):
    H = size[2]; W = size[3]
    cut_ratio = math.sqrt(1. - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    return x1,y1,x2,y2

def apply_mixup(x,y,alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1-lam) * x[idx]
    return mixed, y, y[idx], lam

def apply_cutmix(x,y,alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1,y1,x2,y2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    box_area = (x2-x1)*(y2-y1)
    lam_adj = 1. - box_area / (x.size(2)*x.size(3))
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
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        if self.num_updates < self.warmup:
            d = 1 - (1 - self.decay) * (self.num_updates / self.warmup)
        for n,p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1-d)
    def apply(self, model: nn.Module):
        self.backup = {}
        for n,p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
    def restore(self, model: nn.Module):
        if hasattr(self,'backup'):
            for n,p in model.named_parameters():
                if n in self.backup:
                    p.data.copy_(self.backup[n])
            self.backup = {}

# ==============================
# Warmup + Cosine Scheduler
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

class WarmupCosineEpoch:
    """Epoch-based Warmup + Cosine decay.
    Use when you want LR to change once per epoch instead of every step.
    """
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr_ratio=0.05, base_lr=None):
        self.opt = optimizer
        self.total_epochs = max(1, total_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] if base_lr is None else base_lr for pg in optimizer.param_groups]
        self.last_epoch = 0
    def step(self, epoch):
        self.last_epoch = epoch
        for i, pg in enumerate(self.opt.param_groups):
            base_lr = self.base_lrs[i]
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                lr = base_lr * epoch / self.warmup_epochs
            else:
                # progress through cosine after warmup
                progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                progress = min(1.0, max(0.0, progress))
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * cosine
            pg['lr'] = lr
    def get_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]

# Pause controller removed (interactive interrupts disabled for non-blocking training)
class DummyPauseController:
    def poll(self):
        return
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
    results = {}
    if not SKLEARN_AVAILABLE:
        return results
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    macro_f1 = f1.mean(); weighted_f1 = (f1 * support).sum() / support.sum(); balanced_acc = rec.mean()
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', labels=list(range(num_classes)))
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
    try: mcc = matthews_corrcoef(y_true, y_pred)
    except Exception: mcc = None
    try: kappa = cohen_kappa_score(y_true, y_pred)
    except Exception: kappa = None
    ece = expected_calibration_error(y_prob, y_true, n_bins=15)
    brier = brier_score(y_prob, y_true, num_classes)
    per_class = []
    for i,cname in enumerate(class_names):
        per_class.append({
            'class': cname,
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'pr_auc': None if pr_auc_per_class[i] is None else float(pr_auc_per_class[i])
        })
    results.update({
        'cm': cm, 'cm_norm': cm_norm,
        'confusion_matrix_raw': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'per_class': per_class,
        'macro_f1': float(macro_f1), 'weighted_f1': float(weighted_f1), 'balanced_accuracy': float(balanced_acc),
        'roc_auc_macro_ovr': None if roc_auc_macro is None else float(roc_auc_macro),
        'pr_auc_macro': None if pr_auc_macro is None else float(pr_auc_macro),
        'mcc': None if mcc is None else float(mcc),
        'cohen_kappa': None if kappa is None else float(kappa),
        'ece': ece, 'brier_score': brier
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
        head_param_names = [n for n,_ in self.model.named_parameters() if any(k in n for k in ["attention","scale","classifier","cbam_modules"])]
        if stage == "Head-Only":
            for n,p in self.model.named_parameters():
                p.requires_grad = (n in head_param_names)
            # Freeze backbone BN stats for head-only stability
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        elif stage == "Partial":
            # Unfreeze layer4 + head; re-enable BN training in unfrozen parts
            for n,p in self.model.named_parameters():
                if any(k in n for k in ["layer4","attention","scale","classifier","cbam_modules"]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            for m in self.model.layer4.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.train()
                    for param in m.parameters():
                        param.requires_grad = True
        else:  # Full
            for p in self.model.parameters():
                p.requires_grad = True
            # Reactivate all BN layers for fine-tuning
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.train()

    def evaluate(self, loader, criterion=None, collect_probs=False):
        self.model.eval()
        total_loss = 0.0; correct = 0; total = 0
        y_true=[]; y_pred=[]; y_prob=[]
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.cfg.device); target = target.to(self.cfg.device)
                with torch_amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = self.model(data)
                    loss = criterion(logits, target) if criterion is not None else torch.zeros(1, device=data.device)
                if torch.isfinite(loss): total_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds == target).sum().item(); total += target.size(0)
                if collect_probs:
                    y_true.append(target.cpu()); y_pred.append(preds.cpu()); y_prob.append(softmax_probs(logits).cpu())
        avg_loss = total_loss / max(1,len(loader)); acc = 100.0 * correct / max(1,total)
        if collect_probs:
            y_true = torch.cat(y_true).numpy(); y_pred = torch.cat(y_pred).numpy(); y_prob = torch.cat(y_prob).numpy()
            return avg_loss, acc, (y_true, y_pred, y_prob)
        return avg_loss, acc

    def train_stage(self, stage_name: str, epochs: int, base_lr: float,
                    target_acc: float, use_mixup_cutmix: bool):
        self.set_trainable(stage_name)
        smoothing = self.cfg.label_smoothing_head if stage_name == 'Head-Only' else self.cfg.label_smoothing_rest
        # Track consecutive non-finite batches for fallback
        consecutive_nonfinite = 0
        fallback_activated = False
        if self.cfg.discriminative_lrs:
            groups = {k: [] for k in self.cfg.lr_multipliers.keys()}
            for n,p in self.model.named_parameters():
                if not p.requires_grad: continue
                assigned = False
                for layer_key in ['layer1','layer2','layer3','layer4']:
                    if f"{layer_key}." in n:
                        groups[layer_key].append(p); assigned=True; break
                if not assigned and any(k in n for k in ['attention','scale','classifier','cbam_modules']):
                    groups['head'].append(p); assigned=True
                if not assigned:
                    groups['head'].append(p)
            param_groups = []
            for k, plist in groups.items():
                if not plist: continue
                mult = self.cfg.lr_multipliers.get(k,1.0)
                param_groups.append({'params': plist, 'lr': base_lr * mult})
            optimizer = optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)
            clip_params = []
            for g in optimizer.param_groups:
                clip_params.extend(g['params'])
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=base_lr, weight_decay=self.cfg.weight_decay)
            clip_params = params  # reuse for clipping
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        warmup_epochs = int(self.cfg.warmup_ratio * epochs)
        if self.cfg.lr_epoch_mode:
            scheduler = WarmupCosineEpoch(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs, min_lr_ratio=self.cfg.cosine_min_lr_ratio)
        else:
            scheduler = WarmupCosine(optimizer, total_steps, warmup_steps, min_lr_ratio=self.cfg.cosine_min_lr_ratio)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        ce_no_smooth = nn.CrossEntropyLoss(label_smoothing=0.0)
        stage_best = 0.0; stage_best_state=None; no_improve=0
        # After optimizer created (reuse clip_params), apply head stabilization if needed
        if stage_name == 'Head-Only' and self.cfg.stabilize_head_epoch1:
            original_group_lrs = [pg['lr'] for pg in optimizer.param_groups]
        for ep in range(1, epochs+1):
            pause_controller.poll()
            if stage_name == 'Head-Only' and self.cfg.stabilize_head_epoch1 and ep == 1:
                # Temporarily scale LR down
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = original_group_lrs[i] * self.cfg.head_epoch1_lr_scale
            elif stage_name == 'Head-Only' and self.cfg.stabilize_head_epoch1 and ep == 2:
                # Restore original LR from epoch 2 onward
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = original_group_lrs[i]
            pause_controller.wait_if_paused()
            if pause_controller.should_exit:
                ckpt_path = f"{self.cfg.pause_checkpoint_dir}/exit_ckpt_{stage_name}_ep{ep}.pth"
                save_checkpoint(ckpt_path, {"model": self.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep, "stage": stage_name})
                self.logger.log(f"Emergency checkpoint saved: {ckpt_path}")
                return False
            self.model.train(); epoch_loss=0.0; correct=0; total=0; t0=time.time(); non_finite_batches=0; batches_processed=0
            try:
                for batch_idx,(data,target) in enumerate(self.train_loader):
                    if self.cfg.debug_max_batches and batch_idx >= self.cfg.debug_max_batches: break
                    data = data.to(self.cfg.device); target = target.to(self.cfg.device)
                    optimizer.zero_grad(set_to_none=True)
                    stabilize = (stage_name == 'Head-Only' and self.cfg.stabilize_head_epoch1 and ep == 1)
                    current_use_mix = (use_mixup_cutmix and (random.random() < self.cfg.mix_prob) and not stabilize and not fallback_activated)
                    with torch_amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        if current_use_mix:
                            if random.random()<0.5:
                                dm, ya, yb, lam = apply_mixup(data, target, self.cfg.mixup_alpha)
                            else:
                                dm, ya, yb, lam = apply_cutmix(data, target, self.cfg.cutmix_alpha)
                            logits = self.model(dm)
                            loss = lam * ce_no_smooth(logits, ya) + (1-lam) * ce_no_smooth(logits, yb)
                        else:
                            logits = self.model(data)
                            loss = criterion(logits, target)
                    if not torch.isfinite(loss):
                        consecutive_nonfinite += 1
                        if self.cfg.debug_nan and consecutive_nonfinite <= self.cfg.nan_patience:
                            with torch.no_grad():
                                nan_logits = torch.isnan(logits).sum().item()
                                inf_logits = torch.isinf(logits).sum().item()
                                stats_msg = (f"[{stage_name}] Non-finite loss batch {batch_idx} | "
                                             f"logits nan={nan_logits} inf={inf_logits} min={float(torch.nan_to_num(logits).min()):.3e} "
                                             f"max={float(torch.nan_to_num(logits).max()):.3e} mean={float(torch.nan_to_num(logits).mean()):.3e} "
                                             f"data min={float(data.min()):.3e} max={float(data.max()):.3e}")
                                self.logger.log(stats_msg)
                                if self.cfg.debug_strict_nan:
                                    for g_i, pg in enumerate(optimizer.param_groups):
                                        tot_params = 0; nan_params = 0
                                        for p in pg['params']:
                                            if p.grad is not None:
                                                tot_params += p.numel(); nan_params += torch.isnan(p.data).sum().item()
                                        if nan_params:
                                            self.logger.log(f"ParamGroup {g_i} has {nan_params}/{tot_params} NaN params")
                        if consecutive_nonfinite >= self.cfg.nan_patience and not fallback_activated:
                            self.logger.log(f"[{stage_name}] Activating fallback: reducing LRs x0.1, disabling mixup/cutmix & smoothing for this epoch.")
                            for pg in optimizer.param_groups: pg['lr'] *= 0.1
                            criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
                            fallback_activated = True
                        continue
                    else:
                        consecutive_nonfinite = 0
                    self.scaler.scale(loss).backward()
                    if self.cfg.max_grad_norm:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(clip_params, self.cfg.max_grad_norm)
                    self.scaler.step(optimizer); self.scaler.update()
                    if not self.cfg.lr_epoch_mode: scheduler.step()
                    if self.ema: self.ema.update(self.model)
                    epoch_loss += loss.item(); preds = logits.argmax(1); correct += (preds==target).sum().item(); total += target.size(0); batches_processed += 1
            except KeyboardInterrupt:
                self.logger.log(f"[{stage_name}] KeyboardInterrupt caught. Marking pause.")
                pause_controller.interrupt_pending = True
                pause_controller.paused = True
            except RuntimeError as e:
                if 'DataLoader worker' in str(e):
                    self.logger.log(f"[{stage_name}] DataLoader workers exited after interrupt; ending epoch early.")
                else:
                    raise
            if total == 0 and epoch_loss == 0.0:
                self.logger.log(f"[{stage_name}] WARNING: Epoch {ep} produced no finite batches. Consider checking data pipeline.")
            train_loss = epoch_loss / max(1, batches_processed)
            train_acc = 100.0 * correct / max(1, total)
            val_loss, val_acc = self.evaluate(self.val_loader, criterion)
            gap = train_acc - val_acc; ep_time = time.time()-t0
            gpu_gb = (torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            if val_acc > stage_best:
                stage_best = val_acc; stage_best_state = {k:v.cpu() for k,v in self.model.state_dict().items()}; no_improve=0
                self.logger.log(f"[{stage_name}] 🎯 Stage new best: {val_acc:.2f}%")
            else:
                no_improve += 1
            if val_acc > self.global_best_acc:
                self.global_best_acc = val_acc; self.global_best_state = {k:v.cpu() for k,v in self.model.state_dict().items()}; self.global_best_epoch = ep; self.global_best_stage = stage_name
                self.logger.log(f"[GLOBAL] 🌍 New best validation accuracy: {val_acc:.2f}%")
            current_lr = scheduler.get_lr()[0]
            self.logger.log_epoch(ep, stage_name, train_loss, train_acc, val_loss, val_acc, current_lr, ep_time, gpu_gb, getattr(self.train_loader,'batch_size',0), gap, stage_best, self.global_best_acc)
            if gap > 15.0: self.logger.log(f"[{stage_name}] ⚠️ Train-val gap {gap:.2f}%")
            if val_acc >= target_acc and gap < 10.0:
                self.logger.log(f"[{stage_name}] ✅ Target {target_acc:.2f}% reached (Val={val_acc:.2f}%). Ending stage short-circuit.")
                break
            if no_improve >= self.cfg.early_stop_patience:
                self.logger.log(f"[{stage_name}] ⏹️ Early stopping (no improvement {no_improve} epochs).")
                break
        if stage_best_state:
            os.makedirs(self.cfg.models_dir, exist_ok=True)
            path = f"{self.cfg.models_dir}/model_stage_best_{stage_name}_{self.logger.session_id}.pth"
            torch.save(stage_best_state, path)
            self.logger.log(f"Saved stage best weights -> {path}")
        return True

    def final_test(self):
        # Evaluate current (global best loaded) weights first (raw)
        raw_collect = self.evaluate(self.test_loader, criterion=None, collect_probs=True)
        _, raw_acc, (raw_y_true, raw_y_pred, raw_y_prob) = raw_collect
        raw_metrics = compute_all_metrics(raw_y_true, raw_y_pred, raw_y_prob, self.cfg.num_classes, list(self.cfg.classes))
        ema_acc = None; ema_metrics = {}
        if self.ema and self.cfg.evaluate_both_ema:
            # Rebuild EMA shadow from current model if shadow length mismatches
            tracked = set(self.ema.shadow.keys())
            current = {n for n,_ in self.model.named_parameters() if _.requires_grad or True}
            if tracked != current:
                # Reset EMA shadow to current weights to avoid mismatched stale averages
                self.logger.log("[TEST] Reinitializing EMA shadow to current global best weights (parameter set changed).")
                self.ema = EMA(self.model, decay=self.cfg.ema_decay, warmup=self.cfg.ema_warmup)
            self.logger.log("Applying EMA weights for comparison.")
            self.ema.apply(self.model)
            ema_collect = self.evaluate(self.test_loader, criterion=None, collect_probs=True)
            _, ema_acc, (ema_y_true, ema_y_pred, ema_y_prob) = ema_collect
            ema_metrics = compute_all_metrics(ema_y_true, ema_y_pred, ema_y_prob, self.cfg.num_classes, list(self.cfg.classes))
            self.ema.restore(self.model)
            # Choose better
            if ema_acc >= raw_acc:
                self.logger.log(f"[TEST] Using EMA weights (EMA {ema_acc:.2f}% >= Raw {raw_acc:.2f}%).")
                return ema_acc, ema_metrics, ema_acc, ema_metrics
            else:
                self.logger.log(f"[TEST] Using raw weights (Raw {raw_acc:.2f}% > EMA {ema_acc:.2f}%).")
        return raw_acc, raw_metrics, ema_acc, ema_metrics

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
    bs = auto_batch_size(); logger.log(f"Batch size selected: {bs}")
    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    model = CBAMResNet50Classifier(num_classes=CFG.num_classes,
                                  use_cbam=CFG.use_cbam,
                                  cbam_on=CFG.cbam_on,
                                  dropout_head=CFG.dropout_head,
                                  gradient_checkpointing=CFG.gradient_checkpointing,
                                  checkpoint_layers=CFG.checkpoint_layers)

    # Resume logic
    latest_ckpt=None; ckpt_dir=CFG.pause_checkpoint_dir
    if os.path.isdir(ckpt_dir):
        ckpt_files=[f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir,x)), reverse=True)
            latest_ckpt=os.path.join(ckpt_dir, ckpt_files[0])
    start_stage_idx=0; start_epoch=1
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint=torch.load(latest_ckpt, map_location=CFG.device)
        model.load_state_dict(checkpoint['model'])
        stage_name = checkpoint.get('stage', CFG.stages[0][0])
        start_epoch = checkpoint.get('epoch',1)
        for idx,(sname,_,_,_,_) in enumerate(CFG.stages):
            if sname == stage_name:
                start_stage_idx = idx; break

    trainer = StageTrainer(model, CFG, logger, train_loader, val_loader, test_loader)

    # Run stages (call once per stage for efficiency)
    for stage_idx,(stage_name, epochs, lr, target, mix_flag) in enumerate(CFG.stages):
        if stage_idx < start_stage_idx: continue
        ep_start = start_epoch if stage_idx == start_stage_idx else 1
        # Adjust epochs if resuming mid-stage
        remaining = (epochs - ep_start + 1)
        cont = trainer.train_stage(stage_name, remaining, lr, target, mix_flag)
        if not cont:
            logger.log("Training interrupted before all stages completed.")
            break
        start_epoch = 1

    if trainer.global_best_state:
        logger.log(f"Reloading global best weights ({trainer.global_best_acc:.2f}% from stage {trainer.global_best_stage}).")
        trainer.model.load_state_dict(trainer.global_best_state)

    test_acc, test_metrics, tta_acc, tta_metrics = trainer.final_test()  # TTA values ignored below

    # Save confusion matrices
    import matplotlib.pyplot as plt
    import seaborn as sns
    def save_cm(cm, class_names, out_path, title, fmt):
        plt.figure(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues" if fmt=='d' else 'Greens', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()
    cm = test_metrics.get('cm'); cm_norm = test_metrics.get('cm_norm'); class_names=list(CFG.classes); logs_dir=CFG.logs_dir; sid=logger.session_id
    cm_path = None
    cm_norm_path = None
    if cm is not None:
        p=os.path.join(logs_dir,f"cm_{sid}.png"); save_cm(cm,class_names,p,"Confusion Matrix","d"); logger.log(f"Confusion matrix saved: {p}")
        cm_path = p
    if cm_norm is not None:
        pn=os.path.join(logs_dir,f"cm_norm_{sid}.png"); save_cm(cm_norm,class_names,pn,"Normalized Confusion Matrix",".2f"); logger.log(f"Normalized confusion matrix saved: {pn}")
        cm_norm_path = pn

    # Export unified-style artifacts used by plotting scripts.
    _, _, (y_true_u, _y_pred_u, y_prob_u) = trainer.evaluate(test_loader, criterion=None, collect_probs=True)
    test_logits_u = np.log(np.clip(y_prob_u, 1e-12, 1.0)).astype(np.float32)
    probs_targets = [
        os.path.join(logs_dir, f"resnet50_{sid}.npz"),
        os.path.join(logs_dir, f"probs_resnet50_unified_{sid}.npz"),
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
        "model_kind": "resnet50",
        "session_id": sid,
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
        os.path.join(logs_dir, f"resnet50_{sid}.json"),
        os.path.join(logs_dir, f"unified_eval_resnet50_{sid}.json"),
    ]
    for unified_report_path in unified_report_targets:
        with open(unified_report_path, "w", encoding="utf-8") as f:
            json.dump(unified_report, f, indent=2)
    logger.log(f"Unified report saved: {unified_report_targets[0]}")

    final_report = {
        'session_id': logger.session_id,
        'global_best_val_acc': trainer.global_best_acc,
        'global_best_stage': trainer.global_best_stage,
        'test_accuracy_no_tta': test_acc,
        'test_metrics_no_tta': test_metrics,
        # TTA fields removed intentionally
        'config': asdict(CFG),
        'sklearn_metrics_available': SKLEARN_AVAILABLE
    }
    import numpy as np
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k,v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    final_report = convert(final_report)
    logger.finalize(final_report)
    logger.log("="*100)
    logger.log(f"FINAL: TestAcc (no TTA): {test_acc:.2f}% (TTA omitted)")
    logger.log("Detailed metrics saved in final_report.json")
    logger.log("="*100)

if __name__ == "__main__":
    main()
