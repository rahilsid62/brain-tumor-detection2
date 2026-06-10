#!/usr/bin/env python3
"""Plot macro PR and ROC comparison for EfficientNet, ResNet50, and Hybrid (E2E).

Two operation modes:

1. Legacy weights mode (original behavior):
     - Auto / explicit weight discovery
     - Builds models, calibrates on validation, computes curves from scratch.

2. Unified report mode (preferred for reproducibility):
     - Provide unified evaluation JSON paths produced by the new unified eval pipeline.
     - Script loads associated NPZ probability/logit files (stored in JSON["artifacts"]["probabilities"]).
     - Computes macro ROC / PR directly from stored logits, avoiding model rebuild.
     - Optionally apply calibration (temperature) by dividing logits by stored temperature.

Outputs (both modes):
    results/pr/combined_pr_compare_hybrid_vs_baselines.png
    results/roc/combined_roc_compare_hybrid_vs_baselines.png

Examples (Unified mode):
    python .\\paper\\plot_pr_roc_compare_hybrid.py \
         --eff-json training_logs\\unified_eval_efficientnet_20250925_164722.json \
         --tdn-json training_logs\\unified_eval_tumordetnet_20250925_171154.json \
         --hyb-json training_logs\\unified_eval_hybrid_20250925_170815.json \
         --calibrated

Example (Legacy weight mode):
    python .\\paper\\plot_pr_roc_compare_hybrid.py --hyb-weights models\\hybrid_full.pth

Note: If any --*-json flag is provided the script switches entirely to unified mode; weight args are ignored.
"""

from __future__ import annotations

import os
import glob
import argparse
import json as json_lib
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training_scripts.train_hybrid_integrated import (
    CBAMEfficientNetB0,
    CBAMResNet50Classifier,
    HybridModel,
    build_datasets,
    run_eval,
    temperature_search,
    CFG,
)
from torch.utils.data import DataLoader, Subset

# Fixed label and color scheme required for legends
# Labels: efficientnet (green), resnet (red), hybrid (blue)
FIXED_ORDER = ["efficientnet", "resnet", "hybrid"]
FIXED_COLORS = {
    "efficientnet": "tab:green",
    "resnet": "tab:red",
    "hybrid": "tab:blue",
}
DISPLAY_LABELS = {
    "efficientnet": "efficientnetb0",
    "resnet": "resnet50",
    "hybrid": "hybrid",
}

def canonical_label(name: str) -> str:
    n = (name or "").strip().lower()
    if n in ("efficientnet", "eff", "effnet", "efficientnet-b0"):
        return "efficientnet"
    if n in ("tumordetnet", "resnet", "cbamresnet", "resnet50", "cbam-resnet50"):
        return "resnet"
    if n in ("hybrid", "hybrid e2e", "hybrid_e2e"):
        return "hybrid"
    # default passthrough but lower-case to avoid mismatched keys
    return n


def auto_find(patterns: List[str]) -> Optional[str]:
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def ensure_dirs():
    os.makedirs(os.path.join("results", "pr"), exist_ok=True)
    os.makedirs(os.path.join("results", "roc"), exist_ok=True)


def probs_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    t = torch.from_numpy(logits) / float(T)
    return torch.softmax(t, dim=-1).numpy()


def macro_roc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int, n_points: int = 200):
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    return mean_fpr, mean_tpr, macro_auc


def macro_pr(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int, n_points: int = 200):
    mean_recall = np.linspace(0, 1, n_points)
    precisions = []
    aps = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, c])
        ap = average_precision_score(y_bin, y_prob[:, c])
        aps.append(ap)
        # Interpolate precision as function of recall (monotonic recall)
        # precision_recall_curve returns recall increasing; ensure unique for interp
        recall_unique, idx = np.unique(recall, return_index=True)
        precision_unique = precision[idx]
        p_interp = np.interp(mean_recall, recall_unique, precision_unique)
        precisions.append(p_interp)
    mean_precision = np.mean(precisions, axis=0) if precisions else np.full_like(mean_recall, np.nan)
    macro_ap = float(np.mean(aps)) if aps else float("nan")
    return mean_recall, mean_precision, macro_ap


def main():
    parser = argparse.ArgumentParser(description="Macro PR/ROC comparison: EfficientNet vs ResNet50 vs Hybrid E2E")
    # Unified report mode arguments
    parser.add_argument("--eff-json", type=str, default=None, help="Unified eval JSON for EfficientNet")
    parser.add_argument("--resnet-json", type=str, default=None, help="Unified eval JSON for ResNet50")
    parser.add_argument("--tdn-json", type=str, default=None, help="Legacy alias for --resnet-json")
    parser.add_argument("--hyb-json", type=str, default=None, help="Unified eval JSON for Hybrid")
    parser.add_argument("--calibrated", action="store_true", help="Use calibrated (temperature-scaled) probabilities from unified JSON NPZ (default: raw)")
    parser.add_argument("--label-raw", action="store_true", help="Force legend AUC/AP labels to use raw metrics from JSON even if --calibrated is set")
    # Legacy weight mode arguments (fallback if no JSONs provided)
    parser.add_argument("--data-root", type=str, default=None, help="Path to folder containing Training/ and Testing (legacy mode)")
    parser.add_argument("--eff-weights", type=str, default=None, help="Path to EfficientNet stage-best weights .pth (legacy)")
    parser.add_argument("--tdn-weights", type=str, default=None, help="Path to TumorDetNet stage-best weights .pth (legacy)")
    parser.add_argument("--hyb-weights", type=str, default=None, help="Path to Hybrid (E2E) stage-best weights .pth (legacy)")
    parser.add_argument("--hyb-report", type=str, default=None, help="Path to Hybrid eval-only report JSON to reuse temperature (legacy)")
    args = parser.parse_args()

    # Use repo default config but allow data-root override
    if args.data_root:
        CFG.data_root = args.data_root

    device = CFG.device

    if args.resnet_json is None:
        args.resnet_json = args.tdn_json

    unified_mode = any([args.eff_json, args.resnet_json, args.hyb_json])

    if unified_mode:
        # Unified JSON mode: load NPZ probability files referenced in unified eval reports.
        def load_probs_from_unified(json_path: str, calibrated: bool):
            with open(json_path, 'r', encoding='utf-8') as f:
                rep = json_lib.load(f)
            model_kind = rep.get(
                "model_kind",
                rep.get("provenance", {}).get(
                    "model_kind",
                    rep.get("unified_eval", {}).get("provenance", {}).get("model_kind", "model"),
                ),
            )
            artifacts = rep.get("artifacts")
            if not isinstance(artifacts, dict) or "probabilities" not in artifacts:
                ue = rep.get("unified_eval", {}) if isinstance(rep, dict) else {}
                artifacts = ue.get("artifacts", {}) if isinstance(ue, dict) else {}
            npz_path = artifacts.get("probabilities")
            if npz_path is None:
                raise SystemExit(f"Unified report {json_path} missing artifacts.probabilities (rerun eval with --save-probs)")
            # Resolve relative path relative to report directory
            if not os.path.isabs(npz_path):
                npz_path = os.path.join(os.path.dirname(json_path), os.path.basename(npz_path)) if os.path.exists(os.path.join(os.path.dirname(json_path), os.path.basename(npz_path))) else npz_path
            if not os.path.exists(npz_path):
                # Try relative to repo root
                alt = os.path.join(ROOT_DIR, npz_path)
                if os.path.exists(alt):
                    npz_path = alt
                else:
                    raise SystemExit(f"Probability NPZ not found: {npz_path}")
            data = np.load(npz_path)
            logits = data["test_logits"]
            y_true = data["y_true"]
            T = float(data.get("temperature", 1.0))
            if not calibrated:
                T_eff = 1.0
            else:
                T_eff = T
            probs = torch.softmax(torch.from_numpy(logits) / T_eff, dim=-1).numpy()
            metrics_raw = rep.get("metrics", {}).get("raw", {})
            metrics_cal = rep.get("metrics", {}).get("calibrated", {})
            return model_kind, y_true, probs, (metrics_raw, metrics_cal)
        loaded = []  # list of (canonical_name, y_true, probs, (metrics_raw, metrics_cal))
        jobs = [
            ("efficientnet", args.eff_json),
            ("resnet", args.resnet_json),
            ("hybrid", args.hyb_json),
        ]
        for expected_name, path in jobs:
            if not path:
                continue
            mk, y, p, metrics_pair = load_probs_from_unified(path, args.calibrated)
            can_name = canonical_label(mk)
            # Final report schema may omit model_kind; in that case trust the CLI slot.
            if can_name not in FIXED_ORDER:
                can_name = expected_name
            loaded.append((can_name, y, p, metrics_pair))
        # Ensure we have at least one
        if not loaded:
            raise SystemExit("No unified JSONs loaded.")
        # Verify y_true consistency
        base_y = loaded[0][1]
        for name, yv, _, _ in loaded[1:]:
            if len(yv) != len(base_y) or np.any(yv != base_y):
                raise SystemExit(f"Label mismatch between first model and {name}. Ensure unified evals used same test split.")
        y_true = base_y
        n_classes = loaded[0][2].shape[1]
        curves_roc: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        curves_pr: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        label_auc: Dict[str, float] = {}
        label_ap: Dict[str, float] = {}
        for name, _, probs, (metrics_raw, metrics_cal) in loaded:
            can = canonical_label(name)
            curves_roc[can] = macro_roc(y_true, probs, n_classes)
            curves_pr[can] = macro_pr(y_true, probs, n_classes)
            if args.label_raw:
                label_auc[can] = float(metrics_raw.get("roc_auc_macro_ovr", curves_roc[can][2]))
                label_ap[can] = float(metrics_raw.get("pr_auc_macro", curves_pr[can][2]))
            else:
                # Use calibrated if calibrated mode selected, else raw
                if args.calibrated:
                    label_auc[can] = float(metrics_cal.get("roc_auc_macro_ovr", curves_roc[can][2]))
                    label_ap[can] = float(metrics_cal.get("pr_auc_macro", curves_pr[can][2]))
                else:
                    label_auc[can] = float(metrics_raw.get("roc_auc_macro_ovr", curves_roc[can][2]))
                    label_ap[can] = float(metrics_raw.get("pr_auc_macro", curves_pr[can][2]))
        ensure_dirs()
        # Plot ROC (fixed order and colors)
        plt.figure(figsize=(7, 6))
        for name in FIXED_ORDER:
            if name in curves_roc:
                fpr, tpr, _ = curves_roc[name]
                plt.plot(fpr, tpr, label=f"{DISPLAY_LABELS.get(name, name)} (AUC={label_auc[name]:.3f})", lw=2, color=FIXED_COLORS.get(name))
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        mode_tag = "Calibrated" if args.calibrated else "Raw"
        plt.title(f"Macro ROC Comparison ({mode_tag})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_out = os.path.join("results", "roc", "roc.png" if args.calibrated else "roc_raw.png")
        plt.savefig(roc_out)
        plt.close()
        # Plot PR (fixed order and colors)
        plt.figure(figsize=(7, 6))
        for name in FIXED_ORDER:
            if name in curves_pr:
                rec, prec, _ = curves_pr[name]
                plt.plot(rec, prec, label=f"{DISPLAY_LABELS.get(name, name)} (AP={label_ap[name]:.3f})", lw=2, color=FIXED_COLORS.get(name))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Macro PR Comparison ({mode_tag})")
        plt.legend(loc="lower left")
        plt.tight_layout()
        pr_out = os.path.join("results", "pr", "pr.png" if args.calibrated else "pr_raw.png")
        plt.savefig(pr_out)
        plt.close()
        print(f"[Unified] Saved ROC: {roc_out}")
        print(f"[Unified] Saved PR:  {pr_out}")
        return

    # Legacy (weights) mode below -------------------------------------------------

    # Data (only needed in legacy mode)
    train_ds, val_ds, test_ds = build_datasets(CFG.classes, CFG.stratified_val_ratio)
    bs = 64
    nw = min(8, os.cpu_count() or 4)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    # Resolve models directory relative to repo root
    models_dir = os.path.join(ROOT_DIR, "models")

    # Auto-find weights if not provided (support legacy names and renamed files) using absolute paths
    if not args.eff_weights:
        args.eff_weights = auto_find([
            os.path.join(models_dir, "model_stage_best_EfficientNet_*_*.pth"),
            os.path.join(models_dir, "efficientnet_full.pth"),
            os.path.join(models_dir, "efficientnet_partial.pth"),
            os.path.join(models_dir, "efficientnet_head.pth"),
        ])
    if not args.tdn_weights:
        args.tdn_weights = auto_find([
            os.path.join(models_dir, "model_stage_best_ResNet50_*_*.pth"),
            os.path.join(models_dir, "resnet50_full.pth"),
            os.path.join(models_dir, "resnet50_partial.pth"),
            os.path.join(models_dir, "resnet50_head.pth"),
            os.path.join(models_dir, "model_stage_best_TumorDetNet_*_*.pth"),
            os.path.join(models_dir, "tumordetnet_full.pth"),
            os.path.join(models_dir, "tumordetnet_partial.pth"),
            os.path.join(models_dir, "tumordetnet_head.pth"),
        ])
    if not args.hyb_weights:
        args.hyb_weights = auto_find([
            os.path.join(models_dir, "model_stage_best_Hybrid_*_*.pth"),
            os.path.join(models_dir, "hybrid_full.pth"),
            os.path.join(models_dir, "hybrid_partial.pth"),
            os.path.join(models_dir, "hybrid_head.pth"),
        ])

    if not args.hyb_weights:
        raise SystemExit("Missing hybrid weights. Provide --hyb-weights or place compatible files in ./models.")

    # Build models
    eff_model = None
    if args.eff_weights and os.path.exists(args.eff_weights):
        eff_model = CBAMEfficientNetB0(
            num_classes=len(CFG.classes),
            dropout=CFG.eff_dropout,
            cbam_indices=CFG.eff_cbam_indices,
            use_cbam=True,
            variant=CFG.eff_variant,
        ).to(device)
        eff_model.load_state_dict(torch.load(args.eff_weights, map_location="cpu"))
        eff_model.eval()

    tdn_model = None
    if args.tdn_weights and os.path.exists(args.tdn_weights):
        tdn_model = CBAMResNet50Classifier(
            num_classes=len(CFG.classes),
            use_cbam=True,
            cbam_on=CFG.resnet50_cbam_on,
            dropout_head=CFG.resnet50_dropout_head,
        ).to(device)
        tdn_model.load_state_dict(torch.load(args.tdn_weights, map_location="cpu"))
        tdn_model.eval()

    # Hybrid uses its own saved state dict (contains both backbones)
    hyb_eff = CBAMEfficientNetB0(
        num_classes=len(CFG.classes),
        dropout=CFG.eff_dropout,
        cbam_indices=CFG.eff_cbam_indices,
        use_cbam=True,
        variant=CFG.eff_variant,
    ).to(device)
    hyb_tdn = CBAMResNet50Classifier(
        num_classes=len(CFG.classes),
        use_cbam=True,
        cbam_on=CFG.resnet50_cbam_on,
        dropout_head=CFG.resnet50_dropout_head,
    ).to(device)
    hybrid_model = HybridModel(hyb_eff, hyb_tdn).to(device)
    # Load hybrid checkpoint capturing co-adaptation
    hybrid_model.load_state_dict(torch.load(args.hyb_weights, map_location="cpu"))
    hybrid_model.to(device)
    hybrid_model.eval()

    # Temperature calibration on val split
    eff_T = tdn_T = None
    if eff_model is not None:
        eff_val_y, _, eff_val_logits = run_eval(eff_model, val_loader, device)
        eff_T, _ = temperature_search(torch.from_numpy(eff_val_logits), torch.from_numpy(eff_val_y))
    if tdn_model is not None:
        tdn_val_y, _, tdn_val_logits = run_eval(tdn_model, val_loader, device)
        tdn_T, _ = temperature_search(torch.from_numpy(tdn_val_logits), torch.from_numpy(tdn_val_y))
    # Hybrid temperature: prefer report value if provided
    if args.hyb_report and os.path.exists(args.hyb_report):
        try:
            with open(args.hyb_report, 'r', encoding='utf-8') as f:
                rep = json_lib.load(f)
            hyb_T = float(rep.get('temperature', 1.0))
        except Exception:
            hyb_val_y, _, hyb_val_logits = run_eval(hybrid_model, val_loader, device)
            hyb_T, _ = temperature_search(torch.from_numpy(hyb_val_logits), torch.from_numpy(hyb_val_y))
    else:
        hyb_val_y, _, hyb_val_logits = run_eval(hybrid_model, val_loader, device)
        hyb_T, _ = temperature_search(torch.from_numpy(hyb_val_logits), torch.from_numpy(hyb_val_y))

    # Test eval
    # Gather test targets once
    ys = []
    for _, y in test_loader:
        ys.append(y)
    y_true = torch.cat(ys).numpy()

    eff_probs = tdn_probs = None
    if eff_model is not None and eff_T is not None:
        _, _, eff_logits_test = run_eval(eff_model, test_loader, device)
        eff_probs = probs_with_temperature(eff_logits_test, eff_T)
    if tdn_model is not None and tdn_T is not None:
        _, _, tdn_logits_test = run_eval(tdn_model, test_loader, device)
        tdn_probs = probs_with_temperature(tdn_logits_test, tdn_T)
    _, _, hyb_logits_test = run_eval(hybrid_model, test_loader, device)
    hyb_probs = probs_with_temperature(hyb_logits_test, hyb_T)

    n_classes = len(CFG.classes)

    # ROC macro (canonical keys)
    curves_roc: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    if eff_probs is not None:
        curves_roc['efficientnet'] = macro_roc(y_true, eff_probs, n_classes)
    if tdn_probs is not None:
        curves_roc['resnet'] = macro_roc(y_true, tdn_probs, n_classes)
    curves_roc['hybrid'] = macro_roc(y_true, hyb_probs, n_classes)

    # PR macro (canonical keys)
    curves_pr: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    if eff_probs is not None:
        curves_pr['efficientnet'] = macro_pr(y_true, eff_probs, n_classes)
    if tdn_probs is not None:
        curves_pr['resnet'] = macro_pr(y_true, tdn_probs, n_classes)
    curves_pr['hybrid'] = macro_pr(y_true, hyb_probs, n_classes)

    ensure_dirs()

    # Plot ROC (fixed order and colors)
    plt.figure(figsize=(7, 6))
    for name in FIXED_ORDER:
        if name in curves_roc:
            fpr, tpr, auc_val = curves_roc[name]
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", lw=2, color=FIXED_COLORS.get(name))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro ROC Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_out = os.path.join("results", "roc", "combined_roc_compare_hybrid_vs_baselines.png")
    plt.savefig(roc_out)
    plt.close()

    # Plot PR (fixed order and colors)
    plt.figure(figsize=(7, 6))
    for name in FIXED_ORDER:
        if name in curves_pr:
            rec, prec, ap_val = curves_pr[name]
            plt.plot(rec, prec, label=f"{name} (AP={ap_val:.3f})", lw=2, color=FIXED_COLORS.get(name))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro PR Comparison")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_out = os.path.join("results", "pr", "combined_pr_compare_hybrid_vs_baselines.png")
    plt.savefig(pr_out)
    plt.close()

    print(f"Saved ROC: {roc_out}")
    print(f"Saved PR:  {pr_out}")


if __name__ == "__main__":
    main()
