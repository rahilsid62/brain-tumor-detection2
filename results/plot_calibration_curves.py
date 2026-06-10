
import json
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

CAL_DIR = Path(__file__).resolve().parent.parent / 'results' / 'calibration'
LEGACY_CAL_DIR = Path(__file__).resolve().parent.parent / 'results' / 'callibration'
OUT_DIR = CAL_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    # Display names per user request: efficientnetb0 (green), resnet50 (red), hybrid (blue)
    ('efficientnet_calibration.json', 'efficientnetb0'),
    ('resnet50_calibration.json', 'resnet50'),
    ('hybrid_calibration.json', 'hybrid'),
]

# Color mapping aligned with FILES order: green for EfficientNet, red for ResNet, blue for Hybrid
COLORS = ['tab:green', 'tab:red', 'tab:blue']


def resolve_calibration_path(filename: str) -> Path:
    current = CAL_DIR / filename
    if current.exists():
        return current
    legacy_name = filename.replace('resnet50_', 'tumordetnet_')
    legacy = LEGACY_CAL_DIR / legacy_name
    if legacy.exists():
        return legacy
    return current


def load_bins(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    # Current schema: 'ece_bins': list of {mid,count,acc,conf}
    bins = data.get('ece_bins', [])
    conf = [b.get('conf', 0.0) for b in bins]
    acc = [b.get('acc', 0.0) for b in bins]
    counts = [b.get('count', 0) for b in bins]
    ece = data.get('ece')
    brier = data.get('brier_score') or data.get('brier')
    return np.array(conf), np.array(acc), np.array(counts), ece, brier


def expected_calibration_error(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            bin_acc = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    onehot = np.eye(n_classes)[labels]
    se = (probs - onehot) ** 2
    return float(np.mean(np.sum(se, axis=1)))


def compute_and_write_hybrid_calibration(data_root: str | None = None, hyb_weights: str | None = None, hyb_report: str | None = None, n_bins: int = 15):
    """Compute Hybrid calibration bins and write results/callibration/hybrid_calibration.json.
    Requires the integrated trainer to be available for model/dataset utilities.
    """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from training_scripts.train_hybrid_integrated import (
        CBAMEfficientNetB0,
        CBAMResNet50Classifier,
        HybridModel,
        build_datasets,
        run_eval,
        temperature_search,
        CFG,
    )

    if data_root:
        CFG.data_root = data_root

    device = CFG.device
    # Datasets
    train_ds, val_ds, test_ds = build_datasets(CFG.classes, CFG.stratified_val_ratio)
    bs = 64
    nw = min(8, os.cpu_count() or 4)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=torch.cuda.is_available())

    # Build Hybrid and load weights (support renamed files)
    def auto_hyb(models_dir: Path) -> str | None:
        if not models_dir.is_dir():
            return None
        c_full, c_any = [], []
        for fn in os.listdir(models_dir):
            p = models_dir / fn
            low = fn.lower()
            if (fn.startswith('model_stage_best_Hybrid_Full_') and fn.endswith('.pth')) or low == 'hybrid_full.pth':
                c_full.append(p)
            elif (fn.startswith('model_stage_best_Hybrid_') and fn.endswith('.pth')) or (low.startswith('hybrid_') and low.endswith('.pth')):
                c_any.append(p)
        pool = c_full or c_any
        if not pool:
            return None
        pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pool[0])

    if hyb_weights is None:
        hyb_weights = auto_hyb(ROOT_DIR / 'models')
    if hyb_weights is None or not Path(hyb_weights).exists():
        raise SystemExit('Hybrid weights not found to compute calibration.')

    eff = CBAMEfficientNetB0(num_classes=len(CFG.classes), dropout=CFG.eff_dropout, cbam_indices=CFG.eff_cbam_indices, use_cbam=True, variant=CFG.eff_variant)
    resnet50 = CBAMResNet50Classifier(num_classes=len(CFG.classes), use_cbam=True, cbam_on=CFG.resnet50_cbam_on, dropout_head=CFG.resnet50_dropout_head)
    hyb = HybridModel(eff, resnet50).to(device)
    try:
        state = torch.load(hyb_weights, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(hyb_weights, map_location='cpu')
    hyb.load_state_dict(state)
    hyb.eval()

    # Temperature: prefer report value if provided
    if hyb_report and Path(hyb_report).exists():
        try:
            with open(hyb_report, 'r', encoding='utf-8') as f:
                rep = json.load(f)
            T = float(rep.get('temperature', 1.0))
        except Exception:
            _, _, val_logits = run_eval(hyb, val_loader, device)
            T, _ = temperature_search(torch.from_numpy(val_logits), torch.from_numpy(np.array([])))  # fallback next line fixes
            # Proper fallback
            yv, _, val_logits = run_eval(hyb, val_loader, device)
            T, _ = temperature_search(torch.from_numpy(val_logits), torch.from_numpy(yv))
    else:
        yv, _, val_logits = run_eval(hyb, val_loader, device)
        T, _ = temperature_search(torch.from_numpy(val_logits), torch.from_numpy(yv))

    # Test probs
    yt, _, test_logits = run_eval(hyb, test_loader, device)
    probs = torch.softmax(torch.from_numpy(test_logits) / float(T), dim=-1).numpy()
    preds = probs.argmax(1)
    conf = probs.max(1)
    labels = yt

    # Bins
    n_bins = int(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        low, high = edges[i], edges[i+1]
        mask = (conf > low) & (conf <= high)
        if mask.any():
            bin_acc = float((preds[mask] == labels[mask]).mean())
            bin_conf = float(conf[mask].mean())
            bins.append({
                'mid': float(0.5*(low+high)),
                'low': float(low),
                'high': float(high),
                'count': int(mask.sum()),
                'acc': bin_acc,
                'conf': bin_conf,
            })
        else:
            bins.append({
                'mid': float(0.5*(low+high)),
                'low': float(low),
                'high': float(high),
                'count': 0,
                'acc': 0.0,
                'conf': float(0.5*(low+high)),
            })
    ece = expected_calibration_error(conf, preds, labels, n_bins=n_bins)
    brier = brier_score_multiclass(probs, labels)

    out = {
        'ece_bins': bins,
        'ece': ece,
        'brier_score': brier,
        'n_bins': n_bins,
        'temperature': float(T),
        'samples': int(len(labels)),
    }
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_DIR / 'hybrid_calibration.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Wrote Hybrid calibration ->', CAL_DIR / 'hybrid_calibration.json')


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def compute_calibration_from_unified(json_path: Path, calibrated: bool, n_bins: int = 15) -> dict:
    """Compute calibration bins from a unified evaluation JSON + its NPZ probabilities file.
    The NPZ is expected to have test_logits, y_true, temperature.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        rep = json.load(f)
    artifacts = rep.get('artifacts')
    if not isinstance(artifacts, dict) or 'probabilities' not in artifacts:
        ue = rep.get('unified_eval', {}) if isinstance(rep, dict) else {}
        artifacts = ue.get('artifacts', {}) if isinstance(ue, dict) else {}
    npz_rel = artifacts.get('probabilities')
    if npz_rel is None:
        raise FileNotFoundError(f"No 'probabilities' artifact in {json_path}")
    npz_path = Path(npz_rel)
    if not npz_path.is_absolute():
        # First relative to repo root
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / npz_path
        if candidate.exists():
            npz_path = candidate
        else:
            # relative to JSON directory
            npz_path = json_path.parent / npz_path.name
    if not npz_path.exists():
        raise FileNotFoundError(f"Probability NPZ not found: {npz_path}")
    data = np.load(npz_path)
    logits = data['test_logits']
    y_true = data['y_true']
    T = float(data.get('temperature', 1.0))
    if calibrated:
        logits_eff = logits / T
    else:
        logits_eff = logits
    probs = softmax_np(logits_eff)
    preds = probs.argmax(1)
    conf = probs.max(1)
    labels = y_true
    # Build bins
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out = []
    for i in range(n_bins):
        low, high = edges[i], edges[i+1]
        mask = (conf > low) & (conf <= high)
        if mask.any():
            bin_acc = float((preds[mask] == labels[mask]).mean())
            bin_conf = float(conf[mask].mean())
            cnt = int(mask.sum())
        else:
            bin_acc = 0.0
            bin_conf = float(0.5 * (low + high))
            cnt = 0
        bins_out.append({
            'mid': float(0.5 * (low + high)),
            'low': float(low),
            'high': float(high),
            'count': cnt,
            'acc': bin_acc,
            'conf': bin_conf,
        })
    ece = expected_calibration_error(conf, preds, labels, n_bins=n_bins)
    brier = brier_score_multiclass(probs, labels)
    out = {
        'ece_bins': bins_out,
        'ece': ece,
        'brier_score': brier,
        'n_bins': n_bins,
        'temperature_used': T if calibrated else 1.0,
        'calibration_mode': 'calibrated' if calibrated else 'raw',
        'samples': int(len(labels)),
        'source_unified_json': str(json_path),
    }
    return out


def plot_individual():
    for fname, label in FILES:
        path = resolve_calibration_path(fname)
        if not path.exists():
            print(f"[Individual] Missing file: {fname}")
            continue
        conf, acc, counts, ece, brier = load_bins(path)
        if conf.size == 0:
            print(f"No bins in {fname}, skipping")
            continue
        order = np.argsort(conf)
        conf, acc, counts = conf[order], acc[order], counts[order]
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot([0,1],[0,1],'k--',linewidth=1)
        ax.plot(conf, acc, marker='o', label=f'{label}')
        sizes = 20 + 180*(counts / counts.sum())
        ax.scatter(conf, acc, s=sizes, c='tab:blue', alpha=0.6)
        ax.set_xlabel('Predicted Confidence')
        ax.set_ylabel('Empirical Accuracy')
        ax.set_title(f'{label}\nECE={ece:.3f}  Brier={brier:.3f}')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
        fig.tight_layout()
        out_path = OUT_DIR / f'calibration_{label}.png'
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print('Saved', out_path)


def plot_overlay():
    plt.figure(figsize=(5.2,5.2))
    plt.plot([0,1],[0,1],'k--',linewidth=1, label='Ideal')
    markers = ['o','s','^']
    # Use solid lines for all models per user request
    linestyles = ['-','-','-']
    for idx, ((fname, label), color) in enumerate(zip(FILES, COLORS)):
        path = resolve_calibration_path(fname)
        if not path.exists():
            print(f"[Overlay] Missing file: {fname}")
            continue
        conf, acc, counts, ece, brier = load_bins(path)
        if conf.size == 0:
            print(f"[Overlay] Empty bins: {fname}")
            continue
        order = np.argsort(conf)
        conf, acc, counts = conf[order], acc[order], counts[order]
        # Different visual encoding per model (now all solid lines)
        plt.plot(
            conf,
            acc,
            marker=markers[idx % len(markers)],
            linestyle='-',
            color=color,
            linewidth=1.8,
            markersize=4,
            label=f'{label} (ECE {ece:.3f})'
        )
        # Scatter proportional to counts (semi-transparent) to visualize density
        sizes = 30 + 170 * (counts / counts.sum())
        plt.scatter(conf, acc, s=sizes, color=color, alpha=0.35, edgecolors='none')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Calibration Curves (Overlay)')
    plt.xlim(0,1); plt.ylim(0,1)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    out_path = OUT_DIR / 'calibration_overlay.png'
    plt.tight_layout(); plt.savefig(out_path, dpi=220)
    plt.close()
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser(description='Plot calibration curves for EfficientNet / ResNet50 / Hybrid')
    # Legacy compute options (Hybrid only)
    parser.add_argument('--data-root', type=str, default=None, help='Folder containing Training/ and Testing (legacy compute)')
    parser.add_argument('--hyb-weights', type=str, default=None, help='Path to hybrid weights .pth (legacy compute)')
    parser.add_argument('--hyb-report', type=str, default=None, help='Path to hybrid eval report JSON for temperature reuse (legacy compute)')
    parser.add_argument('--recompute-hybrid', action='store_true', help='Recompute Hybrid calibration from model (legacy path)')
    # Unified evaluation JSON paths
    parser.add_argument('--eff-json', type=str, default=None, help='Unified eval JSON for EfficientNet (for calibration derivation)')
    parser.add_argument('--resnet-json', type=str, default=None, help='Unified eval JSON for ResNet50 (for calibration derivation)')
    parser.add_argument('--tdn-json', type=str, default=None, help='Legacy alias for --resnet-json')
    parser.add_argument('--hyb-json', type=str, default=None, help='Unified eval JSON for Hybrid (for calibration derivation)')
    parser.add_argument('--calibrated', action='store_true', help='Use calibrated probabilities (divide logits by stored temperature) when deriving from unified JSON')
    parser.add_argument('--n-bins', type=int, default=15, help='Number of bins for ECE')
    parser.add_argument('--force', action='store_true', help='Overwrite existing calibration JSON if present')
    args = parser.parse_args()

    if args.resnet_json is None:
        args.resnet_json = args.tdn_json

    unified_mode = any([args.eff_json, args.resnet_json, args.hyb_json])

    # Derive from unified JSONs if provided
    if unified_mode:
        CAL_DIR.mkdir(parents=True, exist_ok=True)
        jobs = [
            (args.eff_json, 'efficientnet_calibration.json'),
            (args.resnet_json, 'resnet50_calibration.json'),
            (args.hyb_json, 'hybrid_calibration.json'),
        ]
        for json_path, out_name in jobs:
            if not json_path:
                continue
            out_file = CAL_DIR / out_name
            if out_file.exists() and not args.force:
                print(f"[Skip] {out_name} exists (use --force to overwrite)")
                continue
            try:
                result = compute_calibration_from_unified(Path(json_path), calibrated=args.calibrated, n_bins=args.n_bins)
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                print('[Unified] Wrote', out_file)
            except Exception as e:
                print('[Unified][Error]', out_name, '->', e)
    else:
        # Legacy path: only hybrid recompute supported
        if args.recompute_hybrid or not (CAL_DIR / 'hybrid_calibration.json').exists():
            try:
                compute_and_write_hybrid_calibration(args.data_root, args.hyb_weights, args.hyb_report, n_bins=args.n_bins)
            except Exception as e:
                print('Hybrid calibration compute skipped due to error:', str(e))

    # Plot using whatever JSONs are present
    plot_individual()
    plot_overlay()

if __name__ == '__main__':
    main()
