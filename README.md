# 🧠 Brain Tumor Detection (PyTorch: EfficientNet-B0 + CBAM ResNet50 Hybrid)

Modern MRI brain tumor classification with:
* CBAM‑enhanced EfficientNet‑B0 training pipeline
* CBAM ResNet50 classifier with a multi-scale aggregation head
* Flask web inference app that currently serves the integrated hybrid model

## ✅ Current Capabilities
* 4-way classification: glioma · meningioma · pituitary · no tumor
* Snapshot ensemble (weights fixed: 0.1 / 0.3 / 0.6 for head/partial/full)
* Temperature calibration utility (grid search) for improved probability reliability
* Clean, minimal web UI (upload → prediction)
* Training scripts with advanced techniques: staged unfreezing, mixup/cutmix, label smoothing, EMA, cosine LR warmup, gradient clipping, optional gradient checkpointing

## 📂 Key Files
| File / Folder | Purpose |
| ------------- | ------- |
| `app.py` | Flask inference app (hybrid EfficientNet-B0 + ResNet50 model) |
| `training_scripts/cbam_efficient.netb0.py` | EfficientNet‑B0 + CBAM training pipeline |
| `training_scripts/cbam_resnet50.py` | ResNet50 + CBAM + multi‑scale head training pipeline |
| `training_scripts/train_hybrid_integrated.py` | Integrated hybrid training pipeline (EfficientNet‑B0 + ResNet50) |
| `templates/index.html` | Web interface (simplified) |
| `models/` | Model weights (EfficientNet, ResNet50, and hybrid checkpoints) |
| `checkpoints/` | Intermediate training checkpoints (ignored) |
| `uploads/` | User uploads (ignored) |
| `training_logs*/` | Training metrics/logs & reports |
| `requirements.txt` | Runtime + training dependencies |

## 🔧 Installation
Python 3.10+ recommended.
```bash
pip install -r requirements.txt
```
Optional (for CUDA, replace with your version):
```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🕸️ Run Inference Web App
```bash
python app.py
```
Open: http://localhost:5000

Upload an MRI image → receive predicted class + confidence + per-class probabilities.

## 🗂️ Dataset Structure (expected during training)
```
dataset/
  glioma/
  meningioma/
  notumor/
  pituitary/
```
Images: JPG / PNG / JPEG. For strong generalization aim for ≥1000 images per class (balanced if possible).

## 🏋️ Training (EfficientNet CBAM)
Basic invocation (adjust arguments inside script or extend with argparse if desired):
```bash
python training_scripts/cbam_efficient.netb0.py
```
Outputs:
* Stage checkpoints in `checkpoints/`
* Final / best stage models in `models/`
* Logs & metrics ( per-epoch, confusion matrices, calibration metrics if enabled ) in `training_logs*/`

### Techniques Implemented (shared concepts with the ResNet50 pipeline)
* Staged fine-tuning (head → partial → full)
* Mixup & CutMix (probabilistic)
* Label smoothing
* EMA (Exponential Moving Average) of weights with comparison selection
* Cosine decay with warmup
* Gradient scaling (AMP) + checkpointing (memory saving)
* Per-class recall & confusion matrix generation

## 🏋️ Training (ResNet50 / CBAM)
The ResNet50 pipeline uses a ResNet50 backbone + CBAM blocks (configurable on layer1–4) and a multi‑scale aggregation head. Use it when you want higher capacity than EfficientNet-B0 alone.

Run:
```bash
python training_scripts/cbam_resnet50.py
```
Outputs (mirrors EfficientNet script):
* Stage artifacts & pause checkpoints → `checkpoints/`
* Final stage best models → `models/`
* Detailed logs / confusion matrices / JSON report → `training_logs*`

Key additions vs EfficientNet pipeline:
* Multi‑scale head fusion (aggregates intermediate feature maps)
* Discriminative LRs per ResNet stage + head boost
* Optional gradient checkpointing on heavy layers (memory saving)
* Richer metric suite (MCC, Cohen’s Kappa, ECE, Brier, optional bootstrap CIs)

Approximate parameter comparison (inference):
* EfficientNet‑B0 variant: ~5.3M params (≈15–18MB state_dict)
* ResNet50 classifier (ResNet50 backbone + head): ~23–25M params (≈90–105MB fp32)

## 🏋️ Training (Integrated Hybrid)
For end-to-end hybrid training that combines EfficientNet‑B0 and ResNet50 in one workflow:

```bash
python training_scripts/train_hybrid_integrated.py
```

Outputs:
* Hybrid checkpoints in `models/`
* Integrated logs, metrics, and reports in `training_logs/`



## 🌡️ Temperature Calibration (Optional)
The function `calibrate_temperature()` in `app.py` performs a grid search minimizing NLL over a labeled validation set (default folder `archive/Testing/<class>/image.jpg`). Example (Python shell):
```python
from main import calibrate_temperature
print(calibrate_temperature('archive/Testing'))  # returns dict with best_temperature
```
After calibration, subsequent web predictions use the stored global temperature.

## 🧪 Snapshot Ensemble Logic
Three checkpoints (head / partial / full) are loaded:
```
models/efficientnet_head.pth
models/efficientnet_partial.pth
models/efficientnet_full.pth
```
Weights (hard-coded) = [0.1, 0.3, 0.6]; logits combined then softmax(T). If any file is missing a warning is printed and that component contributes zeros.

## ➕ Extending / Adding Models
Place additional `.pth` files in `models/` and adapt the `WEIGHT_FILES` list & weights vector in `app.py` accordingly. Keep temperature calibration updated after changes.

## 🛠️ Regenerating Requirements
All current runtime/training dependencies are pinned in `requirements.txt`. Remove entries you do not need for lightweight inference (e.g. seaborn, matplotlib) if shipping to production.

## ⚖️ Disclaimer
This repository is for research & educational purposes only. It is NOT a medical device. Clinical decisions must involve qualified healthcare professionals and validated diagnostic workflows.


## 🤝 Contributions
Issues / PRs improving robustness (data augmentation, calibration, evaluation) or documentation are welcome.

---
If you need a lean inference-only bundle: keep `app.py`, `templates/`, `models/`, `requirements.txt` (trim heavy libs), and deploy behind a production WSGI server (e.g. gunicorn / waitress) with proper security & logging.