from flask import Flask, render_template, request, send_from_directory
import torch, os, time, importlib.util, importlib.machinery, sys
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CLASS_LABELS = ['glioma','meningioma','notumor','pituitary']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TEMPERATURE = 1.0
HYB_TEMPERATURE = DEFAULT_TEMPERATURE
HF_REPO_ID = os.getenv('HF_REPO_ID', '').strip()
HF_MODEL_FILENAME = os.getenv('HF_MODEL_FILENAME', 'hybrid_full.pth').strip()
HF_MODEL_SUBDIR = os.getenv('HF_MODEL_SUBDIR', 'models').strip()
HF_TOKEN = os.getenv('HF_TOKEN')

def dyn_load(name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing required script: {path}')
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before execution so dataclasses & typing resolve __module__ correctly
    sys.modules[name] = module
    loader.exec_module(module)
    return module

HYB_SCRIPT = 'training_scripts/train_hybrid_integrated.py'
hyb_mod = dyn_load('hyb_mod', HYB_SCRIPT)
CBAMEfficientNetB0 = getattr(hyb_mod, 'CBAMEfficientNetB0')
CBAMResNet50Classifier = getattr(hyb_mod, 'CBAMResNet50Classifier')
HybridModel = getattr(hyb_mod, 'HybridModel')

def _auto_find_hybrid_weights(models_dir: str = 'models'):
    if not os.path.isdir(models_dir):
        return None
    c_full, c_any = [], []
    for fn in os.listdir(models_dir):
        path = os.path.join(models_dir, fn)
        lower = fn.lower()
        if (fn.startswith('model_stage_best_Hybrid_Full_') and fn.endswith('.pth')) or lower == 'hybrid_full.pth':
            c_full.append(path)
        elif (fn.startswith('model_stage_best_Hybrid_') and fn.endswith('.pth')) or (lower.startswith('hybrid_') and lower.endswith('.pth')):
            c_any.append(path)
    pool = c_full or c_any
    if not pool:
        return None
    pool.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pool[0]

def _build_hybrid_model():
    efficientnetb0_model = CBAMEfficientNetB0(num_classes=4, dropout=0.5, cbam_indices=(2,4,6), use_cbam=True, variant='b0')
    resnet50_model = CBAMResNet50Classifier(num_classes=4, use_cbam=True, cbam_on=('layer1','layer2','layer3','layer4'), dropout_head=0.5)
    return HybridModel(efficientnetb0_model, resnet50_model)

def _normalize_hybrid_state_dict(raw_state: dict):
    # Handle older checkpoints where hybrid branches were named `eff` and `tdn`.
    state = raw_state.get('state_dict', raw_state) if isinstance(raw_state, dict) else raw_state
    if not isinstance(state, dict):
        return state
    renamed = {}
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.'):]
        if nk.startswith('eff.'):
            nk = 'efficientnetb0.' + nk[len('eff.'):]
        elif nk.startswith('tdn.'):
            nk = 'resnet50.' + nk[len('tdn.'):]
        renamed[nk] = v
    return renamed

def _load_hybrid(weights_path: str | None = None):
    model = _build_hybrid_model()
    if weights_path is None:
        weights_path = _auto_find_hybrid_weights('models')
    if not weights_path or not os.path.exists(weights_path):
        if HF_REPO_ID:
            os.makedirs('models', exist_ok=True)
            weights_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME,
                subfolder=HF_MODEL_SUBDIR or None,
                token=HF_TOKEN,
                local_dir='models',
                local_dir_use_symlinks=False,
            )
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(
                'Hybrid weights not found under models/. Set HF_REPO_ID and HF_MODEL_FILENAME, or place hybrid_full.pth in models/.'
            )
    try:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        state = torch.load(weights_path, map_location='cpu')
    state = _normalize_hybrid_state_dict(state)
    model.load_state_dict(state, strict=True)
    print(f'Loaded Hybrid weights: {weights_path}')
    return model.to(DEVICE).eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HYBRID_MODEL = _load_hybrid()

@torch.no_grad()
def hybrid_predict(image_path: str, temperature: float | None = None, return_debug: bool=False):
    global HYB_TEMPERATURE
    img = Image.open(image_path).convert('RGB')
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    logits = HYBRID_MODEL(x)
    T = HYB_TEMPERATURE if temperature is None else max(1e-4, float(temperature))
    probs = torch.softmax(logits / T, dim=-1).squeeze(0)
    pred_idx = int(probs.argmax().item())
    conf = float(probs[pred_idx].item())
    meta = { 'temperature': T, 'model': 'Hybrid(Full)' }
    if return_debug:
        meta['logits'] = logits.detach().cpu().squeeze(0).tolist()
    return CLASS_LABELS[pred_idx], conf, {c: float(probs[i].item()) for i,c in enumerate(CLASS_LABELS)}, meta

@torch.no_grad()
def calibrate_temperature(dataset_dir: str='archive/Testing', t_min: float=0.3, t_max: float=1.5, t_step: float=0.05):
    global HYB_TEMPERATURE
    images = []
    for cls_idx, cls in enumerate(CLASS_LABELS):
        cdir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cdir):
            continue
        for name in os.listdir(cdir):
            if name.lower().endswith(('.png','.jpg','.jpeg')):
                images.append((os.path.join(cdir, name), cls_idx))
    if not images:
        raise RuntimeError('No images found for calibration.')
    all_logits, all_labels = [], []
    for path, lbl in images:
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        logits = HYBRID_MODEL(x).squeeze(0).cpu()
        all_logits.append(logits)
        all_labels.append(lbl)
    all_logits = torch.stack(all_logits, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    best_t, best_nll = None, float('inf')
    for T in torch.arange(t_min, t_max + 1e-9, t_step):
        logp = torch.log_softmax(all_logits / T.item(), dim=-1)
        nll = -logp[torch.arange(len(all_labels)), all_labels].mean().item()
        if nll < best_nll:
            best_nll = nll
            best_t = T.item()
    HYB_TEMPERATURE = best_t if best_t is not None else DEFAULT_TEMPERATURE
    return {'best_temperature': HYB_TEMPERATURE, 'nll': best_nll, 'samples': len(all_labels)}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action','predict')
        if action == 'calibrate':
            path = request.form.get('calib_path','archive/Testing')
            try:
                res = calibrate_temperature(path)
                return render_template('index.html', result=None, calib=res)
            except Exception as e:
                return render_template('index.html', result=None, calib_error=str(e))
        f = request.files.get('file')
        if f and f.filename:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(save_path)
            temp_field = request.form.get('temperature','')
            temp_value = None
            if temp_field:
                try: temp_value = float(temp_field)
                except: temp_value = None
            debug_flag = request.form.get('debug') == 'on'
            label, conf, probs, meta = hybrid_predict(save_path, temperature=temp_value, return_debug=debug_flag)
            display = 'No Tumor' if label == 'notumor' else f'Tumor: {label}'
            return render_template('index.html', result=display, confidence=f'{conf*100:.2f}', file_path=f'/uploads/{f.filename}', probs=probs, meta=meta)
    return render_template('index.html', result=None)

@app.route('/uploads/<path:filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print('Hybrid (EfficientNetB0+ResNet50) app running at http://localhost:5000')
    print(f'Device: {DEVICE}')
    app.run(debug=True)
