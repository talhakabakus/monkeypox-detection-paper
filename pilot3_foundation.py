"""
Pilot 3 on MSLD v2.0 (official patient-independent folds, duplicates screened as in the paper). Pilot 2 showed that
MONET + SigLIP features with a logistic regression (88.4% accuracy) beat every attention head and that the augmented
images add nothing, so this pilot drops both and asks three questions:

  A  backbone panel: frozen features + linear probe for general, dermatology and biomedical foundation models
       general      DenseNet201 (ImageNet), DINOv2-L, DINOv3-L, SigLIP so400m, SigLIP2 so400m
       dermatology  MONET, DermLIP (ViT-B/16), DermLIP-PanDerm (ViT-B/16)
       biomedical   BiomedCLIP
     models that cannot be downloaded (licence gate, old library) are skipped with a message, not fatal
  B  LoRA fine-tuning of MONET and SigLIP (rank 16 on q/k/v/out projections, linear classifier, on-the-fly
     augmentation of the original training images, early stopping on validation loss), and their ensemble
  C  clinical reliability on the pooled test predictions of all 5 folds:
       pooled accuracy with bootstrap 95% CI, McNemar against the reference, ECE
       split conformal prediction (LAC and APS) calibrated on each fold's validation split: coverage, set size
       selective prediction: accuracy when the least confident 10/20/30% are referred, AURC

Reference: 'LP concat monet+siglip_so400m', the best configuration of pilot 2.

Kaggle: Accelerator = GPU T4, Internet = On, input = the MSLD v2.0 dataset. The notebook must contain ONLY this cell
(Save & Run All runs every cell). Start it with "Save Version -> Save & Run All". Expected time: about 1.5-2 h
(A ~15 min, B ~60-90 min).
"""
import glob
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# Settings
# =============================================================================
PANEL = {  # name: (group, library, model id, extra)
    'densenet201':     ('general', 'timm', 'densenet201.tv_in1k', {'img_size': None, 'input': 224}),
    'dinov2_vitl14':   ('general', 'timm', 'vit_large_patch14_dinov2.lvd142m', {'img_size': 224, 'input': 224}),
    'dinov3_vitl16':   ('general', 'timm', 'vit_large_patch16_dinov3.lvd1689m', {'img_size': None, 'input': None}),
    'siglip_so400m':   ('general', 'hf', 'google/siglip-so400m-patch14-384', {}),
    'siglip2_so400m':  ('general', 'hf', 'google/siglip2-so400m-patch14-384', {}),
    'monet':           ('dermatology', 'hf', 'suinleelab/monet', {}),
    'dermlip':         ('dermatology', 'openclip', 'hf-hub:redlessone/DermLIP_ViT-B-16', {}),
    'dermlip_panderm': ('dermatology', 'openclip', 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256', {}),
    'biomedclip':      ('biomedical', 'openclip', 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', {}),
}
REQUIRED = ['monet', 'siglip_so400m']              # the reference needs both
FUSIONS = [['monet', 'siglip_so400m'], ['monet', 'siglip2_so400m'], ['dermlip', 'siglip_so400m']]
LORA_BACKBONES = ['monet', 'siglip_so400m']
LORA_SEEDS = [0]
LORA = dict(rank=16, alpha=32, dropout=0.1, targets=('q_proj', 'k_proj', 'v_proj', 'out_proj'),
            lr=2e-4, head_lr=1e-3, weight_decay=0.01, batch=8, accumulate=2, max_epochs=20, patience=5,
            warmup_epochs=1, label_smoothing=0.1)
FOLDS_DIR = None            # None = search /kaggle/input
OUT = '/kaggle/working/pilot3'
NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
BATCH = 32
WORKERS = 4
ALPHAS = [0.10, 0.05]
REFER = [0.0, 0.1, 0.2, 0.3]                      # fraction of least confident cases referred to a clinician
BOOTSTRAP = 2000
REFERENCE = 'LP concat monet+siglip_so400m'
CONTEXT = {'pilot 2, LP concat monet+siglip_so400m': (88.43, 4.51), 'pilot 2, LP siglip_so400m': (86.17, 7.22),
           'paper, DenseNet201 fine-tuned + augmented': (81.55, 5.10), 'Ali et al. 2024 benchmark': (83.59, 2.11)}

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
ALIASES = {'monkeypox': 'Mpox', 'normal': 'Healthy'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)


def ensure_packages():
    """open_clip for DermLIP/BiomedCLIP; timm >= 1.0.20 for DINOv3 (checked before timm is imported)"""
    def pip(*args):
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], capture_output=True, text=True)
        if r.returncode:
            print(f'  pip install {" ".join(args)} failed: {r.stderr.strip()[-300:]}')
    try:
        importlib.metadata.version('open_clip_torch')
    except importlib.metadata.PackageNotFoundError:
        pip('open_clip_torch')
    try:
        v = tuple(int(x) for x in importlib.metadata.version('timm').split('.')[:3])
    except Exception:
        v = (0,)
    if v < (1, 0, 20) and 'timm' not in sys.modules:
        pip('-U', '--no-deps', 'timm')


# =============================================================================
# Data (identical to pilots 1 and 2)
# =============================================================================
def match_class(name):
    name = name.strip().lower()
    return next((c for c in CLASSES if c.lower() == name), ALIASES.get(name))


def list_images(split_dir):
    rows = []
    for class_dir in sorted(glob.glob(os.path.join(split_dir, '*'))):
        cls = match_class(os.path.basename(class_dir)) if os.path.isdir(class_dir) else None
        if cls:
            rows += [(f, CLASSES.index(cls)) for f in sorted(glob.glob(os.path.join(class_dir, '*')))
                     if f.lower().endswith(IMAGE_EXTENSIONS)]
    return rows


def find_dir(pattern, exclude_aug, root='/kaggle/input'):
    hits = [d for d, sub, _ in os.walk(root) if not (exclude_aug and 'aug' in d.lower())
            and any(pattern.match(s) for s in sub)]
    if len(hits) != 1:
        raise FileNotFoundError(f'expected one folder with {pattern.pattern} sub-folders, found {hits}')
    return hits[0]


def discover_folds(folds_dir):
    folds = []
    for fd in sorted((d for d in glob.glob(os.path.join(folds_dir, '*')) if FOLD_RE.match(os.path.basename(d))),
                     key=lambda d: int(re.sub(r'\D', '', os.path.basename(d)))):
        fold = {'name': os.path.basename(fd)}
        for sd in glob.glob(os.path.join(fd, '*')):
            key = SPLITS.get(os.path.basename(sd).lower())
            if key:
                fold[key] = list_images(sd)
        assert {'train', 'val', 'test'} <= set(fold), fd
        folds.append(fold)
    return folds


_md5, _ahash = {}, {}


def md5(path):
    if path not in _md5:
        with open(path, 'rb') as f:
            _md5[path] = hashlib.md5(f.read()).hexdigest()
    return _md5[path]


def ahash(path, grid=16):
    h = md5(path)
    if h not in _ahash:
        g = np.asarray(Image.open(path).convert('L').resize((224, 224), Image.BICUBIC), dtype=np.float32)
        g = g.reshape(grid, 224 // grid, grid, 224 // grid).mean(axis=(1, 3))
        _ahash[h] = (g > g.mean()).ravel().astype(np.float32)
    return _ahash[h]


def min_hamming(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf)
    return (A @ (1 - B).T + (1 - A) @ B.T).min(axis=1)


def screen(fold):
    train_h = {md5(p) for p, _ in fold['train']}
    report = {'fold': fold['name']}
    seen = set(train_h)
    for split in ('val', 'test'):
        exact = [md5(p) in seen for p, _ in fold[split]]
        kept = [x for x, e in zip(fold[split], exact) if not e]
        near = min_hamming(np.stack([ahash(p) for p, _ in kept]), np.stack([ahash(p) for p, _ in fold['train']]))
        fold[split] = [x for x, n in zip(kept, near) if n > NEAR_THRESHOLD]
        report[f'{split}_exact_removed'] = int(sum(exact))
        report[f'{split}_near_removed'] = int((near <= NEAR_THRESHOLD).sum())
        seen |= {md5(p) for p, _ in fold[split]}
    report.update({f'n_{s}': len(fold[s]) for s in ('train', 'val', 'test')})
    print(json.dumps(report))
    return fold


# =============================================================================
# A: frozen features
# =============================================================================
def as_tensor(out):
    """get_image_features returns a tensor in most transformers versions, a model output in some newer ones"""
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    return out if hasattr(out, 'float') else out[0]


def load_extractor(name, device):
    import torch
    _, lib, mid, extra = PANEL[name]
    if lib == 'timm':
        import timm
        kw = {'img_size': extra['img_size']} if extra.get('img_size') else {}
        model = timm.create_model(mid, pretrained=True, num_classes=0, **kw).eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        if extra.get('input'):
            cfg['input_size'] = (3, extra['input'], extra['input'])
        cfg['crop_pct'] = 1.0  # whole image, no centre crop (lesions reach the borders)
        tf = timm.data.create_transform(**cfg, is_training=False)
        return lambda imgs: model(torch.stack([tf(i) for i in imgs]).to(device))
    if lib == 'hf':
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(mid).eval().to(device)
        proc = AutoImageProcessor.from_pretrained(mid)
        return lambda imgs: as_tensor(model.get_image_features(**proc(images=imgs, return_tensors='pt').to(device)))
    if lib == 'openclip':
        import open_clip
        model, _, tf = open_clip.create_model_and_transforms(mid)
        model = model.eval().to(device)
        return lambda imgs: model.encode_image(torch.stack([tf(i) for i in imgs]).to(device))
    raise ValueError(name)


def extract(name, paths, device):
    """features for every unique image (keyed by MD5), cached on disk; None if the model cannot be loaded"""
    import torch
    cache = os.path.join(OUT, f'features_{name}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return dict(zip(z['keys'], z['feats']))
    try:
        t0 = time.time()
        fn = load_extractor(name, device)
        todo = {}
        for p in paths:
            todo.setdefault(md5(p), p)
        keys, feats = list(todo), {}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            for i in range(0, len(keys), BATCH):
                batch = keys[i:i + BATCH]
                f = fn([Image.open(todo[k]).convert('RGB') for k in batch]).float().cpu().numpy()
                feats.update(zip(batch, f))
        print(f'  {name:16s} ({PANEL[name][0]}): {len(keys)} images in {time.time() - t0:.0f}s, dim {f.shape[1]}')
    except Exception as e:  # licence gate, missing weights, library too old
        if name in REQUIRED:
            raise
        print(f'  {name:16s} SKIPPED: {type(e).__name__}: {str(e).strip().splitlines()[0][:200] if str(e).strip() else ""}')
        return None
    finally:
        fn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    np.savez(cache, keys=np.array(list(feats)), feats=np.stack(list(feats.values())))
    return feats


# =============================================================================
# Metrics and calibration
# =============================================================================
def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def metrics(y, probs):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                                 roc_auc_score)
    pred = probs.argmax(1)
    present = sorted(set(y))
    p = probs[:, present] / probs[:, present].sum(1, keepdims=True)
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
            'f1_macro': f1_score(y, pred, average='macro', labels=list(range(len(CLASSES))), zero_division=0),
            'mcc': matthews_corrcoef(y, pred),
            'auc_macro_ovr': roc_auc_score(y, p, multi_class='ovr', labels=present) if len(present) > 2 else np.nan,
            'ece': ece(probs, y)}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    from scipy.optimize import minimize_scalar
    nll = lambda T: -np.log(softmax(logits / T)[np.arange(len(y)), y] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x


def full_logits(clf, X):
    out = np.full((len(X), len(CLASSES)), -1e4)
    out[:, clf.classes_] = clf.decision_function(X)
    return out


def linear_probe(Xtr, ytr, Xva, yva, Xte):
    """C chosen on validation balanced accuracy, temperature fitted on validation; returns raw test probabilities and
    calibrated validation and test probabilities"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=0).fit(Xtr, ytr)
        s = balanced_accuracy_score(yva, clf.predict(Xva))
        if best is None or s > best[0]:
            best = (s, C, clf)
    clf = best[2]
    lva, lte = full_logits(clf, Xva), full_logits(clf, Xte)
    T = fit_temperature(lva, yva)
    return softmax(lte), softmax(lva / T), softmax(lte / T), {'C': best[1], 'T': T}


# =============================================================================
# B: LoRA fine-tuning
# =============================================================================
def lora_encoder(name, device):
    """vision tower of a transformers CLIP/SigLIP model with LoRA adapters; returns (encoder, size, mean, std)"""
    import torch
    from torch import nn
    from transformers import AutoImageProcessor, AutoModel
    mid = PANEL[name][2]
    model = AutoModel.from_pretrained(mid)
    ip = AutoImageProcessor.from_pretrained(mid)
    size = model.config.vision_config.image_size

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.empty(LORA['rank'], base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, LORA['rank']))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.drop, self.scale = nn.Dropout(LORA['dropout']), LORA['alpha'] / LORA['rank']

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale

    class Encoder(nn.Module):
        def __init__(self, vision, proj):
            super().__init__()
            self.vision, self.proj = vision, proj

        def forward(self, x):
            z = self.vision(pixel_values=x).pooler_output
            return self.proj(z) if self.proj is not None else z

    enc = Encoder(model.vision_model, getattr(model, 'visual_projection', None))
    del model
    for p in enc.parameters():
        p.requires_grad = False
    targets = [(parent, attr) for parent in enc.modules() for attr, child in parent.named_children()
               if attr in LORA['targets'] and isinstance(child, nn.Linear)
               and not isinstance(parent, nn.MultiheadAttention)]  # SigLIP's pooling head uses fused weights
    assert targets, f'no LoRA target modules found in {name}'
    for parent, attr in targets:
        setattr(parent, attr, LoRALinear(getattr(parent, attr)))
    return enc.to(device), size, list(ip.image_mean), list(ip.image_std)


class ImageSet:
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, c = self.rows[i]
        return self.tf(Image.open(p).convert('RGB')), c


def train_lora(name, fold, seed, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.time()
    enc, size, mean, std = lora_encoder(name, device)
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.1, 0.02), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    gen = torch.Generator().manual_seed(seed)
    loader = lambda rows, tf, shuffle, bs: DataLoader(ImageSet(rows, tf), batch_size=bs, shuffle=shuffle,
                                                       num_workers=WORKERS, generator=gen, drop_last=shuffle and
                                                       len(rows) > bs, pin_memory=device == 'cuda')
    train_dl = loader(fold['train'], train_tf, True, LORA['batch'])
    eval_dl = {s: loader(fold[s], eval_tf, False, BATCH) for s in ('val', 'test')}
    with torch.no_grad():
        dim = enc(torch.zeros(1, 3, size, size, device=device)).shape[-1]
    head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, len(CLASSES))).to(device)
    ytr = np.array([c for _, c in fold['train']])
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.maximum(np.bincount(ytr, minlength=len(CLASSES)), 1)),
                     dtype=torch.float32, device=device)
    train_loss = nn.CrossEntropyLoss(weight=w, label_smoothing=LORA['label_smoothing'])
    val_loss = nn.CrossEntropyLoss(weight=w)
    lora_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': lora_params, 'lr': LORA['lr']},
                             {'params': head.parameters(), 'lr': LORA['head_lr']}], weight_decay=LORA['weight_decay'])
    steps_per_epoch = max(1, len(train_dl) // LORA['accumulate'])
    total, warm = steps_per_epoch * LORA['max_epochs'], steps_per_epoch * LORA['warmup_epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    amp = lambda: torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda')
    trainable = lambda: {k: v.detach().clone() for k, v in list(enc.state_dict().items()) +
                         [('head.' + k, v) for k, v in head.state_dict().items()]
                         if k.endswith(('.A', '.B')) or k.startswith('head.')}

    def logits(split):
        enc.eval(), head.eval()
        out = []
        with torch.no_grad(), amp():
            for x, _ in eval_dl[split]:
                out.append(head(enc(x.to(device))).float().cpu())
        return torch.cat(out)

    yva = torch.tensor([c for _, c in fold['val']])
    best, best_state, wait = np.inf, None, 0
    for epoch in range(LORA['max_epochs']):
        enc.train(), head.train()
        opt.zero_grad(set_to_none=True)
        for i, (x, yb) in enumerate(train_dl):
            with amp():
                loss = train_loss(head(enc(x.to(device))), yb.to(device)) / LORA['accumulate']
            scaler.scale(loss).backward()
            if (i + 1) % LORA['accumulate'] == 0 or i + 1 == len(train_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        vl = val_loss(logits('val').to(device), yva.to(device)).item()
        if vl < best - 1e-4:
            best, wait, best_state = vl, 0, trainable()
        else:
            wait += 1
            if wait >= LORA['patience']:
                break
    enc.load_state_dict({k: v for k, v in best_state.items() if not k.startswith('head.')}, strict=False)
    head.load_state_dict({k[5:]: v for k, v in best_state.items() if k.startswith('head.')})
    lva, lte = logits('val').numpy(), logits('test').numpy()
    temp = fit_temperature(lva, yva.numpy())
    info = {'epochs': epoch + 1, 'best_epoch': epoch + 1 - wait, 'T': temp, 'seconds': time.time() - t0}
    del enc, head, opt
    if device == 'cuda':
        torch.cuda.empty_cache()
    return softmax(lte), softmax(lva / temp), softmax(lte / temp), info


# =============================================================================
# C: statistics, conformal prediction, selective prediction
# =============================================================================
def corrected_ttest(a, b, ratio):
    from scipy import stats
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * var)
    return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def mcnemar(correct_a, correct_b):
    from scipy.stats import binomtest
    b, c = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def conformal_sets(p_cal, y_cal, p_test, alpha, method):
    """split conformal prediction; LAC: score 1 - p_y; APS (deterministic): probability mass down to the true class"""
    n = len(y_cal)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    if method == 'LAC':
        q = np.quantile(1 - p_cal[np.arange(n), y_cal], level, method='higher')
        return p_test >= 1 - q
    order = np.argsort(-p_cal, axis=1)
    cum = np.cumsum(np.take_along_axis(p_cal, order, 1), 1)
    rank = np.argmax(order == y_cal[:, None], axis=1)
    q = np.quantile(cum[np.arange(n), rank], level, method='higher')
    order = np.argsort(-p_test, axis=1)
    cum_excl = np.cumsum(np.take_along_axis(p_test, order, 1), 1) - np.take_along_axis(p_test, order, 1)
    sets = np.zeros_like(p_test, dtype=bool)
    np.put_along_axis(sets, order, cum_excl < q, 1)
    return sets


def selective(probs, y):
    conf, correct = probs.max(1), probs.argmax(1) == y
    correct = correct[np.argsort(-conf, kind='stable')]
    n = len(y)
    out = {f'acc refer {int(r * 100)}%': correct[:max(1, int(round(n * (1 - r))))].mean() for r in REFER}
    out['AURC'] = np.mean(1 - np.cumsum(correct) / np.arange(1, n + 1))
    return out


# =============================================================================
def main():
    ensure_packages()
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import transformers
    print(f'device: {device}; torch {torch.__version__}; transformers {transformers.__version__}; '
          f'timm {importlib.metadata.version("timm")}')
    folds = [screen(f) for f in discover_folds(FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True))]
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})
    ratio = np.mean([len(f['test']) / len(f['train']) for f in folds])

    print('\n--- A: frozen features ---')
    feats = {}
    for name in PANEL:
        f = extract(name, originals, device)
        if f is not None:
            feats[name] = f
    fusions = [fu for fu in FUSIONS if all(b in feats for b in fu)]

    rows, preds = [], []

    def record(config, f, raw_test, cal_val, cal_test, **extra):
        y = np.array([c for _, c in f['test']])
        m = metrics(y, cal_test)
        m['ece_uncalibrated'] = ece(raw_test, y)
        rows.append(dict(config=config, fold=f['name'], **m, **extra))
        for split, probs in (('val', cal_val), ('test', cal_test)):
            preds.append(pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES]).assign(
                config=config, fold=f['name'], split=split, y_true=[c for _, c in f[split]],
                path=[p for p, _ in f[split]]))
        print(f'  {config:40s} | {f["name"]} | acc={m["accuracy"]:.4f} bacc={m["balanced_accuracy"]:.4f} '
              f'ece {m["ece_uncalibrated"]:.3f}->{m["ece"]:.3f}', flush=True)

    for f in folds:
        y = {s: np.array([c for _, c in f[s]]) for s in ('train', 'val', 'test')}
        X = lambda names, s: np.concatenate([np.stack([feats[n][md5(p)] for p, _ in f[s]]) for n in names], 1)
        for config, names in [(f'LP {n}', [n]) for n in feats] + [(f'LP concat {"+".join(fu)}', fu) for fu in fusions]:
            raw, cva, cte, info = linear_probe(X(names, 'train'), y['train'], X(names, 'val'), y['val'],
                                               X(names, 'test'))
            record(config, f, raw, cva, cte, **info)

    print('\n--- B: LoRA fine-tuning ---')
    lora = {}
    for name in LORA_BACKBONES:
        for f in folds:
            runs = [train_lora(name, f, seed, device) for seed in LORA_SEEDS]
            raw, cva, cte = (np.mean([r[k] for r in runs], 0) for k in range(3))  # seed ensemble if > 1 seed
            info = {k: float(np.mean([r[3][k] for r in runs])) for k in runs[0][3]}
            lora[name, f['name']] = (cva, cte)
            record(f'LoRA {name}', f, raw, cva, cte, **info)
            print(f'    {info["epochs"]:.0f} epochs (best {info["best_epoch"]:.0f}), {info["seconds"]:.0f}s')
    if len(LORA_BACKBONES) > 1:
        for f in folds:
            cva, cte = (np.mean([lora[n, f['name']][k] for n in LORA_BACKBONES], 0) for k in range(2))
            record(f'ensemble LoRA {"+".join(LORA_BACKBONES)}', f, cte, cva, cte)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'pilot3_folds.csv'), index=False)
    pr = pd.concat(preds)
    pr.to_csv(os.path.join(OUT, 'pilot3_predictions.csv'), index=False)
    if REFERENCE not in set(res['config']):
        raise RuntimeError(f'{REFERENCE} missing; results are in {OUT}/pilot3_folds.csv')
    configs = list(dict.fromkeys(res['config']))
    pcols = [f'p_{c}' for c in CLASSES]
    group = lambda c: next((PANEL[n][0] for n in PANEL if c == f'LP {n}'), 'fusion' if 'concat' in c or
                           'ensemble' in c else 'fine-tuned')

    # ---- table 1: mean ± SD over folds, corrected resampled t-test
    ref = res[res['config'] == REFERENCE].set_index('fold')
    t1 = []
    for config in configs:
        g = res[res['config'] == config]
        row = {'config': config, 'group': group(config)}
        for k in ('accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_macro_ovr'):
            row[k] = f'{g[k].mean() * 100:.2f} ± {g[k].std(ddof=1) * 100:.2f}'
        row['ece raw->cal'] = f'{g["ece_uncalibrated"].mean() * 100:.1f} -> {g["ece"].mean() * 100:.1f}'
        row['_d'], row['_p'] = (np.nan, np.nan) if config == REFERENCE else corrected_ttest(
            g['accuracy'].to_numpy(), ref.loc[g['fold'], 'accuracy'].to_numpy(), ratio)
        t1.append(row)
    t1 = pd.DataFrame(t1)
    ok = t1['_p'].notna()
    t1.loc[ok, '_h'] = holm(t1.loc[ok, '_p'])
    t1['acc vs ref (t-test)'] = ['(reference)' if c == REFERENCE else f'{d * 100:+.2f} (p={p:.3f}, Holm {h:.3f})'
                                 if pd.notna(p) else f'{d * 100:+.2f} (identical folds)'
                                 for c, d, p, h in zip(t1['config'], t1['_d'], t1['_p'], t1['_h'])]

    # ---- table 2: pooled test predictions of all folds
    test = pr[pr['split'] == 'test']
    ref_t = test[test['config'] == REFERENCE]
    ref_correct = ref_t[pcols].to_numpy().argmax(1) == ref_t['y_true'].to_numpy()
    rng = np.random.default_rng(0)
    t2 = []
    for config in configs:
        g = test[test['config'] == config]
        assert (g['path'].to_numpy() == ref_t['path'].to_numpy()).all()
        P, y = g[pcols].to_numpy(), g['y_true'].to_numpy()
        correct = P.argmax(1) == y
        boot = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(BOOTSTRAP)]
        m = metrics(y, P)
        t2.append({'config': config, 'n': len(y),
                   'accuracy [95% CI]': f'{correct.mean() * 100:.2f} [{np.percentile(boot, 2.5) * 100:.1f}, '
                                        f'{np.percentile(boot, 97.5) * 100:.1f}]',
                   'balanced_acc': f'{m["balanced_accuracy"] * 100:.2f}', 'mcc': f'{m["mcc"] * 100:.2f}',
                   'mpox recall': f'{(P.argmax(1)[y == CLASSES.index("Mpox")] == CLASSES.index("Mpox")).mean() * 100:.1f}',
                   'ece': f'{m["ece"] * 100:.1f}',
                   '_p': np.nan if config == REFERENCE else mcnemar(correct, ref_correct)})
    t2 = pd.DataFrame(t2)
    ok = t2['_p'].notna()
    t2.loc[ok, '_h'] = holm(t2.loc[ok, '_p'])
    t2['McNemar vs ref'] = [f'p={p:.3f}, Holm {h:.3f}' if pd.notna(p) else '(reference)'
                            for p, h in zip(t2['_p'], t2['_h'])]

    # ---- table 3: conformal prediction (calibrated on each fold's validation split) and selective prediction
    t3 = []
    for config in configs:
        row = {'config': config}
        for alpha in ALPHAS:
            for method in ('LAC', 'APS'):
                sets, ys = [], []
                for fold in res['fold'].unique():
                    g = pr[(pr['config'] == config) & (pr['fold'] == fold)]
                    va, te = g[g['split'] == 'val'], g[g['split'] == 'test']
                    sets.append(conformal_sets(va[pcols].to_numpy(), va['y_true'].to_numpy(), te[pcols].to_numpy(),
                                               alpha, method))
                    ys.append(te['y_true'].to_numpy())
                S, y = np.concatenate(sets), np.concatenate(ys)
                cover = S[np.arange(len(y)), y]
                mp = y == CLASSES.index('Mpox')
                row[f'{method} {int((1 - alpha) * 100)}%: cov/size/single/mpox'] = (
                    f'{cover.mean() * 100:.1f}/{S.sum(1).mean():.2f}/{(S.sum(1) == 1).mean() * 100:.0f}/'
                    f'{cover[mp].mean() * 100:.0f}')
        g = test[test['config'] == config]
        row.update({k: f'{v * 100:.1f}' if k != 'AURC' else f'{v:.4f}'
                    for k, v in selective(g[pcols].to_numpy(), g['y_true'].to_numpy()).items()})
        t3.append(row)
    t3 = pd.DataFrame(t3)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print('\n=== Pilot 3, table 1: mean ± SD over the 5 official folds (%); calibrated on validation ===')
    print(t1.drop(columns=['_d', '_p', '_h']).to_string(index=False))
    print(f'\n=== Pilot 3, table 2: pooled test predictions of the 5 folds (%); bootstrap {BOOTSTRAP}x ===')
    print(t2.drop(columns=['_p', '_h']).to_string(index=False))
    print('\n=== Pilot 3, table 3a: split conformal prediction, pooled over folds '
          '(coverage % / mean set size / singleton % / mpox coverage %) ===')
    print(t3[['config'] + [c for c in t3 if ':' in c]].to_string(index=False))
    print('\n=== Pilot 3, table 3b: selective prediction, pooled (accuracy % after referring the least confident) ===')
    print(t3[['config'] + [c for c in t3 if c.startswith('acc refer') or c == 'AURC']].to_string(index=False))
    print('\nFor context (accuracy, mean ± SD over folds):')
    for k, (m, s) in CONTEXT.items():
        print(f'  {k}: {m:.2f} ± {s:.2f}')
    for name, t in (('table1', t1.drop(columns=['_d', '_p', '_h'])), ('table2', t2), ('table3', t3)):
        t.to_csv(os.path.join(OUT, f'pilot3_{name}.csv'), index=False)
    print(f'\nFiles: {OUT}/pilot3_folds.csv, pilot3_predictions.csv, pilot3_table1-3.csv, features_*.npz')


main()
