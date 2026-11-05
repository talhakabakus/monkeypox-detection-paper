"""
Pilot 4 on MSLD v2.0 (official patient-independent folds, duplicates screened as in the paper). Pilot 3 made
'LP concat monet+siglip2_so400m' the main model (89.7% pooled accuracy) but its conformal sets covered only 87.7% of
the test images at the 90% target, and only 82% of the mpox images. This pilot needs no new training of networks; it
reuses the frozen features saved by pilot 3 and asks:

  1  protocol: in pilots 2-3 the validation split chose C, fitted the temperature AND calibrated conformal prediction.
       reuse   as in pilot 3
       clean   C and temperature chosen by 5-fold cross-validation inside the training split; the validation split is
               used only for conformal calibration
  2  conformal prediction: marginal LAC versus class-conditional (Mondrian) LAC, which guarantees the coverage for
     every class separately, mpox included
  3  mpox screening: confusion matrix, mpox-vs-rest sensitivity/specificity/PPV/NPV/AUC, and an operating point chosen
     on validation for 95% mpox sensitivity
  4  selective prediction, and figures: reliability diagram, risk-coverage curve, confusion matrix, class coverage

Kaggle: Accelerator = None (CPU is enough), inputs = the MSLD v2.0 dataset AND the output of the pilot 3 notebook
(Add Input -> Your Work -> the pilot 3 notebook). Paste the whole file into ONE cell (the notebook must contain only
this cell) and start it with "Save Version -> Save & Run All". Expected time: about 10-20 min. Missing feature files
are recomputed (then Internet = On is needed, and a GPU makes it faster).
"""
import glob
import hashlib
import importlib.metadata
import json
import math
import os
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
FEATURE_DIRS = ['/kaggle/input', '/kaggle/working']  # where pilot 3's features_*.npz are searched
FOLDS_DIR = None            # None = search /kaggle/input
OUT = '/kaggle/working/pilot4'
MAIN = 'LP concat monet+siglip2_so400m'
CONFIGS = {  # config: backbones whose features are concatenated
    'LP densenet201': ['densenet201'], 'LP dinov3_vitl16': ['dinov3_vitl16'], 'LP biomedclip': ['biomedclip'],
    'LP monet': ['monet'], 'LP siglip_so400m': ['siglip_so400m'], 'LP siglip2_so400m': ['siglip2_so400m'],
    'LP concat monet+siglip_so400m': ['monet', 'siglip_so400m'], MAIN: ['monet', 'siglip2_so400m']}
CONFORMAL_CONFIGS = ['LP densenet201', 'LP monet', 'LP siglip2_so400m', 'LP concat monet+siglip_so400m', MAIN]
CURVE_CONFIGS = ['LP densenet201', 'LP monet', 'LP siglip2_so400m', MAIN]
REQUIRED = ['monet', 'siglip2_so400m']
NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5
ALPHAS = [0.10, 0.05]
SENS_TARGET = 0.95
REFER = [0.0, 0.1, 0.2, 0.3]
BOOTSTRAP = 2000
BATCH = 32
DPI = 300
CLASS_COLOR = {'Chickenpox': '#E69F00', 'Cowpox': '#56B4E9', 'HFMD': '#009E73', 'Healthy': '#0072B2',
               'Measles': '#CC79A7', 'Mpox': '#D55E00'}   # Okabe-Ito, as in the paper's figures
CONFIG_COLOR = {'LP densenet201': '#999999', 'LP monet': '#E69F00', 'LP siglip2_so400m': '#56B4E9', MAIN: '#D55E00'}

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
MPOX = CLASSES.index('Mpox')
ALIASES = {'monkeypox': 'Mpox', 'normal': 'Healthy'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)

# backbones exactly as in pilot 3 (only used when a feature file is missing)
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



def selective(probs, y):
    conf, correct = probs.max(1), probs.argmax(1) == y
    correct = correct[np.argsort(-conf, kind='stable')]
    n = len(y)
    out = {f'acc refer {int(r * 100)}%': correct[:max(1, int(round(n * (1 - r))))].mean() for r in REFER}
    out['AURC'] = np.mean(1 - np.cumsum(correct) / np.arange(1, n + 1))
    return out




# =============================================================================
# Pilot 4
# =============================================================================
def get_features(name, originals, device):
    """pilot 3's features (keyed by image MD5) if they cover every image, otherwise recomputed"""
    need = {md5(p) for p in originals}
    hits = sorted({f for root in FEATURE_DIRS for f in glob.glob(os.path.join(root, '**', f'features_{name}.npz'),
                                                                   recursive=True)},
                  key=lambda f: ('pilot3' not in f, f))
    for f in hits:
        z = np.load(f)
        feats = dict(zip(z['keys'], z['feats']))
        if need <= set(feats):
            print(f'  {name:16s} features from {f}')
            return feats
        print(f'  {name:16s} {f} lacks {len(need - set(feats))} images, ignored')
    print(f'  {name:16s} no feature file found: extracting')
    return extract(name, originals, device)


def probe_clean(Xtr, ytr, Xva, Xte):
    """C chosen on out-of-fold balanced accuracy inside the training split, temperature fitted on the out-of-fold
    logits; the validation split is not touched. Returns raw test, calibrated validation and test probabilities"""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    splits = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=0).split(Xtr, ytr))

    def oof_part(C, tr, te):
        sc = StandardScaler().fit(Xtr[tr])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=0)
        clf.fit(sc.transform(Xtr[tr]), ytr[tr])
        return C, te, full_logits(clf, sc.transform(Xtr[te]))

    parts = Parallel(n_jobs=-1)(delayed(oof_part)(C, tr, te) for C in C_GRID for tr, te in splits)
    oof = {C: np.zeros((len(ytr), len(CLASSES))) for C in C_GRID}
    for C, te, lg in parts:
        oof[C][te] = lg
    score = {C: balanced_accuracy_score(ytr, oof[C].argmax(1)) for C in C_GRID}
    C = max(C_GRID, key=lambda c: (score[c], -c))  # ties -> stronger regularisation
    T = fit_temperature(oof[C], ytr)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=0).fit(sc.transform(Xtr), ytr)
    lva, lte = full_logits(clf, sc.transform(Xva)), full_logits(clf, sc.transform(Xte))
    return softmax(lte), softmax(lva / T), softmax(lte / T), {'C': C, 'T': T}


def lac_sets(p_cal, y_cal, p_test, alpha, mondrian):
    """split conformal prediction with the LAC score 1 - p_y; Mondrian = one threshold per class"""
    def quantile(scores):
        n = len(scores)
        if n == 0:
            return 1.0
        k = math.ceil((n + 1) * (1 - alpha))
        return 1.0 if k > n else np.sort(scores)[k - 1]
    scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    if not mondrian:
        return p_test >= 1 - quantile(scores)
    q = np.array([quantile(scores[y_cal == k]) for k in range(len(CLASSES))])
    return p_test >= 1 - q[None, :]


def binary(y, p, thr=None):
    """mpox vs rest; positive = mpox predicted (argmax) or p_mpox >= thr"""
    from sklearn.metrics import roc_auc_score
    pos = y == MPOX
    flag = p.argmax(1) == MPOX if thr is None else p[:, MPOX] >= thr
    tp, fn = (flag & pos).sum(), (~flag & pos).sum()
    tn, fp = (~flag & ~pos).sum(), (flag & ~pos).sum()
    div = lambda a, b: a / b if b else np.nan
    return {'sensitivity': div(tp, tp + fn), 'specificity': div(tn, tn + fp), 'ppv': div(tp, tp + fp),
            'npv': div(tn, tn + fn), 'auc': roc_auc_score(pos, p[:, MPOX])}


def threshold_for_sensitivity(y, p, target):
    """largest threshold whose validation mpox sensitivity is at least the target"""
    s = np.sort(p[y == MPOX, MPOX])[::-1]
    return s[max(0, math.ceil(target * len(s)) - 1)] if len(s) else 0.5


def pooled(preds, protocol, config, key):
    return np.concatenate([preds[protocol, config, f][key] for f in FOLD_NAMES])


def figures(preds, cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    y = pooled(preds, 'clean', MAIN, 'y_test')

    # reliability diagram of the main model, before and after temperature scaling
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for key, label, color in (('test_raw', 'uncalibrated', '#999999'), ('test', 'temperature-scaled', '#D55E00')):
        p = pooled(preds, 'clean', MAIN, key)
        conf, correct = p.max(1), p.argmax(1) == y
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [correct[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ax.plot(xs, ys, 'o-', color=color, ms=3.5, lw=1.2, label=f'{label} (ECE {ece(p, y) * 100:.1f}%)')
    ax.plot([0, 1], [0, 1], ':', color='#555555', lw=0.8)
    ax.set(xlabel='Confidence', ylabel='Accuracy', xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_reliability.png'), dpi=DPI)
    plt.close(fig)

    # risk-coverage curves
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for config in CURVE_CONFIGS:
        if ('clean', config, FOLD_NAMES[0]) not in preds:
            continue
        p = pooled(preds, 'clean', config, 'test')
        correct = (p.argmax(1) == y)[np.argsort(-p.max(1), kind='stable')]
        cov = np.arange(1, len(y) + 1) / len(y)
        ax.plot(cov * 100, np.cumsum(correct) / np.arange(1, len(y) + 1) * 100, color=CONFIG_COLOR.get(config),
                lw=1.6 if config == MAIN else 1.1, label=config.replace('LP ', '').replace('concat ', ''))
    ax.set(xlabel='Coverage (%)', ylabel='Accuracy on retained cases (%)', xlim=(20, 100))
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_risk_coverage.png'), dpi=DPI)
    plt.close(fig)

    # confusion matrix (row-normalised colour, counts as text)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=8,
                    color='white' if norm[i, j] > 0.55 else '#222222')
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha='right')
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel='Predicted', ylabel='True')
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_confusion.png'), dpi=DPI)
    plt.close(fig)

    # class-wise coverage at 90%: marginal vs Mondrian
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    width = 0.38
    for off, mondrian, label, hatch in ((-width / 2, False, 'marginal', ''), (width / 2, True, 'class-conditional', '//')):
        S = np.concatenate([lac_sets(preds['clean', MAIN, f]['val'], preds['clean', MAIN, f]['y_val'],
                                     preds['clean', MAIN, f]['test'], 0.10, mondrian) for f in FOLD_NAMES])
        cover = S[np.arange(len(y)), y]
        vals = [cover[y == k].mean() * 100 for k in range(len(CLASSES))]
        ax.bar(np.arange(len(CLASSES)) + off, vals, width, color=[CLASS_COLOR[c] for c in CLASSES],
               hatch=hatch, edgecolor='white', linewidth=0.6, label=label)
    ax.axhline(90, ls=':', color='#555555', lw=0.8)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha='right')
    ax.set(ylabel='Coverage (%)', ylim=(50, 101))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#BBBBBB', label='marginal'),
                       Patch(facecolor='#BBBBBB', hatch='//', edgecolor='white', label='class-conditional')],
              frameon=False, fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_class_coverage.png'), dpi=DPI)
    plt.close(fig)


def main():
    global FOLD_NAMES
    ensure_packages()
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    folds = [screen(f) for f in discover_folds(FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True))]
    FOLD_NAMES = [f['name'] for f in folds]
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})

    print('\n--- features ---')
    needed = list(dict.fromkeys(b for bs in CONFIGS.values() for b in bs))
    feats = {n: f for n in needed if (f := get_features(n, originals, device)) is not None}
    configs = {c: bs for c, bs in CONFIGS.items() if all(b in feats for b in bs)}

    print('\n--- probes ---')
    preds, t0 = {}, time.time()
    for f in folds:
        y = {s: np.array([c for _, c in f[s]]) for s in ('train', 'val', 'test')}
        for config, names in configs.items():
            X = {s: np.concatenate([np.stack([feats[n][md5(p)] for p, _ in f[s]]) for n in names], 1)
                 for s in ('train', 'val', 'test')}
            for protocol in ('reuse', 'clean'):
                if protocol == 'reuse':
                    raw, cva, cte, info = linear_probe(X['train'], y['train'], X['val'], y['val'], X['test'])
                else:
                    raw, cva, cte, info = probe_clean(X['train'], y['train'], X['val'], X['test'])
                preds[protocol, config, f['name']] = dict(test_raw=raw, val=cva, test=cte, y_val=y['val'],
                                                          y_test=y['test'], **info)
        print(f'  {f["name"]} done ({time.time() - t0:.0f}s)', flush=True)

    rng = np.random.default_rng(0)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    y = pooled(preds, 'clean', MAIN, 'y_test')
    tables = {}

    # ---- table 1: pooled performance under both protocols
    t1 = []
    for protocol in ('reuse', 'clean'):
        ref_correct = pooled(preds, protocol, MAIN, 'test').argmax(1) == y
        for config in configs:
            P, R = pooled(preds, protocol, config, 'test'), pooled(preds, protocol, config, 'test_raw')
            correct = P.argmax(1) == y
            boot = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(BOOTSTRAP)]
            m, b = metrics(y, P), binary(y, P)
            accs = [(preds[protocol, config, f]['test'].argmax(1) == preds[protocol, config, f]['y_test']).mean()
                    for f in FOLD_NAMES]
            t1.append({'protocol': protocol, 'config': config,
                       'acc folds': f'{np.mean(accs) * 100:.2f} ± {np.std(accs, ddof=1) * 100:.2f}',
                       'acc pooled [95% CI]': f'{correct.mean() * 100:.2f} [{np.percentile(boot, 2.5) * 100:.1f}, '
                                              f'{np.percentile(boot, 97.5) * 100:.1f}]',
                       'bal_acc': f'{m["balanced_accuracy"] * 100:.2f}', 'f1': f'{m["f1_macro"] * 100:.2f}',
                       'mcc': f'{m["mcc"] * 100:.2f}', 'ece raw->cal': f'{ece(R, y) * 100:.1f} -> {m["ece"] * 100:.1f}',
                       'mpox sens/spec': f'{b["sensitivity"] * 100:.1f}/{b["specificity"] * 100:.1f}',
                       'mpox auc': f'{b["auc"] * 100:.2f}',
                       'C (folds)': ','.join(f'{preds[protocol, config, f]["C"]:g}' for f in FOLD_NAMES),
                       '_p': np.nan if config == MAIN else mcnemar(correct, ref_correct)})
    t1 = pd.DataFrame(t1)
    for protocol in ('reuse', 'clean'):
        ok = (t1['protocol'] == protocol) & t1['_p'].notna()
        t1.loc[ok, '_h'] = holm(t1.loc[ok, '_p'])
    t1[f'McNemar vs main'] = ['(main)' if pd.isna(p) else f'p={p:.3f}, Holm {h:.3f}'
                              for p, h in zip(t1['_p'], t1['_h'])]
    tables['table1_performance'] = t1.drop(columns=['_p', '_h'])

    # ---- table 2: conformal prediction
    t2 = []
    for protocol in ('reuse', 'clean'):
        for config in [c for c in CONFORMAL_CONFIGS if c in configs]:
            for alpha in ALPHAS:
                for mondrian in (False, True):
                    S = np.concatenate([lac_sets(preds[protocol, config, f]['val'], preds[protocol, config, f]['y_val'],
                                                 preds[protocol, config, f]['test'], alpha, mondrian)
                                        for f in FOLD_NAMES])
                    cover, size = S[np.arange(len(y)), y], S.sum(1)
                    per_class = [cover[y == k].mean() for k in range(len(CLASSES))]
                    t2.append({'protocol': protocol, 'config': config, 'target': f'{int((1 - alpha) * 100)}%',
                               'method': 'class-conditional' if mondrian else 'marginal',
                               'coverage': f'{cover.mean() * 100:.1f}', 'mpox cov': f'{per_class[MPOX] * 100:.1f}',
                               'min class cov': f'{min(per_class) * 100:.1f} ({CLASSES[int(np.argmin(per_class))]})',
                               'mean size': f'{size.mean():.2f}', 'singleton %': f'{(size == 1).mean() * 100:.0f}',
                               'empty %': f'{(size == 0).mean() * 100:.1f}',
                               'non-mpox sets with mpox %': f'{S[y != MPOX, MPOX].mean() * 100:.1f}'})
    tables['table2_conformal'] = pd.DataFrame(t2)

    # ---- table 3: confusion matrix and per-class metrics of the main model (clean protocol)
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    P = pooled(preds, 'clean', MAIN, 'test')
    cm = confusion_matrix(y, P.argmax(1), labels=range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, P.argmax(1), labels=range(len(CLASSES)), zero_division=0)
    t3 = pd.DataFrame(cm, index=[f'true {c}' for c in CLASSES], columns=[f'pred {c}' for c in CLASSES])
    t3['n'], t3['recall %'], t3['precision %'], t3['f1 %'] = sup, (rc * 100).round(1), (pr * 100).round(1), \
        (f1 * 100).round(1)
    tables['table3_confusion_main'] = t3.reset_index().rename(columns={'index': 'class'})

    # ---- table 4: mpox screening operating points (clean protocol)
    t4 = []
    for config in configs:
        thr = {f: threshold_for_sensitivity(preds['clean', config, f]['y_val'], preds['clean', config, f]['val'],
                                            SENS_TARGET) for f in FOLD_NAMES}
        flags = np.concatenate([preds['clean', config, f]['test'][:, MPOX] >= thr[f] for f in FOLD_NAMES])
        P = pooled(preds, 'clean', config, 'test')
        Pt = P.copy()
        Pt[:, MPOX] = np.where(flags, 1.0, 0.0)  # binary() with thr=0.5 reads the per-fold decision
        for label, b in (('argmax', binary(y, P)), (f'val threshold for {SENS_TARGET:.0%} sens.', binary(y, Pt, 0.5))):
            t4.append({'config': config, 'operating point': label,
                       **{k: f'{v * 100:.1f}' for k, v in b.items() if k != 'auc'},
                       'mpox auc': f'{binary(y, P)["auc"] * 100:.2f}'})
    tables['table4_mpox_screening'] = pd.DataFrame(t4)

    # ---- table 5: selective prediction (clean protocol)
    t5 = []
    for config in configs:
        s = selective(pooled(preds, 'clean', config, 'test'), y)
        t5.append({'config': config, **{k: f'{v * 100:.1f}' if k != 'AURC' else f'{v:.4f}' for k, v in s.items()}})
    tables['table5_selective'] = pd.DataFrame(t5)

    titles = {'table1_performance': 'pooled test predictions of the 5 folds (%); McNemar against the main model',
              'table2_conformal': 'split conformal prediction (LAC), calibrated on each fold\'s validation split',
              'table3_confusion_main': f'confusion matrix of {MAIN} (clean protocol, pooled)',
              'table4_mpox_screening': 'mpox vs rest (clean protocol, pooled; threshold chosen per fold on validation)',
              'table5_selective': 'selective prediction (clean protocol): accuracy % after referring the least confident'}
    for name, t in tables.items():
        print(f'\n=== Pilot 4, {name.split("_")[0]}: {titles[name]} ===')
        print(t.to_string(index=False))
        t.to_csv(os.path.join(OUT, f'pilot4_{name}.csv'), index=False)
    rows = [dict(protocol=pr_, config=c, fold=f, path_order=i, y_true=yt, **{f'p_{k}': v for k, v in zip(CLASSES, p)})
            for (pr_, c, f), d in preds.items() for i, (yt, p) in enumerate(zip(d['y_test'], d['test']))]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pilot4_test_predictions.csv'), index=False)
    figures(preds, cm)
    print(f'\nFiles in {OUT}: pilot4_table*.csv, pilot4_test_predictions.csv, '
          'fig_reliability.png, fig_risk_coverage.png, fig_confusion.png, fig_class_coverage.png')


main()
