"""
Final experiments of the paper (MSLD v2.0 with its official patient-independent 5 folds; external test sets MSID and
MCSI). One script for every number of the paper: it merges pilots 3-8 and writes every table as CSV (and prints it),
plus the per-image predictions from which the figures are drawn.

  S1  data: duplicate screening of the folds (MD5 and 16x16 aHash, Hamming <= 10)
  S2  benchmark: frozen features of 9 foundation backbones + linear probe (C and temperature by inner 5-fold CV inside
      the training split), fusions, optional LoRA fine-tuning; corrected resampled t-test and McNemar, Holm-adjusted
  S3  reliability of the main model: confusion matrix, mpox operating points, selective prediction, split conformal
  S4  zero-shot clinical concepts (MONET + SigLIP2 text encoders): profiles, mpox vs chickenpox, error profiles,
      concept-bottleneck and two-stage models
  S5  shortcut audit: objective image attributes, skin tone (ITA), masked images, attribute-only models
  S6  external validation on MSID and MCSI after removing copies of MSLD images; ITA checked against the MCSI authors
  S7  skin tone: reweighting, grey-scale, tone augmentation; AUC within skin-tone groups, sensitivity at the argmax
      and at a common specificity

Kaggle: inputs = MSLD v2.0, MCSI, MSID (and, to reuse their features, the notebook that ran pilots 3-8);
Accelerator = GPU (T4), Internet = On. Paste the whole file into ONE cell (the notebook must contain only this cell),
"Save Version -> Save & Run All". Expected time: about 1-1.5 h when the pilot features are reused, about 2 h without
them; RUN_LORA = True adds about 3 h.
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
OUT = '/kaggle/working/final'
FOLDS_DIR = None            # None = search /kaggle/input
EXTERNAL_DIRS = ['/kaggle/input/datasets/talhakabakus/mcsi-mpox-close-skin-images',      # MCSI (Zenodo v2)
                 '/kaggle/input/datasets/talhakabakus/msid-monkeypox-skin-images-dataset']  # MSID (Mendeley v6)
# features are deterministic, so files of the pilots are reused when they cover every image ([] = recompute all)
FEATURE_DIRS = ['/kaggle/working', '/kaggle/input']
PREVIOUS_OUT = ['/kaggle/working/pilot8', '/kaggle/working/pilot7', '/kaggle/working/pilot6']
RUN_LORA = False            # LoRA fine-tuning ablation (about 3 h on a T4)
LORA_BACKBONES = ['monet', 'siglip2_so400m']
LORA_SEEDS = [0]
LORA = dict(rank=16, alpha=32, dropout=0.1, targets=('q_proj', 'k_proj', 'v_proj', 'out_proj'),
            lr=2e-4, head_lr=1e-3, weight_decay=0.01, batch=8, accumulate=2, max_epochs=20, patience=5,
            warmup_epochs=1, label_smoothing=0.1)
WORKERS = 4

MAIN = 'LP concat monet+siglip2_so400m'
SINGLE = ['densenet201', 'dinov2_vitl14', 'dinov3_vitl16', 'biomedclip', 'siglip_so400m', 'siglip2_so400m',
          'monet', 'dermlip', 'dermlip_panderm']
FUSIONS = [['monet', 'siglip_so400m'], ['monet', 'siglip2_so400m'], ['dermlip', 'siglip2_so400m']]
CONFIGS = {**{f'LP {n}': [n] for n in SINGLE}, **{f'LP concat {"+".join(f)}': f for f in FUSIONS}}
REQUIRED = ['monet', 'siglip2_so400m']
EXTERNAL_CONFIGS = ['LP densenet201', 'LP monet', 'LP siglip2_so400m', MAIN]
CONFORMAL_CONFIGS = ['LP densenet201', 'LP siglip2_so400m', MAIN]
ALPHAS = [0.10, 0.05]
SENS_TARGET = 0.95
SPEC_TARGET = 0.95
REFER = [0.0, 0.1, 0.2, 0.3]

CONCEPT_MODELS = {'monet': 'clip', 'siglip2_so400m': 'siglip'}
PAIR = ('Chickenpox', 'Mpox')
TEMPLATES = ['a clinical photo of skin with {}', 'a dermatology image showing {}', 'this is a skin image of {}']
NEUTRAL = ['a clinical photo of skin', 'a dermatology image', 'this is a skin image']
CONCEPTS = {  # name: (group, synonyms)
    'umbilicated lesions':      ('morphology', ['umbilicated papules', 'umbilicated pustules',
                                                'lesions with a central depression']),
    'deep-seated pustules':     ('morphology', ['deep-seated firm pustules', 'firm well-circumscribed pustules']),
    'vesicles':                 ('morphology', ['vesicles', 'small clear fluid-filled blisters']),
    'dewdrop vesicles':         ('morphology', ['vesicles on a red base', 'dewdrop on a rose petal vesicles']),
    'pustules':                 ('morphology', ['pustules', 'pus-filled bumps']),
    'papules':                  ('morphology', ['papules', 'small raised bumps']),
    'macules':                  ('morphology', ['macules', 'flat red spots']),
    'crusts':                   ('morphology', ['crusts', 'scabs', 'crusted lesions']),
    'black eschar':             ('morphology', ['a black eschar', 'a necrotic black crust']),
    'ulcer':                    ('morphology', ['an ulcer', 'an ulcerated lesion']),
    'erythema':                 ('morphology', ['erythema', 'redness of the skin']),
    'maculopapular rash':       ('morphology', ['a maculopapular rash', 'a blotchy red rash']),
    'confluent rash':           ('morphology', ['a confluent rash', 'merging red patches']),
    'hemorrhagic lesions':      ('morphology', ['hemorrhagic lesions', 'bleeding lesions']),
    'scars':                    ('morphology', ['scars', 'healed lesions']),
    'lesions at same stage':    ('stage/extent', ['lesions all at the same stage', 'uniform lesions of the same size']),
    'lesions at different stages': ('stage/extent', ['lesions at different stages',
                                                     'papules, vesicles and crusts at the same time']),
    'single or few lesions':    ('stage/extent', ['a single lesion', 'a few isolated lesions']),
    'many lesions':             ('stage/extent', ['many scattered lesions', 'a widespread rash']),
    'face':                     ('body site', ['lesions on the face']),
    'hands or palms':           ('body site', ['lesions on the hands', 'lesions on the palms']),
    'feet or soles':            ('body site', ['lesions on the feet', 'lesions on the soles of the feet']),
    'mouth':                    ('body site', ['lesions in or around the mouth']),
    'trunk':                    ('body site', ['lesions on the trunk', 'lesions on the chest and back']),
    'genital area':             ('body site', ['lesions in the genital or perianal area']),
    'normal skin':              ('normal', ['normal healthy skin', 'skin without any lesions']),
    'child':                    ('attribute', ['a child', 'an infant']),
    'dark skin tone':           ('attribute', ['dark skin', 'skin of color']),
    'light skin tone':          ('attribute', ['light skin', 'fair skin']),
    'text or watermark':        ('attribute', ['text or a watermark on the image']),
    'blurry photo':             ('attribute', ['a blurry low-quality photo']),
    'extreme close-up':         ('attribute', ['an extreme close-up']),
}

MASKS = ['centre masked', 'skin masked', 'background masked']
CENTRE_AREA = 0.5           # fraction of the image area grey-filled by 'centre masked'
GREY = (124, 116, 104)      # ImageNet mean colour
ITA_GROUPS = [(41, 'light (ITA > 41)'), (10, 'intermediate (10-41)'), (-999, 'dark (ITA <= 10)')]
METHODS = {  # skin-tone methods: (image variant of the features, augmented variants added to training, reweighting)
    'baseline': ('original', [], False),
    'reweighted': ('original', [], True),
    'grey + equalised': ('grey', [], False),
    'tone aug +-10': ('original', ['lighter10', 'darker10'], False),
    'tone aug +-15': ('original', ['lighter', 'darker'], False),
    'tone aug +-25': ('original', ['lighter25', 'darker25'], False),
    'tone aug +-15 + reweighted': ('original', ['lighter', 'darker'], True)}
TONE_SHIFT = {  # L* shift and b* factor (CIE Lab), whole image; 'lighter'/'darker' = +-15
    'lighter10': (+10.0, 0.90), 'darker10': (-10.0, 1.13), 'lighter': (+15.0, 0.85), 'darker': (-15.0, 1.20),
    'lighter25': (+25.0, 0.75), 'darker25': (-25.0, 1.33)}

NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5
BOOTSTRAP = 2000
BATCH = 32
DPI = 300
CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
MPOX = CLASSES.index('Mpox')
CHICKENPOX = CLASSES.index('Chickenpox')
ALIASES = {'monkeypox': 'Mpox', 'monkey pox': 'Mpox', 'mpox': 'Mpox', 'normal': 'Healthy', 'normal skin': 'Healthy',
           'healthy skin': 'Healthy', 'chicken pox': 'Chickenpox', 'varicella': 'Chickenpox', 'cow pox': 'Cowpox',
           'hand foot mouth': 'HFMD', 'hand foot and mouth': 'HFMD', 'hand foot and mouth disease': 'HFMD',
           'hfmd': 'HFMD', 'rubeola': 'Measles'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif')
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)


# shared with pilots 3-4
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
    fold['report'] = report
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




def full_logits(clf, X):
    """decision function over all classes (classes absent from training get -1e4); binary models give [0, d]"""
    d = clf.decision_function(X)
    if d.ndim == 1:
        d = np.stack([np.zeros_like(d), d], 1)
    out = np.full((len(X), len(CLASSES)), -1e4)
    out[:, clf.classes_] = d
    return out



# =============================================================================
# Pilot 6: data of the external datasets
# =============================================================================
INPUT_ROOT = '/kaggle/input'
EXT_ROOTS = []   # folders of the external datasets (searched for skin-tone label files)
GENERIC_DIRS = {'images', 'image', 'img', 'imgs', 'data', 'dataset', 'original', 'original images', 'all', 'photos'}


def match_class(name):
    """class of a folder name, tolerant to case, '_', '-', ',' and a trailing 'images' (shared with pilots 1-5)"""
    name = re.sub(r'[\s_\-,]+', ' ', name.strip().lower()).strip()
    name = re.sub(r'\s*(images?|imgs?|photos?)$', '', name).strip()
    return next((c for c in CLASSES if c.lower() == name), ALIASES.get(name))


SPECIAL_ROOTS = ('datasets', 'competitions', 'models', 'notebooks', 'kernels')


def dataset_key(path):
    parts = os.path.relpath(path, INPUT_ROOT).split(os.sep)
    n = 3 if parts[0] in SPECIAL_ROOTS else 1
    return os.path.join(INPUT_ROOT, *parts[:n])


def class_images(root):
    """(path, folder name) of every image in a folder that directly contains images; 'aug' folders are skipped"""
    rows = []
    for d, sub, files in os.walk(root):
        sub[:] = sorted(s for s in sub if 'aug' not in s.lower())
        imgs = sorted(f for f in files if f.lower().endswith(IMAGE_EXTENSIONS))
        if imgs and d != root:
            rows += [(os.path.join(d, f), os.path.basename(d)) for f in imgs]
    return rows


def external_datasets(msld_dir):
    """{dataset name: [(path, label or -1, class name)]}; a dataset needs at least two folders named like MSLD
    classes. Folders with other names (e.g. acne) are kept as non-mpox classes outside MSLD."""
    if EXTERNAL_DIRS:
        roots = list(EXTERNAL_DIRS)
    else:
        roots = set()
        for d, sub, _ in os.walk(INPUT_ROOT):
            rel = os.path.relpath(d, INPUT_ROOT)
            depth = 0 if rel == '.' else rel.count(os.sep) + 1
            want = 3 if rel.split(os.sep)[0] in SPECIAL_ROOTS else 1
            if depth >= want:
                roots.add(d)
                sub[:] = []
        roots = sorted(roots - {dataset_key(msld_dir)})
    out, EXT_ROOTS[:] = {}, []
    for root in roots:
        if not os.path.isdir(root):
            print(f'  WARNING: {root} does not exist (check Add Input and the path in EXTERNAL_DIRS)')
            continue
        EXT_ROOTS.append(root)
        search = [root]
        rows = class_images(root)
        zips = sorted(glob.glob(os.path.join(root, '**', '*.zip'), recursive=True))
        if zips and not any(match_class(n) for _, n in rows):   # images still packed in a zip inside the dataset
            import zipfile
            dest = os.path.join(OUT, 'unzipped', os.path.basename(root.rstrip(os.sep)))
            for z in zips:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest)
            print(f'  {root}: unpacked {len(zips)} zip file(s) to {dest}')
            rows = class_images(dest)
            EXT_ROOTS.append(dest)
            search.append(dest)
        if not rows:
            print(f'  WARNING: no images found in {root}')
        if rows and len({match_class(n) for _, n in rows} - {None}) < 2:   # labels in a CSV instead of folders
            got = csv_labels(search, rows)
            if got:
                rows = got
            else:
                print(f'  WARNING: {root}: no class folders and no label CSV recognised; folders: '
                      f'{sorted({n for _, n in rows})[:10]}')
        known = {n for _, n in rows if match_class(n)}
        if len({match_class(n) for n in known}) < 2:
            continue
        keep = []
        for p, n in rows:
            c = match_class(n)
            low = n.strip().lower()
            if c is None and (low in SPLITS or low in GENERIC_DIRS or FOLD_RE.match(low)):
                continue
            keep.append((p, CLASSES.index(c) if c else -1, c or n.strip().title()))
        out[os.path.basename(root.rstrip(os.sep))] = keep
    return out


def csv_labels(dirs, rows):
    """(path, class name) from a CSV that lists the image files and their diagnosis (e.g. MCSI metadata.csv);
    severity suffixes such as 'acne_l2' are merged ('acne')"""
    names = {os.path.basename(p).lower(): p for p, _ in rows}
    for f in sorted({x for r in dirs for x in glob.glob(os.path.join(r, '**', '*.csv'), recursive=True)}):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        for fc in t.columns:
            base = t[fc].map(lambda s: os.path.basename(str(s)).lower())
            if base.isin(names).mean() < 0.5:
                continue
            for lc in t.columns:
                if lc == fc or t[lc].nunique() > 20:
                    continue
                vals = t[lc].map(lambda s: re.sub(r'[\s_-]*(l|lvl|level)\s*\d+$', '', str(s).strip(), flags=re.I))
                if len({match_class(v) for v in vals} - {None}) >= 2:
                    print(f'  labels from {f} (file column {fc!r}, label column {lc!r})')
                    out = [(names[bb], v) for bb, v in zip(base, vals) if bb in names]
                    known = {v.lower() for v in vals}
                    extra = []
                    for bb, p in names.items():   # files missing from the CSV (e.g. MCSI acne_lvl1_1.png vs acne_l1)
                        if bb in set(base):
                            continue
                        stem = re.sub(r'[\s_-]*\d+$', '', os.path.splitext(bb)[0])
                        stem = re.sub(r'[\s_-]*(l|lvl|level)\s*\d+$', '', stem, flags=re.I)
                        if stem in known:
                            extra.append((p, stem))
                    if extra:
                        print(f'  {len(extra)} images not listed in {os.path.basename(f)} labelled from their file '
                              f'name: {sorted({s for _, s in extra})}')
                    return out + extra
    return None


def tone_labels(root_files):
    """skin-tone labels shipped with a dataset (e.g. MCSI v2 'skin_tone_labels.csv': Filename, ITA, CLS, Type), matched
    by unique file name. Returns {file name: 'light'/'dark'} and {file name: ITA of the dataset authors}"""
    names = pd.Series([os.path.basename(p).lower() for p in root_files])
    unique = set(names[~names.duplicated(keep=False)])
    labels, itas = {}, {}
    for f in sorted({x for r in EXT_ROOTS for x in glob.glob(os.path.join(r, '**', '*.csv'), recursive=True)}):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        low = lambda s: str(s).strip().lower()
        tone = [c for c in t.columns if t[c].nunique() <= 5 and set(t[c].map(low)) <= {'light', 'dark', 'nan'}
                and t[c].map(low).isin(['light', 'dark']).any()]
        ita = [c for c in t.columns if low(c) == 'ita']
        if not tone and not ita:
            continue
        for c in t.columns:
            base = t[c].map(lambda s: os.path.basename(str(s)).lower())
            if base.isin(unique).mean() > 0.5:
                for i, b in enumerate(base):
                    if b in unique:
                        if tone:
                            labels[b] = low(t[tone[0]].iloc[i])
                        if ita:
                            itas[b] = float(t[ita[0]].iloc[i])
                print(f'  skin-tone labels from {f} ({len(base[base.isin(unique)])} images)')
                break
    return labels, itas


# =============================================================================
# Pilot 6: objective image attributes and masks
# =============================================================================
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
FILE_ATTRS = ['width', 'height', 'aspect', 'megapixels', 'bytes per pixel', 'jpeg quality', 'png']
PIXEL_ATTRS = ['brightness', 'saturation', 'sharpness', 'skin fraction', 'ITA', 'border brightness', 'border std']
ATTRS = FILE_ATTRS + PIXEL_ATTRS


def rgb_to_lab(a):
    """sRGB in [0, 1] -> CIE L*a*b* (D65)"""
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])


def skin_mask(a):
    """skin-coloured pixels, YCrCb rule (Chai & Ngan); crusts, eschars and deep-red lesions may fall outside it"""
    a = a.astype(np.float32)
    Y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    Cr, Cb = (a[..., 0] - Y) * 0.713 + 128, (a[..., 2] - Y) * 0.564 + 128
    return (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127) & (Y > 40)


def jpeg_quality(im):
    q = getattr(im, 'quantization', None)
    if not q:
        return np.nan
    t = np.array(list(q[0] if 0 in q else next(iter(q.values()))), dtype=float)
    s = 100 * t.sum() / STD_LUMA.sum()
    return float(np.clip((200 - s) / 2 if s <= 100 else 5000 / s, 1, 100))


def attributes(path):
    im = Image.open(path)
    w, h = im.size
    fmt, qual = (im.format or '').upper(), jpeg_quality(im)
    rgb = im.convert('RGB')
    rgb.thumbnail((512, 512))
    a = np.asarray(rgb)
    L, _, B = rgb_to_lab(a / 255.0)
    f = a / 255.0
    mx, mn = f.max(2), f.min(2)
    lap = -4 * L[1:-1, 1:-1] + L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:]
    m = skin_mask(a)
    bw = max(1, int(0.05 * min(L.shape)))
    ring = np.ones(L.shape, bool)
    ring[bw:-bw, bw:-bw] = False
    return {'width': w, 'height': h, 'aspect': w / h, 'megapixels': w * h / 1e6,
            'bytes per pixel': os.path.getsize(path) / (w * h), 'jpeg quality': qual, 'png': float(fmt == 'PNG'),
            'brightness': L.mean(), 'saturation': np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean(),
            'sharpness': lap.var(), 'skin fraction': m.mean(),
            'ITA': float(np.median(np.degrees(np.arctan2(L[m] - 50, B[m])))) if m.mean() >= 0.02 else np.nan,
            'border brightness': L[ring].mean(), 'border std': L[ring].std()}


def ita_group(v):
    if not np.isfinite(v):
        return 'unknown (< 2% skin pixels)'
    return next(name for lo, name in ITA_GROUPS if v > lo)


def masked(path, variant):
    from scipy.ndimage import binary_dilation
    a = np.array(Image.open(path).convert('RGB'))
    if variant == 'centre masked':
        h, w = a.shape[:2]
        s = math.sqrt(CENTRE_AREA)
        dh, dw = int(round(h * (1 - s) / 2)), int(round(w * (1 - s) / 2))
        a[dh:h - dh, dw:w - dw] = GREY
    elif variant == 'skin masked':
        a[binary_dilation(skin_mask(a), iterations=3)] = GREY
    elif variant == 'background masked':
        a[~skin_mask(a)] = GREY
    else:
        raise ValueError(variant)
    return Image.fromarray(a)


def features_for(name, sets, device):
    """{tag: {key: feature}} for {tag: [(key, image loader)]}; the model is loaded once, every set cached on disk"""
    import torch
    out, todo = {}, {}
    for tag, items in sets.items():
        fname = f'feat6_{name}_{re.sub(r"\W+", "_", tag)}.npz'
        cache = os.path.join(OUT, fname)
        cached = None
        for d in [OUT] + PREVIOUS_OUT:   # reuse features of earlier pilots when they cover every image
            if cached is None and os.path.exists(os.path.join(d, fname)):
                z = np.load(os.path.join(d, fname))
                cached = dict(zip(z['keys'], z['feats']))
                if not {k for k, _ in items} <= set(cached):
                    cached = None
        if cached is not None and {k for k, _ in items} <= set(cached):
            out[tag] = cached
        elif items:
            todo[tag] = (dict(items), cache)
    if not todo:
        return out
    try:
        fn = load_extractor(name, device)
    except Exception as e:
        if name in REQUIRED:
            raise
        print(f'  {name:16s} SKIPPED: {type(e).__name__}: {str(e).strip()[:200]}')
        return out
    try:
        for tag, (items, cache) in todo.items():
            t0, keys, feats = time.time(), list(items), {}
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                for i in range(0, len(keys), BATCH):
                    b = keys[i:i + BATCH]
                    feats.update(zip(b, fn([items[k]() for k in b]).float().cpu().numpy()))
            np.savez(cache, keys=np.array(keys), feats=np.stack([feats[k] for k in keys]))
            out[tag] = feats
            print(f'  {name:16s} {tag}: {len(keys)} images in {time.time() - t0:.0f}s', flush=True)
    finally:
        fn = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# =============================================================================
# Pilot 6: models and metrics
# =============================================================================
def attr_probs(Xtr, ytr, Xte):
    """gradient boosting on the image attributes only (missing values allowed), fixed hyper-parameters"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight='balanced',
                                         random_state=0).fit(Xtr, ytr)
    out = np.zeros((len(Xte), len(CLASSES)))
    out[:, clf.classes_] = clf.predict_proba(Xte)
    return out


def probe_full(X, y):
    """as probe_clean, on all MSLD images: C and temperature by inner cross-validation. Returns a predictor of
    calibrated probabilities and the calibrated out-of-fold probabilities (for the operating threshold)"""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    splits = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=0).split(X, y))

    def oof_part(C, tr, te):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=0)
        clf.fit(sc.transform(X[tr]), y[tr])
        return C, te, full_logits(clf, sc.transform(X[te]))

    parts = Parallel(n_jobs=-1)(delayed(oof_part)(C, tr, te) for C in C_GRID for tr, te in splits)
    oof = {C: np.zeros((len(y), len(CLASSES))) for C in C_GRID}
    for C, te, lg in parts:
        oof[C][te] = lg
    score = {C: balanced_accuracy_score(y, oof[C].argmax(1)) for C in C_GRID}
    C = max(C_GRID, key=lambda c: (score[c], -c))
    T = fit_temperature(oof[C], y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=0).fit(sc.transform(X), y)
    return (lambda Xn: softmax(full_logits(clf, sc.transform(Xn)) / T)), softmax(oof[C] / T), {'C': C, 'T': T}


def pair_scores(y, p):
    m = np.isin(y, [CHICKENPOX, MPOX])
    return (y[m] == MPOX), p[m, MPOX] / (p[m, MPOX] + p[m, CHICKENPOX] + 1e-12)


def safe_auc(pos, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(pos, s) if 0 < pos.sum() < len(pos) else np.nan


def boot_ci(fn, n, rng):
    vals = []
    for _ in range(BOOTSTRAP):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def evaluate(y, p, rng, thr=None):
    """y: MSLD class index or -1 (class outside MSLD, counts as non-mpox). Multi-class metrics use the images of MSLD
    classes with the prediction restricted to the classes present; mpox metrics use every image"""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    known = y >= 0
    present = sorted(set(y[known]))
    yk = y[known]
    pk = p[known][:, present] / p[known][:, present].sum(1, keepdims=True)
    pred = np.array(present)[pk.argmax(1)]
    correct = pred == yk
    pos, flag = y == MPOX, p.argmax(1) == MPOX
    pp, ps = pair_scores(y, p)
    div = lambda a, b: a / b * 100 if b else np.nan
    lo, hi = boot_ci(lambda i: correct[i].mean(), len(correct), rng)
    auc = safe_auc(pos, p[:, MPOX])
    alo, ahi = boot_ci(lambda i: safe_auc(pos[i], p[i, MPOX]), len(y), rng)
    r = {'n': int(len(y)), 'acc [95% CI]': f'{correct.mean() * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]',
         'bal_acc': f'{balanced_accuracy_score(yk, pred) * 100:.1f}',
         'f1': f'{f1_score(yk, pred, labels=present, average="macro", zero_division=0) * 100:.1f}',
         'mpox sens/spec': f'{div((flag & pos).sum(), pos.sum()):.1f}/{div((~flag & ~pos).sum(), (~pos).sum()):.1f}',
         'mpox AUC [95% CI]': f'{auc:.3f} [{alo:.3f}, {ahi:.3f}]' if np.isfinite(auc) else '',
         'Mpox-vs-Chickenpox AUC': f'{safe_auc(pp, ps):.3f}' if len(pp) else '',
         'ece': f'{ece(pk, np.searchsorted(present, yk)) * 100:.1f}'}
    if thr is not None:
        f2 = p[:, MPOX] >= thr
        r['sens/spec at MSLD 95%-sens threshold'] = (f'{div((f2 & pos).sum(), pos.sum()):.1f}/'
                                                     f'{div((~f2 & ~pos).sum(), (~pos).sum()):.1f}')
    return r, correct


def figures6(ext_conf, ita_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    if ext_conf:
        fig, axes = plt.subplots(1, len(ext_conf), figsize=(3.4 * len(ext_conf), 3.0), squeeze=False)
        for ax, (name, t) in zip(axes[0], ext_conf.items()):
            M = t[CLASSES].to_numpy(float)
            R = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax.imshow(R, cmap='Blues', vmin=0, vmax=1)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if M[i, j]:
                        ax.text(j, i, int(M[i, j]), ha='center', va='center', fontsize=7,
                                color='white' if R[i, j] > 0.6 else '#222222')
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha='right')
            ax.set_yticks(range(len(t)), [f'{c} (n={n})' for c, n in zip(t['true class'], t['n'])])
            ax.set(title=name, xlabel='predicted (model trained on MSLD)')
            ax.spines[:].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig_external_confusion.png'), dpi=DPI)
        plt.close(fig)
    d = pd.DataFrame(ita_rows).dropna(subset=['ITA'])
    sources = list(dict.fromkeys(d['source']))
    classes = [c for c in CLASSES if c in set(d['class'])] + sorted(set(d['class']) - set(CLASSES))
    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * len(classes) * len(sources) / 2 + 2), 3.0))
    width = 0.8 / len(sources)
    shades = ['#555555', '#E69F00', '#56B4E9', '#009E73', '#CC79A7']
    for k, s in enumerate(sources):
        data = [d[(d['source'] == s) & (d['class'] == c)]['ITA'].to_numpy() for c in classes]
        pos = [i + (k - (len(sources) - 1) / 2) * width for i in range(len(classes))]
        keep = [(x, p) for x, p in zip(data, pos) if len(x)]
        if keep:
            bp = ax.boxplot([x for x, _ in keep], positions=[p for _, p in keep], widths=width * 0.9,
                            patch_artist=True, showfliers=False, medianprops={'color': '#222222'})
            for b in bp['boxes']:
                b.set(facecolor=shades[k % len(shades)], alpha=0.7)
        ax.plot([], [], 's', color=shades[k % len(shades)], label=s)
    for lo, _ in ITA_GROUPS[:-1]:
        ax.axhline(lo, color='#999999', lw=0.6, ls='--')
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha='right')
    ax.set(ylabel='skin tone, ITA (degrees)')
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_ita_by_class.png'), dpi=DPI)
    plt.close(fig)


# =============================================================================
# Pilot 7: skin-tone variants of the images
# =============================================================================
def lab_to_rgb(L, A, B):
    fy = (L + 16) / 116
    f = np.stack([fy + A / 500, fy, fy - B / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    lin = np.clip(xyz @ np.linalg.inv(M).T, 0, 1)
    return np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin)


def variant(path, kind):
    from PIL import ImageOps
    im = Image.open(path).convert('RGB')
    if kind == 'original':
        return im
    if kind == 'grey':
        return ImageOps.equalize(im.convert('L')).convert('RGB')
    dL, fb = TONE_SHIFT[kind]
    L, A, B = rgb_to_lab(np.asarray(im) / 255.0)
    rgb = lab_to_rgb(np.clip(L + dL, 0, 100), A, B * fb)
    return Image.fromarray((rgb * 255).round().astype(np.uint8))


def probe7(X, y, tests, aug=(), w=None):
    """linear probe as probe_clean (C and temperature by inner 5-fold CV inside the training images), with optional
    augmented copies of the training images (kept in the same inner fold as their original) and sample weights.
    Returns calibrated probabilities for every array in tests and the calibrated out-of-fold probabilities"""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    splits = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=0).split(X, y))

    def fit(C, idx):
        Xs = np.concatenate([X[idx]] + [a[idx] for a in aug])
        ys = np.tile(y[idx], 1 + len(aug))
        ws = None if w is None else np.tile(w[idx], 1 + len(aug))
        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(C=C, max_iter=5000, class_weight=None if w is not None else 'balanced',
                                 random_state=0).fit(sc.transform(Xs), ys, sample_weight=ws)
        return sc, clf

    def oof_part(C, tr, te):
        sc, clf = fit(C, tr)
        return C, te, full_logits(clf, sc.transform(X[te]))

    parts = Parallel(n_jobs=-1)(delayed(oof_part)(C, tr, te) for C in C_GRID for tr, te in splits)
    oof = {C: np.zeros((len(y), len(CLASSES))) for C in C_GRID}
    for C, te, lg in parts:
        oof[C][te] = lg
    score = {C: balanced_accuracy_score(y, oof[C].argmax(1)) for C in C_GRID}
    C = max(C_GRID, key=lambda c: (score[c], -c))
    T = fit_temperature(oof[C], y)
    sc, clf = fit(C, np.arange(len(y)))
    return [softmax(full_logits(clf, sc.transform(t)) / T) for t in tests], softmax(oof[C] / T)


def tone_weights(y, groups):
    """every (class x skin-tone group) cell gets the same total weight; mean weight 1"""
    cell = np.array([f'{a}|{b}' for a, b in zip(y, groups)])
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.mean()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k, n):
    p, lo, hi = wilson(k, n)
    return f'{p * 100:.1f} [{lo * 100:.0f}-{hi * 100:.0f}] ({k}/{n})' if n else ''


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
# statistics and concepts
# =============================================================================
def corrected_ttest(a, b, ratio):
    from scipy import stats
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * var)
    return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)


def text_embeddings(name, prompts, device):
    """L2-normalised text embeddings in the image space of the backbone whose features pilot 3 saved"""
    import torch
    from transformers import AutoModel, AutoProcessor
    mid = PANEL[name][2]
    model = AutoModel.from_pretrained(mid).eval().to(device)
    proc = AutoProcessor.from_pretrained(mid)
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            batch = [p.lower() for p in prompts[i:i + 64]]
            if CONCEPT_MODELS[name] == 'siglip':   # SigLIP was trained with 64-token padded text
                inp = proc(text=batch, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            else:
                inp = proc(text=batch, padding=True, truncation=True, return_tensors='pt')
            inp = {k: v.to(device) for k, v in inp.items() if k in ('input_ids', 'attention_mask')}
            out.append(as_tensor(model.get_text_features(**inp)).float().cpu().numpy())
    del model
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def concept_scores(name, feats, keys, device):
    """raw concept scores (images x concepts): cos(image, mean concept prompt) - cos(image, mean neutral prompt)"""
    prompts, owner = [], []
    for c, (_, syns) in CONCEPTS.items():
        for s in syns:
            for t in TEMPLATES:
                prompts.append(t.format(s))
                owner.append(c)
    prompts += NEUTRAL
    owner += ['__neutral__'] * len(NEUTRAL)
    E = text_embeddings(name, prompts, device)
    owner = np.array(owner)

    def centroid(c):
        v = E[owner == c].mean(0)
        return v / np.linalg.norm(v)
    X = np.stack([feats[k] for k in keys]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    neutral = X @ centroid('__neutral__')
    S = np.stack([X @ centroid(c) - neutral for c in CONCEPTS], 1)
    return S - S.mean(1, keepdims=True)  # relative to the image's other concepts: removes the shared 'lesion' factor


def auc_ci(score, pos, rng):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(pos, score)
    boot = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(pos), len(pos))
        if pos[i].any() and (~pos[i]).any():
            boot.append(roc_auc_score(pos[i], score[i]))
    return a, np.percentile(boot, 2.5), np.percentile(boot, 97.5)



def paired_auc(pos, s_a, s_b, rng):
    """AUC of a and b on the same images, and the bootstrap 95% CI and two-sided p of b - a"""
    a, b = safe_auc(pos, s_a), safe_auc(pos, s_b)
    d = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(pos), len(pos))
        if 0 < pos[i].sum() < len(i):
            d.append(safe_auc(pos[i], s_b[i]) - safe_auc(pos[i], s_a[i]))
    d = np.array(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
    lo, hi = (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) else (np.nan, np.nan)
    return a, b, lo, hi, p



# =============================================================================
# Final: helpers
# =============================================================================
TABLES, TITLES = {}, {}


def add_table(name, df, title):
    TABLES[name], TITLES[name] = df, title
    print(f'\n=== {name}: {title} ===')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, f'{name}.csv'), index=False)


def ci_text(v, lo, hi, scale=100, digits=1):
    return f'{v * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]'


def pooled_eval(y, p, rng):
    """accuracy with bootstrap CI and the usual metrics of pooled test predictions"""
    correct = p.argmax(1) == y
    lo, hi = boot_ci(lambda i: correct[i].mean(), len(y), rng)
    m, b = metrics(y, p), binary(y, p)
    return correct, {'acc pooled [95% CI]': ci_text(correct.mean(), lo, hi),
                     'bal_acc': f'{m["balanced_accuracy"] * 100:.2f}', 'f1': f'{m["f1_macro"] * 100:.2f}',
                     'mcc': f'{m["mcc"] * 100:.2f}', 'auc macro': f'{m["auc_macro_ovr"]:.3f}',
                     'ece': f'{m["ece"] * 100:.1f}',
                     'mpox sens/spec': f'{b["sensitivity"] * 100:.1f}/{b["specificity"] * 100:.1f}',
                     'mpox AUC': f'{b["auc"]:.3f}'}


def short(lab):
    return lab.split(' (')[0]


# =============================================================================
# Final: main
# =============================================================================
def main():
    ensure_packages()
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    from sklearn.preprocessing import StandardScaler
    os.makedirs(OUT, exist_ok=True)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.max_rows', 400)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    T0 = time.time()
    rng = np.random.default_rng(0)

    # =========================================================================
    # S1 data
    # =========================================================================
    print('\n##### S1 data #####')
    folds_dir = FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True)
    folds = [screen(f) for f in discover_folds(folds_dir)]
    fold_names = [f['name'] for f in folds]
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})
    label_of = {md5(p): c for f in folds for s in ('train', 'val', 'test') for p, c in f[s]}
    path_of = {md5(p): p for p in originals}
    keys = sorted(label_of)
    labels = np.array([label_of[k] for k in keys])
    row = {k: i for i, k in enumerate(keys)}
    t = []
    for f in folds:
        r = f['report']
        t.append({'fold': f['name'], **{f'n {s}': r[f'n_{s}'] for s in ('train', 'val', 'test')},
                  'val copies removed (exact/near)': f'{r["val_exact_removed"]}/{r["val_near_removed"]}',
                  'test copies removed (exact/near)': f'{r["test_exact_removed"]}/{r["test_near_removed"]}',
                  **{f'test {c}': int(sum(1 for _, y in f['test'] if y == i)) for i, c in enumerate(CLASSES)}})
    add_table('S1_folds', pd.DataFrame(t), f'official folds after duplicate screening ({len(keys)} unique originals)')

    # =========================================================================
    # S2 benchmark
    # =========================================================================
    print('\n##### S2 benchmark: features #####')
    needed = list(dict.fromkeys(b for bs in CONFIGS.values() for b in bs))
    feats = {n: f for n in needed if (f := get_features(n, originals, device)) is not None}
    configs = {c: bs for c, bs in CONFIGS.items() if all(b in feats for b in bs)}
    skipped = [c for c in CONFIGS if c not in configs]
    if skipped:
        print(f'  configurations without features (skipped): {skipped}')
    Xof = lambda bs, ks, src=None: np.concatenate([np.stack([(src or feats)[n][k] for k in ks]) for n in bs], 1)

    print('\n##### S2 benchmark: probes #####')
    preds, t0 = {}, time.time()
    fold_keys = {f['name']: {s: [md5(p) for p, _ in f[s]] for s in ('train', 'val', 'test')} for f in folds}
    for f in folds:
        ks = fold_keys[f['name']]
        y = {s: labels[[row[k] for k in ks[s]]] for s in ks}
        for c, bs in configs.items():
            raw, cva, cte, info = probe_clean(Xof(bs, ks['train']), y['train'], Xof(bs, ks['val']),
                                              Xof(bs, ks['test']))
            preds[c, f['name']] = dict(test_raw=raw, val=cva, test=cte, y_val=y['val'], y_test=y['test'],
                                       keys=ks['test'], **info)
        print(f'  {f["name"]} done ({time.time() - t0:.0f}s)', flush=True)
    if RUN_LORA:
        print('\n##### S2 benchmark: LoRA #####')
        for n in LORA_BACKBONES:
            for f in folds:
                runs = [train_lora(n, f, s, device) for s in LORA_SEEDS]
                raw, cva, cte = (np.mean([r[i] for r in runs], 0) for i in range(3))
                ks = fold_keys[f['name']]
                preds[f'LoRA {n}', f['name']] = dict(test_raw=raw, val=cva, test=cte,
                                                     y_val=labels[[row[k] for k in ks['val']]],
                                                     y_test=labels[[row[k] for k in ks['test']]], keys=ks['test'],
                                                     **runs[0][3])
                print(f'  LoRA {n} {f["name"]}: {runs[0][3]["seconds"]:.0f}s', flush=True)
    all_configs = list(dict.fromkeys(c for c, _ in preds))
    pool = lambda c, k: np.concatenate([preds[c, f][k] for f in fold_names])
    y = pool(MAIN, 'y_test')
    key_test = [k for f in fold_names for k in preds[MAIN, f]['keys']]
    ratio = np.mean([len(fold_keys[f]['test']) / len(fold_keys[f]['train']) for f in fold_names])
    acc_f = lambda c: np.array([(preds[c, f]['test'].argmax(1) == preds[c, f]['y_test']).mean() for f in fold_names])
    main_correct = pool(MAIN, 'test').argmax(1) == y
    t = []
    for c in all_configs:
        P = pool(c, 'test')
        correct, r = pooled_eval(y, P, rng)
        d, pt = corrected_ttest(acc_f(c), acc_f(MAIN), ratio)
        t.append({'config': c, 'group': PANEL[configs[c][0]][0] if c in configs and len(configs[c]) == 1 else
                  ('fusion' if c in configs else 'LoRA'),
                  'acc folds mean ± SD': f'{acc_f(c).mean() * 100:.2f} ± {acc_f(c).std(ddof=1) * 100:.2f}', **r,
                  'ece raw': f'{ece(pool(c, "test_raw"), y) * 100:.1f}',
                  '_main': c == MAIN, '_t': np.nan if c == MAIN else pt,
                  '_m': np.nan if c == MAIN else mcnemar(correct, main_correct)})
    t = pd.DataFrame(t)
    for col in ('_t', '_m'):
        ok = t[col].notna()
        t.loc[ok, col + 'h'] = holm(t.loc[ok, col])
    fmt = lambda mn, p, h: '(main)' if mn else ('identical folds' if pd.isna(p) else f'{p:.3g} (Holm {h:.3g})')
    t['corrected t-test vs main'] = [fmt(mn, p, h) for mn, p, h in zip(t['_main'], t['_t'], t['_th'])]
    t['McNemar vs main'] = [fmt(mn, p, h) for mn, p, h in zip(t['_main'], t['_m'], t['_mh'])]
    add_table('S2_benchmark', t.drop(columns=['_main', '_t', '_m', '_th', '_mh']),
              'MSLD, 5 official folds, clean protocol: per-fold accuracy mean ± SD and pooled test predictions; '
              'corrected resampled t-test (Nadeau-Bengio) and McNemar against the main model, Holm-adjusted')
    rows_out = []
    for (c, f), d in preds.items():
        for k, yt, p in zip(d['keys'], d['y_test'], d['test']):
            rows_out.append({'config': c, 'fold': f, 'md5': k, 'y': CLASSES[yt],
                             **{f'p {cl}': v for cl, v in zip(CLASSES, p)}})
    pd.DataFrame(rows_out).to_csv(os.path.join(OUT, 'predictions_msld_cv.csv'), index=False)

    # =========================================================================
    # S3 reliability of the main model
    # =========================================================================
    print('\n##### S3 reliability #####')
    P = pool(MAIN, 'test')
    cm = confusion_matrix(y, P.argmax(1), labels=range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, P.argmax(1), labels=range(len(CLASSES)), zero_division=0)
    t = pd.DataFrame(cm, columns=[f'pred {c}' for c in CLASSES])
    t.insert(0, 'true class', CLASSES)
    t['n'], t['recall %'], t['precision %'], t['f1 %'] = sup, (rc * 100).round(1), (pr * 100).round(1), \
        (f1 * 100).round(1)
    add_table('S3a_confusion_main', t, f'confusion matrix of {MAIN}, pooled test predictions')
    t = []
    for c in CONFORMAL_CONFIGS:
        if (c, fold_names[0]) not in preds:
            continue
        Pc = pool(c, 'test')
        flags = np.concatenate([preds[c, f]['test'][:, MPOX] >= threshold_for_sensitivity(
            preds[c, f]['y_val'], preds[c, f]['val'], SENS_TARGET) for f in fold_names])
        Pt = Pc.copy()
        Pt[:, MPOX] = flags.astype(float)
        for lab, b in (('argmax', binary(y, Pc)),
                       (f'threshold for {SENS_TARGET:.0%} sensitivity on validation', binary(y, Pt, 0.5))):
            t.append({'config': c, 'operating point': lab, **{k: f'{v * 100:.1f}' for k, v in b.items() if k != 'auc'},
                      'mpox AUC': f'{binary(y, Pc)["auc"]:.3f}'})
    add_table('S3b_mpox_screening', pd.DataFrame(t), 'mpox vs rest, pooled test predictions')
    t = []
    for c in CONFORMAL_CONFIGS:
        if (c, fold_names[0]) in preds:
            s = selective(pool(c, 'test'), y)
            t.append({'config': c, **{k: f'{v * 100:.1f}' if k != 'AURC' else f'{v:.4f}' for k, v in s.items()}})
    add_table('S3c_selective', pd.DataFrame(t), 'selective prediction: accuracy (%) after referring the least '
                                                 'confident cases to a clinician')
    t = []
    for c in CONFORMAL_CONFIGS:
        if (c, fold_names[0]) not in preds:
            continue
        for alpha in ALPHAS:
            for mondrian in (False, True):
                S = np.concatenate([lac_sets(preds[c, f]['val'], preds[c, f]['y_val'], preds[c, f]['test'], alpha,
                                             mondrian) for f in fold_names])
                cover, size = S[np.arange(len(y)), y], S.sum(1)
                per = [cover[y == k].mean() for k in range(len(CLASSES))]
                t.append({'config': c, 'target': f'{int(round((1 - alpha) * 100))}%',
                          'method': 'class-conditional' if mondrian else 'marginal',
                          'coverage': f'{cover.mean() * 100:.1f}', 'mpox coverage': f'{per[MPOX] * 100:.1f}',
                          'min class coverage': f'{min(per) * 100:.1f} ({CLASSES[int(np.argmin(per))]})',
                          'mean set size': f'{size.mean():.2f}', 'singletons %': f'{(size == 1).mean() * 100:.0f}'})
    add_table('S3d_conformal', pd.DataFrame(t), 'split conformal prediction (LAC), calibrated on the validation split '
                                                'of each fold')

    # =========================================================================
    # S4 concepts
    # =========================================================================
    print('\n##### S4 concepts #####')
    names = list(CONCEPTS)
    Zs = {}
    for n in CONCEPT_MODELS:
        raw = concept_scores(n, feats[n], keys, device)
        Zs[n] = (raw - raw.mean(0)) / raw.std(0)
        if device == 'cuda':
            torch.cuda.empty_cache()
    models = list(CONCEPT_MODELS)
    Z = np.mean([Zs[n] for n in models], 0)
    t = []
    for j, c in enumerate(names):
        aucs = [roc_auc_score(labels == i, Z[:, j]) for i in range(len(CLASSES))]
        t.append({'concept': c, 'group': CONCEPTS[c][0],
                  'rho monet~siglip2': f'{spearmanr(Zs[models[0]][:, j], Zs[models[1]][:, j])[0]:.2f}',
                  **{k: f'{Z[labels == i, j].mean():+.2f}' for i, k in enumerate(CLASSES)},
                  'highest in (one-vs-rest AUC)': f'{CLASSES[int(np.argmax(aucs))]} ({max(aucs):.2f})'})
    add_table('S4a_concept_profiles', pd.DataFrame(t), 'mean concept score (z, consensus of MONET and SigLIP2) per '
                                                       'class, all MSLD originals')
    a, b = CLASSES.index(PAIR[0]), CLASSES.index(PAIR[1])
    sel = np.isin(labels, [a, b])
    pos = labels[sel] == b
    t = []
    for j, c in enumerate(names):
        auc, lo, hi = auc_ci(Z[sel, j], pos, rng)
        sd = np.sqrt((Z[sel][pos, j].var(ddof=1) + Z[sel][~pos, j].var(ddof=1)) / 2)
        t.append({'concept': c, 'group': CONCEPTS[c][0], 'AUC': auc, 'AUC [95% CI]': f'{auc:.3f} [{lo:.3f}, {hi:.3f}]',
                  f'z {PAIR[0]}': f'{Z[labels == a, j].mean():+.2f}', f'z {PAIR[1]}': f'{Z[labels == b, j].mean():+.2f}',
                  "Cohen's d": f'{(Z[sel][pos, j].mean() - Z[sel][~pos, j].mean()) / sd:+.2f}',
                  **{f'AUC {n}': f'{roc_auc_score(pos, Zs[n][sel, j]):.3f}' for n in models}})
    pair_tab = pd.DataFrame(t)
    pair_tab = pair_tab.reindex((pair_tab['AUC'] - 0.5).abs().sort_values(ascending=False).index)
    add_table('S4b_concepts_mpox_vs_chickenpox', pair_tab.drop(columns='AUC'),
              f'{PAIR[1]} vs {PAIR[0]}: concept AUC (> 0.5 = higher in {PAIR[1]}), sorted by separation')
    C_all = np.concatenate([Zs[n] for n in models], 1)
    cp = {k: [] for k in ('CBM (concepts only)', 'main + concepts', 'two-stage')}
    for f in fold_names:
        ks = fold_keys[f]
        yy = {s: labels[[row[k] for k in ks[s]]] for s in ks}
        Xm = {s: Xof(CONFIGS[MAIN], ks[s]) for s in ks}
        Xc = {s: C_all[[row[k] for k in ks[s]]] for s in ks}
        cp['CBM (concepts only)'].append(probe_clean(Xc['train'], yy['train'], Xc['val'], Xc['test'])[2])
        Xmc = {s: np.concatenate([Xm[s], Xc[s]], 1) for s in ks}
        cp['main + concepts'].append(probe_clean(Xmc['train'], yy['train'], Xmc['val'], Xmc['test'])[2])
        main_te = preds[MAIN, f]['test']
        tr = np.isin(yy['train'], [a, b])
        pair_te = probe_clean(Xm['train'][tr], yy['train'][tr], Xm['val'], Xm['test'])[2]
        two = main_te.copy()
        route = np.isin(main_te.argmax(1), [a, b])
        mass = main_te[route][:, [a, b]].sum(1)
        share = pair_te[route][:, [a, b]] / pair_te[route][:, [a, b]].sum(1, keepdims=True)
        two[np.ix_(route, [a, b])] = share * mass[:, None]
        cp['two-stage'].append(two)
    t = []
    for k, p in [('main', pool(MAIN, 'test'))] + [(k, np.concatenate(v)) for k, v in cp.items()]:
        correct, r = pooled_eval(y, p, rng)
        pred = p.argmax(1)
        t.append({'model': k, 'acc pooled [95% CI]': r['acc pooled [95% CI]'], 'bal_acc': r['bal_acc'],
                  f'{PAIR[0]} recall': f'{(pred[y == a] == a).mean() * 100:.1f}',
                  f'{PAIR[1]} recall': f'{(pred[y == b] == b).mean() * 100:.1f}',
                  f'{PAIR[0]}<->{PAIR[1]} errors': int(((pred == a) & (y == b)).sum() + ((pred == b) & (y == a)).sum()),
                  'McNemar vs main': '(main)' if k == 'main' else f'p={mcnemar(correct, main_correct):.3g}'})
    add_table('S4c_concept_models', pd.DataFrame(t), 'concept-bottleneck and two-stage models, pooled test predictions')
    pred = pool(MAIN, 'test').argmax(1)
    zt = Z[[row[k] for k in key_test]]
    groups = {f'{PAIR[0]} correct': (y == a) & (pred == a), f'{PAIR[0]} -> {PAIR[1]}': (y == a) & (pred == b),
              f'{PAIR[1]} -> {PAIR[0]}': (y == b) & (pred == a), f'{PAIR[1]} correct': (y == b) & (pred == b)}
    top = pair_tab['concept'].head(10).tolist()
    t = pd.DataFrame({g: [f'{zt[m, names.index(c)].mean():+.2f}' if m.any() else '' for c in top]
                      for g, m in groups.items()}, index=top)
    t.loc['n images'] = [str(int(m.sum())) for m in groups.values()]
    add_table('S4d_concept_error_profiles', t.reset_index().rename(columns={'index': 'concept'}),
              f'mean concept z of the main model\'s {PAIR[0]}/{PAIR[1]} test images by outcome')
    pd.DataFrame(Z, columns=names).assign(md5=keys, label=[CLASSES[i] for i in labels]).to_csv(
        os.path.join(OUT, 'concept_scores_msld.csv'), index=False)

    # =========================================================================
    # S5 shortcut audit
    # =========================================================================
    print('\n##### S5 shortcut audit #####')
    A = pd.DataFrame([attributes(path_of[k]) for k in keys])
    grp_msld = np.array([ita_group(v) for v in A['ITA']])
    t = []
    for col in ATTRS:
        v = A[col].to_numpy(float)
        ok = sel & np.isfinite(v)
        pz, s = labels[ok] == b, v[ok]
        auc = safe_auc(pz, s)
        lo, hi = boot_ci(lambda i: safe_auc(pz[i], s[i]), len(s), rng)
        t.append({'attribute': col, **{c: f'{np.nanmedian(v[labels == i]):.3g}' for i, c in enumerate(CLASSES)},
                  'Mpox vs Chickenpox AUC [95% CI]': f'{auc:.3f} [{lo:.3f}, {hi:.3f}]', '_s': abs(auc - 0.5)})
    t = pd.DataFrame(t).sort_values('_s', ascending=False).drop(columns='_s')
    add_table('S5a_image_attributes', t, 'median image attribute per class (MSLD originals) and Mpox vs Chickenpox '
                                         'AUC (> 0.5 = higher in Mpox)')
    mask_items = {v: [(k, (lambda p=path_of[k], v=v: masked(p, v))) for k in keys] for v in MASKS}
    mfeats = {}
    for n in CONFIGS[MAIN]:
        got = features_for(n, mask_items, device)
        mfeats[n] = got
    X_attr = A[ATTRS].to_numpy(float)
    attr_sets = {'image attributes (all 14)': ATTRS, 'file attributes (size, format, JPEG)': FILE_ATTRS,
                 'pixel attributes (colour, skin tone, sharpness)': PIXEL_ATTRS}
    sp = {**{f'{MAIN}, {v}': [] for v in MASKS}, **{k: [] for k in attr_sets}}
    for f in fold_names:
        ks = fold_keys[f]
        yy = {s: labels[[row[k] for k in ks[s]]] for s in ks}
        for v in MASKS:
            src = {n: mfeats[n][v] for n in CONFIGS[MAIN]}
            sp[f'{MAIN}, {v}'].append(probe_clean(Xof(CONFIGS[MAIN], ks['train'], src), yy['train'],
                                                  Xof(CONFIGS[MAIN], ks['val'], src),
                                                  Xof(CONFIGS[MAIN], ks['test'], src))[2])
        itr, ite = [row[k] for k in ks['train']], [row[k] for k in ks['test']]
        for k, cols in attr_sets.items():
            j = [ATTRS.index(c) for c in cols]
            sp[k].append(attr_probs(X_attr[itr][:, j], yy['train'], X_attr[ite][:, j]))
    t = []
    for k, p in [(MAIN, pool(MAIN, 'test'))] + [(k, np.concatenate(v)) for k, v in sp.items()]:
        r, correct = evaluate(y, p, rng)
        r['McNemar vs main'] = '(main)' if k == MAIN else f'p={mcnemar(correct, main_correct):.3g}'
        t.append({'model': k, **r})
    maj = np.bincount(y, minlength=len(CLASSES)).max() / len(y)
    t.append({'model': 'chance (majority class / uniform)', 'n': len(y), 'acc [95% CI]': f'{maj * 100:.1f}',
              'bal_acc': f'{100 / len(CLASSES):.1f}', 'mpox AUC [95% CI]': '0.500'})
    add_table('S5b_masks_and_attribute_models', pd.DataFrame(t).fillna(''),
              'pooled test predictions: main model on masked images and models that see only image attributes')
    g_test = grp_msld[[row[k] for k in key_test]]
    t = []
    for _, lab in ITA_GROUPS:
        m = g_test == lab
        rec = lambda c: fmt_rate(int((pred[m & (y == c)] == c).sum()), int((m & (y == c)).sum()))
        t.append({'ITA group': lab, 'n': int(m.sum()), 'accuracy': f'{(pred[m] == y[m]).mean() * 100:.1f}',
                  'Mpox recall': rec(b), 'Chickenpox recall': rec(a),
                  **{c: int((y[m] == i).sum()) for i, c in enumerate(CLASSES)}})
    add_table('S5c_msld_by_skin_tone', pd.DataFrame(t), 'main model by skin-tone group (ITA), pooled test images; '
                                                        'rate [95% Wilson CI] (k/n)')
    A.assign(md5=keys, label=[CLASSES[i] for i in labels], tone_group=grp_msld).to_csv(
        os.path.join(OUT, 'attributes_msld.csv'), index=False)

    # =========================================================================
    # S6 external validation
    # =========================================================================
    print('\n##### S6 external validation #####')
    ext = external_datasets(folds_dir)
    msld_hash = np.stack([ahash(path_of[k]) for k in keys])
    rows = []
    for name, items in ext.items():
        seen = set()
        for p, yv, c in items:
            h = md5(p)
            if h not in seen:
                seen.add(h)
                rows.append({'dataset': name, 'path': p, 'md5': h, 'y': yv, 'class': c, 'exact': h in label_of})
        print(f'  {name}: {len(items)} images, {len(seen)} unique, classes {sorted({c for _, _, c in items})}')
    E = pd.DataFrame(rows, columns=['dataset', 'path', 'md5', 'y', 'class', 'exact'])
    if E.empty:
        raise SystemExit('no external dataset found: check EXTERNAL_DIRS and the inputs of the notebook')
    E['hamming to MSLD'] = min_hamming(np.stack([ahash(p) for p in E['path']]), msld_hash)
    E['near'] = ~E['exact'] & (E['hamming to MSLD'] <= NEAR_THRESHOLD)
    E['independent'] = ~E['exact'] & ~E['near']
    shipped, shipped_ita = tone_labels(list(E['path']))
    base_name = E['path'].map(lambda p: os.path.basename(p).lower())
    E['tone label'] = base_name.map(lambda v: shipped.get(v, ''))
    E['ITA authors'] = base_name.map(lambda v: shipped_ita.get(v, np.nan))
    AE = pd.DataFrame([attributes(p) for p in E['path']], columns=ATTRS)
    E['ITA'] = AE['ITA'].to_numpy()
    E['tone group'] = E['ITA'].map(ita_group)
    t = []
    for (name, c), g in E.groupby(['dataset', 'class'], sort=False):
        t.append({'dataset': name, 'class': c, 'in MSLD classes': 'yes' if g['y'].iloc[0] >= 0 else 'no',
                  'unique images': len(g), 'exact copies of MSLD': int(g['exact'].sum()),
                  'near copies of MSLD': int(g['near'].sum()), 'independent': int(g['independent'].sum()),
                  **{f'independent {short(lab)}': int((g['independent'] & (g['tone group'] == lab)).sum())
                     for _, lab in ITA_GROUPS}})
    add_table('S6a_external_data', pd.DataFrame(t), 'external images: copies of the MSLD originals (removed from the '
                                                    'test) and skin-tone groups of the independent images')
    ok = E['ITA authors'].notna() & E['ITA'].notna()
    if ok.sum() >= 10:
        o = E[ok]
        ours = np.where(o['ITA'] > 10, 'light', 'dark')
        has = o['tone label'].isin(['light', 'dark']).to_numpy()
        add_table('S6b_ita_check', pd.DataFrame([{
            'dataset': ', '.join(sorted(set(o['dataset']))), 'images': int(ok.sum()),
            'Spearman rho': f'{spearmanr(o["ITA"], o["ITA authors"])[0]:.3f}',
            'median abs difference (deg)': f'{np.median(np.abs(o["ITA"] - o["ITA authors"])):.1f}',
            'light/dark agreement': f'{(ours[has] == o["tone label"].to_numpy()[has]).mean() * 100:.1f}%',
            'Cohen kappa': f'{cohen_kappa_score(ours[has], o["tone label"].to_numpy()[has]):.3f}'}]),
            'our ITA (YCrCb skin pixels; light = ITA > 10) against the ITA and labels of the MCSI authors')

    # features of every image variant needed by S6 and S7
    ext_keys = list(dict.fromkeys(E['md5']))
    ext_path = dict(zip(E['md5'], E['path']))
    kinds_msld = sorted({k for v, au, _ in METHODS.values() for k in [v] + au} - {'original'})
    kinds_ext = sorted({v for v, _, _ in METHODS.values()} - {'original'})
    vf = {}
    for n in dict.fromkeys(bb for c in EXTERNAL_CONFIGS if c in configs for bb in configs[c]):
        sets = {'external original': [(h, (lambda p=ext_path[h]: variant(p, 'original'))) for h in ext_keys]}
        if n in CONFIGS[MAIN]:
            sets.update({f'msld {k}': [(h, (lambda p=path_of[h], k=k: variant(p, k))) for h in keys]
                         for k in kinds_msld})
            sets.update({f'external {k}': [(h, (lambda p=ext_path[h], k=k: variant(p, k))) for h in ext_keys]
                         for k in kinds_ext})
        for tag, fe in features_for(n, sets, device).items():
            src, kind = tag.split(' ', 1)
            vf[(n, src, kind)] = fe
        vf[(n, 'msld', 'original')] = feats[n]
    XV = lambda bs, src, kind, ks: np.concatenate([np.stack([vf[(n, src, kind)][k] for k in ks]) for n in bs], 1)

    Ei = E[E['independent']].reset_index(drop=True)
    Ec = E[~E['independent']].reset_index(drop=True)
    yi, yc = Ei['y'].to_numpy(int), Ec['y'].to_numpy(int)
    subsets = [(n, (Ei['dataset'] == n).to_numpy()) for n in ext] + \
        ([('all external', np.ones(len(Ei), bool))] if len(ext) > 1 else [])
    PE, thr_sens = {}, {}
    for c in [c for c in EXTERNAL_CONFIGS if c in configs]:
        (pi, pc), oof = probe7(XV(configs[c], 'msld', 'original', keys), labels,
                               [XV(configs[c], 'external', 'original', list(Ei['md5'])),
                                XV(configs[c], 'external', 'original', list(Ec['md5']))])
        PE[c] = (pi, pc)
        thr_sens[c] = threshold_for_sensitivity(labels, oof, SENS_TARGET)
    p_attr = attr_probs(X_attr, labels, AE.loc[E['independent'].to_numpy(), ATTRS].to_numpy(float))
    t, conf = [], []
    for name, s in subsets:
        ref = evaluate(yi[s], PE[MAIN][0][s], np.random.default_rng(0))[1]
        for c, (pi, _) in PE.items():
            r, corr = evaluate(yi[s], pi[s], rng, thr=thr_sens[c])
            known, other = s & (yi >= 0), s & (yi < 0)
            t.append({'subset': name, 'model': c, **r,
                      'acc 6-way': f'{(pi[known].argmax(1) == yi[known]).mean() * 100:.1f}',
                      'outside MSLD called mpox': fmt_rate(int((pi[other].argmax(1) == MPOX).sum()), int(other.sum()))
                      if other.any() else '',
                      'McNemar vs main': '(main)' if c == MAIN else f'p={mcnemar(corr, ref):.3g}'})
        r, _ = evaluate(yi[s], p_attr[s], rng)
        t.append({'subset': name, 'model': 'image attributes (all 14)', **r})
        if name == subsets[-1][0] and len(Ec) >= 10:
            r, _ = evaluate(yc, PE[MAIN][1], rng, thr=thr_sens[MAIN])
            t.append({'subset': 'copies of MSLD (not independent)', 'model': MAIN, **r})
        pm = PE[MAIN][0][s].argmax(1)
        for cls in dict.fromkeys(Ei.loc[s, 'class']):
            m = (Ei.loc[s, 'class'] == cls).to_numpy()
            conf.append({'subset': name, 'true class': cls, 'n': int(m.sum()),
                         **{c: int((pm[m] == i).sum()) for i, c in enumerate(CLASSES)}})
    add_table('S6c_external', pd.DataFrame(t).fillna(''),
              'independent external images, models trained on all MSLD images (no re-training). acc/bal_acc/f1: '
              'MSLD classes, prediction restricted to the classes of the dataset; acc 6-way: all six classes '
              'allowed; mpox metrics: all images (classes outside MSLD = non-mpox); threshold: 95% mpox '
              'sensitivity on MSLD out-of-fold predictions')
    add_table('S6d_external_confusion', pd.DataFrame(conf), 'main model on the independent external images')

    # =========================================================================
    # S7 skin tone
    # =========================================================================
    print('\n##### S7 skin tone #####')
    bs = CONFIGS[MAIN]
    cvp = {m: [] for m in METHODS}
    for f in fold_names:
        ks = fold_keys[f]
        itr = [row[k] for k in ks['train']]
        ytr = labels[itr]
        for m, (kind, aug, rw) in METHODS.items():
            w = tone_weights(ytr, grp_msld[itr]) if rw else None
            (p,), _ = probe7(XV(bs, 'msld', kind, ks['train']), ytr, [XV(bs, 'msld', kind, ks['test'])],
                             aug=[XV(bs, 'msld', au, ks['train']) for au in aug], w=w)
            cvp[m].append(p)
        print(f'  {f} done', flush=True)
    t = []
    for m, ps in cvp.items():
        p = np.concatenate(ps)
        pr_ = p.argmax(1)
        r = {'method': m, 'acc': f'{(pr_ == y).mean() * 100:.1f}', 'bal_acc': f'{balanced_accuracy_score(y, pr_) * 100:.1f}',
             'mpox AUC': f'{safe_auc(y == MPOX, p[:, MPOX]):.3f}'}
        for _, lab in ITA_GROUPS:
            s = (y == MPOX) & (g_test == lab)
            r[f'mpox sens, {short(lab)}'] = fmt_rate(int((pr_[s] == MPOX).sum()), int(s.sum()))
        t.append(r)
    add_table('S7a_tone_methods_msld_cv', pd.DataFrame(t), f'{MAIN}, MSLD pooled test predictions of the 5 folds, '
                                                           'by skin-tone method')
    w_all = tone_weights(labels, grp_msld)
    PT = {}
    for m, (kind, aug, rw) in METHODS.items():
        (p,), _ = probe7(XV(bs, 'msld', kind, keys), labels, [XV(bs, 'external', kind, list(Ei['md5']))],
                         aug=[XV(bs, 'msld', au, keys) for au in aug], w=w_all if rw else None)
        PT[m] = p
    t = []
    for name, s in subsets:
        for m, p in PT.items():
            r, _ = evaluate(yi[s], p[s], rng)
            known, other = s & (yi >= 0), s & (yi < 0)
            t.append({'subset': name, 'method': m, **r,
                      'acc 6-way': f'{(p[known].argmax(1) == yi[known]).mean() * 100:.1f}',
                      'outside MSLD called mpox': fmt_rate(int((p[other].argmax(1) == MPOX).sum()), int(other.sum()))
                      if other.any() else ''})
    add_table('S7b_tone_methods_external', pd.DataFrame(t).fillna(''), f'{MAIN}, independent external images, by '
                                                                       'skin-tone method')
    full = subsets[-1][1]
    pos_all = yi == MPOX
    thr = {m: np.quantile(p[full & ~pos_all, MPOX], SPEC_TARGET) for m, p in PT.items()}
    groupings = [('ITA (ours)', Ei['tone group'].to_numpy(), [l for _, l in ITA_GROUPS])]
    if (Ei['tone label'] != '').any():
        groupings.append(('label of the MCSI authors', Ei['tone label'].to_numpy(), ['light', 'dark']))
    t = []
    for src, grp, levels in groupings:
        for lab in levels:
            s = full & (grp == lab)
            pz = pos_all[s]
            if not pz.any() or pz.all():
                continue
            for m, p in PT.items():
                a0, b1, lo, hi, pv = paired_auc(pz, PT['baseline'][s, MPOX], p[s, MPOX], rng)
                fl, at = p[s].argmax(1) == MPOX, p[s, MPOX] > thr[m]
                t.append({'skin tone from': src, 'group': short(lab), 'method': m, 'mpox n': int(pz.sum()),
                          'non-mpox n': int((~pz).sum()), 'AUC in group': f'{b1:.3f}',
                          'AUC - baseline [95% CI]': '' if m == 'baseline' else f'{b1 - a0:+.3f} [{lo:+.3f}, {hi:+.3f}]',
                          'p': '' if m == 'baseline' else f'{pv:.3g}',
                          'sens (argmax)': fmt_rate(int(fl[pz].sum()), int(pz.sum())),
                          'FPR (argmax)': fmt_rate(int(fl[~pz].sum()), int((~pz).sum())),
                          f'sens at {SPEC_TARGET:.0%} spec': fmt_rate(int(at[pz].sum()), int(pz.sum())),
                          f'FPR at {SPEC_TARGET:.0%} spec': fmt_rate(int(at[~pz].sum()), int((~pz).sum()))})
    add_table('S7c_tone_within_group', pd.DataFrame(t),
              f'{subsets[-1][0]}, independent images: AUC within each skin-tone group (paired bootstrap difference '
              f'to the baseline), sensitivity and FPR at the argmax and at a threshold giving {SPEC_TARGET:.0%} '
              'specificity over all external non-mpox images')

    out = Ei.drop(columns='path').copy()
    for c, (pi, _) in PE.items():
        out[[f'p {cl} | {c}' for cl in CLASSES]] = pi
    for m, p in PT.items():
        out[[f'p {cl} | {MAIN} | {m}' for cl in CLASSES]] = p
    out.to_csv(os.path.join(OUT, 'predictions_external.csv'), index=False)
    E.drop(columns='path').assign(**{f'attr {c}': AE[c].to_numpy() for c in ATTRS}).to_csv(
        os.path.join(OUT, 'external_images.csv'), index=False)
    print(f'\nDone in {(time.time() - T0) / 60:.0f} min. Files in {OUT}: {len(TABLES)} tables (*.csv), '
          'predictions_msld_cv.csv, predictions_external.csv, concept_scores_msld.csv, attributes_msld.csv, '
          'external_images.csv')


main()
