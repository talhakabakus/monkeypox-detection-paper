"""
Pilot 5 on MSLD v2.0 (official patient-independent folds, duplicates screened as in the paper): clinical concepts.
Pilot 4 showed that 75% of the errors of the main model ('LP concat monet+siglip2_so400m') are chickenpox <-> mpox
confusions. MONET and SigLIP2 embed images and text in one space, so the saved image features can be scored against
written clinical concepts (e.g. "umbilicated pustules", "lesions at different stages") without any training:

  score_c(image) = cos(image, concept prompts) - cos(image, neutral skin prompts), minus the image's mean over all
                   concepts (so a score says how well a concept fits relative to the others), z-scored per concept,
                   averaged over both models

  1  concept validity: class profile of every concept, and agreement between MONET and SigLIP2 (Spearman rho)
  2  mpox vs chickenpox: which concepts separate the two (AUC with bootstrap 95% CI)
  3  error analysis: concept profile of the misclassified chickenpox/mpox test images of the main model
  4  models (clean protocol of pilot 4, pooled test predictions of the 5 folds):
       CBM             logistic regression on the concept scores only (interpretable)
       main + concepts main features plus concept scores
       two-stage       main model; images it calls chickenpox or mpox are re-decided by a chickenpox-vs-mpox specialist
  5  shortcut check: image/subject attributes (child, skin tone, watermark, blur) per class

Concept scores are zero-shot and have not been validated by dermatologists; they are hypotheses about what the model
uses, not ground truth.

Kaggle: inputs = the MSLD v2.0 dataset and the pilot 3 output (as for pilot 4), Internet = On (the two text encoders
are downloaded), Accelerator = None or GPU. Paste the whole file into ONE cell (the notebook must contain only this
cell), "Save Version -> Save & Run All". Expected time: about 10-15 min, mostly downloading MONET and SigLIP2.
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
FOLDS_DIR = None
OUT = '/kaggle/working/pilot5'
MAIN_BACKBONES = ['monet', 'siglip2_so400m']
CONCEPT_MODELS = {'monet': 'clip', 'siglip2_so400m': 'siglip'}   # text encoder family of each backbone
REQUIRED = MAIN_BACKBONES
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
NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5
REFER = [0.0, 0.1, 0.2, 0.3]
BOOTSTRAP = 2000
BATCH = 32
DPI = 300
CLASS_COLOR = {'Chickenpox': '#E69F00', 'Cowpox': '#56B4E9', 'HFMD': '#009E73', 'Healthy': '#0072B2',
               'Measles': '#CC79A7', 'Mpox': '#D55E00'}

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
MPOX = CLASSES.index('Mpox')
ALIASES = {'monkeypox': 'Mpox', 'normal': 'Healthy'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
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
# Pilot 5: concepts
# =============================================================================
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


def figures(Z, labels, pair_tab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    names = list(CONCEPTS)
    M = np.stack([Z[labels == k].mean(0) for k in range(len(CLASSES))], 1)
    fig, ax = plt.subplots(figsize=(4.2, 7.0))
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label='mean concept score (z)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_concept_profiles.png'), dpi=DPI)
    plt.close(fig)

    t = pair_tab.sort_values('AUC')
    fig, ax = plt.subplots(figsize=(4.2, 6.0))
    colors = [CLASS_COLOR['Mpox'] if a > 0.5 else CLASS_COLOR['Chickenpox'] for a in t['AUC']]
    ax.barh(t['concept'], t['AUC'] - 0.5, left=0.5, color=colors, height=0.7)
    ax.errorbar(t['AUC'], range(len(t)), xerr=[t['AUC'] - t['lo'], t['hi'] - t['AUC']], fmt='none',
                ecolor='#333333', elinewidth=0.6, capsize=1.5)
    ax.axvline(0.5, color='#555555', lw=0.8)
    ax.set(xlabel=f'AUC, {PAIR[1]} vs {PAIR[0]} (> 0.5: higher in {PAIR[1]})', xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_concepts_mpox_vs_chickenpox.png'), dpi=DPI)
    plt.close(fig)


def main():
    global FOLD_NAMES
    ensure_packages()
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    folds = [screen(f) for f in discover_folds(FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True))]
    FOLD_NAMES = [f['name'] for f in folds]
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})
    label_of = {md5(p): c for f in folds for s in ('train', 'val', 'test') for p, c in f[s]}
    keys = sorted(label_of)
    labels = np.array([label_of[k] for k in keys])
    row = {k: i for i, k in enumerate(keys)}

    print('\n--- features ---')
    feats = {n: get_features(n, originals, device) for n in MAIN_BACKBONES}

    print('\n--- concept scores ---')
    raw, Zs = {}, {}
    for n in CONCEPT_MODELS:
        t0 = time.time()
        raw[n] = concept_scores(n, feats[n], keys, device)
        Zs[n] = (raw[n] - raw[n].mean(0)) / raw[n].std(0)
        print(f'  {n}: {raw[n].shape[1]} concepts for {raw[n].shape[0]} images ({time.time() - t0:.0f}s)')
        if device == 'cuda':
            torch.cuda.empty_cache()
    Z = np.mean([Zs[n] for n in CONCEPT_MODELS], 0)  # consensus of the two models
    rng = np.random.default_rng(0)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 200)
    tables = {}
    names = list(CONCEPTS)
    models = list(CONCEPT_MODELS)

    # ---- table 1: class profiles and model agreement
    from sklearn.metrics import roc_auc_score
    t1 = []
    for j, c in enumerate(names):
        r = {'concept': c, 'group': CONCEPTS[c][0],
             'rho monet~siglip2': f'{spearmanr(Zs[models[0]][:, j], Zs[models[1]][:, j])[0]:.2f}'}
        r.update({k: f'{Z[labels == i, j].mean():+.2f}' for i, k in enumerate(CLASSES)})
        aucs = [roc_auc_score(labels == i, Z[:, j]) for i in range(len(CLASSES))]
        r['highest in (one-vs-rest AUC)'] = f'{CLASSES[int(np.argmax(aucs))]} ({max(aucs):.2f})'
        t1.append(r)
    tables['table1_concept_profiles'] = pd.DataFrame(t1)

    # ---- table 2: mpox vs chickenpox
    a, b = CLASSES.index(PAIR[0]), CLASSES.index(PAIR[1])
    sel = np.isin(labels, [a, b])
    pos = labels[sel] == b
    t2 = []
    for j, c in enumerate(names):
        auc, lo, hi = auc_ci(Z[sel, j], pos, rng)
        sd = np.sqrt((Z[sel][pos, j].var(ddof=1) + Z[sel][~pos, j].var(ddof=1)) / 2)
        t2.append({'concept': c, 'group': CONCEPTS[c][0], 'AUC': auc, 'lo': lo, 'hi': hi,
                   f'z {PAIR[0]}': Z[labels == a, j].mean(), f'z {PAIR[1]}': Z[labels == b, j].mean(),
                   "Cohen's d": (Z[sel][pos, j].mean() - Z[sel][~pos, j].mean()) / sd,
                   **{f'AUC {n}': roc_auc_score(pos, Zs[n][sel, j]) for n in models}})
    pair_tab = pd.DataFrame(t2)
    pair_tab['_sep'] = (pair_tab['AUC'] - 0.5).abs()
    pair_tab = pair_tab.sort_values('_sep', ascending=False).drop(columns='_sep')
    show = pair_tab.copy()
    show['AUC [95% CI]'] = [f'{x:.3f} [{l:.3f}, {h:.3f}]' for x, l, h in zip(show['AUC'], show['lo'], show['hi'])]
    for col in [f'z {PAIR[0]}', f'z {PAIR[1]}', "Cohen's d"] + [f'AUC {n}' for n in models]:
        show[col] = show[col].map(lambda v: f'{v:+.2f}' if 'z ' in col or 'd' == col[-1] else f'{v:.3f}')
    tables['table2_mpox_vs_chickenpox'] = show[['concept', 'group', 'AUC [95% CI]', f'z {PAIR[0]}', f'z {PAIR[1]}',
                                                "Cohen's d"] + [f'AUC {n}' for n in models]]

    # ---- models on the clean protocol (as in pilot 4)
    print('\n--- probes (clean protocol) ---')
    C_all = np.concatenate([Zs[n] for n in models], 1)   # concept features of both models
    preds = {k: [] for k in ('main', 'CBM (concepts only)', 'main + concepts', 'two-stage')}
    y_all, key_all, t0 = [], [], time.time()
    for f in folds:
        idx = {s: np.array([row[md5(p)] for p, _ in f[s]]) for s in ('train', 'val', 'test')}
        y = {s: labels[idx[s]] for s in idx}
        Xmain = {s: np.concatenate([np.stack([feats[n][keys[i]] for i in idx[s]]) for n in MAIN_BACKBONES], 1)
                 for s in idx}
        Xc = {s: C_all[idx[s]] for s in idx}
        main_te = probe_clean(Xmain['train'], y['train'], Xmain['val'], Xmain['test'])[2]
        preds['main'].append(main_te)
        preds['CBM (concepts only)'].append(probe_clean(Xc['train'], y['train'], Xc['val'], Xc['test'])[2])
        Xmc = {s: np.concatenate([Xmain[s], Xc[s]], 1) for s in idx}
        preds['main + concepts'].append(probe_clean(Xmc['train'], y['train'], Xmc['val'], Xmc['test'])[2])
        tr = np.isin(y['train'], [a, b])
        pair_te = probe_clean(Xmain['train'][tr], y['train'][tr], Xmain['val'], Xmain['test'])[2]
        two = main_te.copy()
        route = np.isin(main_te.argmax(1), [a, b])
        mass = main_te[route][:, [a, b]].sum(1)
        share = pair_te[route][:, [a, b]] / pair_te[route][:, [a, b]].sum(1, keepdims=True)
        two[np.ix_(route, [a, b])] = share * mass[:, None]
        preds['two-stage'].append(two)
        y_all.append(y['test'])
        key_all += [keys[i] for i in idx['test']]
        print(f'  {f["name"]} done ({time.time() - t0:.0f}s)', flush=True)
    y = np.concatenate(y_all)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    main_correct = P['main'].argmax(1) == y
    t3 = []
    for k, p in P.items():
        correct = p.argmax(1) == y
        boot = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(BOOTSTRAP)]
        m, bn = metrics(y, p), binary(y, p)
        rec = lambda c: (p.argmax(1)[y == c] == c).mean() * 100
        t3.append({'model': k, 'acc [95% CI]': f'{correct.mean() * 100:.2f} [{np.percentile(boot, 2.5) * 100:.1f}, '
                                                f'{np.percentile(boot, 97.5) * 100:.1f}]',
                   'bal_acc': f'{m["balanced_accuracy"] * 100:.2f}', 'f1': f'{m["f1_macro"] * 100:.2f}',
                   f'{PAIR[0]} recall': f'{rec(a):.1f}', f'{PAIR[1]} recall': f'{rec(b):.1f}',
                   f'{PAIR[0]}<->{PAIR[1]} errors': int(((p.argmax(1) == a) & (y == b)).sum() +
                                                     ((p.argmax(1) == b) & (y == a)).sum()),
                   'mpox sens/spec': f'{bn["sensitivity"] * 100:.1f}/{bn["specificity"] * 100:.1f}',
                   'ece': f'{m["ece"] * 100:.1f}',
                   'McNemar vs main': '(main)' if k == 'main' else f'p={mcnemar(correct, main_correct):.3f}'})
    tables['table3_models'] = pd.DataFrame(t3)

    # ---- table 4: concept profile of the main model's chickenpox/mpox errors
    pred = P['main'].argmax(1)
    zt = Z[[row[k] for k in key_all]]
    groups = {f'{PAIR[0]} correct': (y == a) & (pred == a), f'{PAIR[0]} -> {PAIR[1]}': (y == a) & (pred == b),
              f'{PAIR[1]} -> {PAIR[0]}': (y == b) & (pred == a), f'{PAIR[1]} correct': (y == b) & (pred == b)}
    top = pair_tab['concept'].head(10).tolist()
    t4 = pd.DataFrame({g: [f'{zt[m, names.index(c)].mean():+.2f}' if m.any() else '' for c in top]
                       for g, m in groups.items()}, index=top)
    t4.loc['n images'] = [str(int(m.sum())) for m in groups.values()]
    tables['table4_error_profiles'] = t4.reset_index().rename(columns={'index': 'concept (top 10 of table 2)'})

    # ---- table 5: interpretable mpox-vs-chickenpox weights (descriptive: fitted on all original images)
    sc = StandardScaler().fit(Z[sel])
    clf = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(sc.transform(Z[sel]), pos)
    w = pd.DataFrame({'concept': names, 'group': [CONCEPTS[c][0] for c in names], 'weight': clf.coef_[0]})
    w = w.reindex(w['weight'].abs().sort_values(ascending=False).index)
    w['direction'] = np.where(w['weight'] > 0, f'-> {PAIR[1]}', f'-> {PAIR[0]}')
    w['weight'] = w['weight'].map(lambda v: f'{v:+.3f}')
    tables['table5_pair_weights'] = w

    titles = {'table1_concept_profiles': 'mean concept score (z, consensus of MONET and SigLIP2) per class, all '
                                         f'{len(keys)} original images',
              'table2_mpox_vs_chickenpox': f'{PAIR[1]} vs {PAIR[0]}: concept AUC (> 0.5 = higher in {PAIR[1]}), '
                                           'sorted by separation',
              'table3_models': 'pooled test predictions of the 5 folds (clean protocol)',
              'table4_error_profiles': f'mean concept z of the main model\'s {PAIR[0]}/{PAIR[1]} test images by outcome',
              'table5_pair_weights': f'logistic regression {PAIR[1]} vs {PAIR[0]} on the consensus concepts '
                                     '(descriptive, all images)'}
    for name, t in tables.items():
        print(f'\n=== Pilot 5, {name.split("_")[0]}: {titles[name]} ===')
        print(t.to_string(index=False))
        t.to_csv(os.path.join(OUT, f'pilot5_{name}.csv'), index=False)
    pd.DataFrame(Z, columns=names).assign(md5=keys, label=[CLASSES[i] for i in labels]).to_csv(
        os.path.join(OUT, 'pilot5_concept_scores.csv'), index=False)
    figures(Z, labels, pair_tab)
    print(f'\nFiles in {OUT}: pilot5_table*.csv, pilot5_concept_scores.csv, fig_concept_profiles.png, '
          'fig_concepts_mpox_vs_chickenpox.png')


main()
