"""
Pilot 7 on MSLD v2.0 (official patient-independent folds, duplicates screened as in the paper): skin-tone bias.
Pilot 6 showed that mpox images in MSLD (and in MSID) have darker skin than chickenpox images (median ITA 17.7 vs 38.2)
and that the main model, tested on MSID without re-training, detected only 41% of the mpox cases on light skin
(89% on dark skin). This pilot
  1  adds MCSI (labels in metadata.csv, skin-tone labels and ITA of the dataset authors in skin_tone_labels.csv) as a
     second external test set, and checks our ITA estimate against the authors' ITA
  2  tries to remove the skin-tone shortcut, for SigLIP2 and for the main model MONET+SigLIP2:
       baseline                  as in pilots 4-6
       reweighted                every (class x skin-tone group) cell of the training images weighs the same
       grey + equalised          features of grey-scale images with equalised histogram (no colour, no global tone)
       tone augmentation         every training image is added twice more, artificially lightened and darkened
       tone aug + reweighted     both
     and reports, in MSLD cross-validation and on the external images, the mpox sensitivity per skin-tone group.

Kaggle: the same notebook and inputs as pilot 6 (MSLD v2.0, pilot 3 output, MCSI, MSID), Accelerator = GPU (T4),
Internet = On. Paste the whole file into ONE cell (the notebook must contain only this cell), "Save Version -> Save
& Run All". Expected time: about 25-35 min.
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
EXTERNAL_DIRS = ['/kaggle/input/datasets/talhakabakus/mcsi-mpox-close-skin-images',      # MCSI (Zenodo v2)
                 '/kaggle/input/datasets/talhakabakus/msid-monkeypox-skin-images-dataset']  # MSID (Mendeley v6)
# None = every other dataset under /kaggle/input with class folders
OUT = '/kaggle/working/pilot7'
MAIN = 'LP concat monet+siglip2_so400m'
CONFIGS = {'LP siglip2_so400m': ['siglip2_so400m'], MAIN: ['monet', 'siglip2_so400m']}
METHODS = {  # name: (image variant of the features, augmented variants added to training, reweighting)
    'baseline': ('original', [], False),
    'reweighted': ('original', [], True),
    'grey + equalised': ('grey', [], False),
    'tone augmentation': ('original', ['lighter', 'darker'], False),
    'tone aug + reweighted': ('original', ['lighter', 'darker'], True)}
TONE_SHIFT = {'lighter': (+15.0, 0.85), 'darker': (-15.0, 1.20)}  # L* shift and b* factor (CIE Lab), whole image
REQUIRED = ['monet', 'siglip2_so400m']
MASKS = ['centre masked', 'skin masked', 'background masked']  # masked-image variants (main model only)
CENTRE_AREA = 0.5           # fraction of the image area grey-filled by 'centre masked'
GREY = (124, 116, 104)      # ImageNet mean colour
ITA_GROUPS = [(41, 'light (ITA > 41)'), (10, 'intermediate (10-41)'), (-999, 'dark (ITA <= 10)')]
NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5
SENS_TARGET = 0.95
BOOTSTRAP = 2000
BATCH = 32
DPI = 300
CLASS_COLOR = {'Chickenpox': '#E69F00', 'Cowpox': '#56B4E9', 'HFMD': '#009E73', 'Healthy': '#0072B2',
               'Measles': '#CC79A7', 'Mpox': '#D55E00'}

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
                    return [(names[b], v) for b, v in zip(base, vals) if b in names]
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
        cache = os.path.join(OUT, f'feat6_{name}_{re.sub(r"\W+", "_", tag)}.npz')
        cached = None
        if os.path.exists(cache):
            z = np.load(cache)
            cached = dict(zip(z['keys'], z['feats']))
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


def figure7(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                         'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    groups = [n for _, n in ITA_GROUPS if n in set(d['group'])]
    cfgs = list(dict.fromkeys(d['model']))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.6 * len(cfgs), 2.8), squeeze=False, sharey=True)
    shades = ['#999999', '#E69F00', '#56B4E9', '#009E73', '#D55E00']
    for ax, cfg in zip(axes[0], cfgs):
        width = 0.8 / len(METHODS)
        for k, m in enumerate(METHODS):
            for i, g in enumerate(groups):
                r = d[(d['model'] == cfg) & (d['method'] == m) & (d['group'] == g)]
                if r.empty or not r['n'].iloc[0]:
                    continue
                p, lo, hi = wilson(int(r['k'].iloc[0]), int(r['n'].iloc[0]))
                x = i + (k - (len(METHODS) - 1) / 2) * width
                ax.bar(x, p * 100, width * 0.9, color=shades[k % len(shades)], label=m if i == 0 else None)
                ax.errorbar(x, p * 100, yerr=[[max(0, p - lo) * 100], [max(0, hi - p) * 100]], fmt='none', ecolor='#333333',
                            elinewidth=0.6, capsize=1.5)
        ax.set_xticks(range(len(groups)), [g.split(' (')[0] + f'\n({g.split(" (")[1]}' for g in groups])
        ax.set(title=cfg, ylim=(0, 100))
    axes[0][0].set_ylabel('mpox sensitivity, external images (%)')
    axes[0][-1].legend(frameon=False, fontsize=6.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_mpox_sensitivity_by_skin_tone.png'), dpi=DPI)
    plt.close(fig)


# =============================================================================
# Pilot 7: main
# =============================================================================
def main():
    ensure_packages()
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    folds_dir = FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True)
    folds = [screen(f) for f in discover_folds(folds_dir)]
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})
    label_of = {md5(p): c for f in folds for s in ('train', 'val', 'test') for p, c in f[s]}
    path_of = {md5(p): p for p in originals}
    keys = sorted(label_of)
    labels = np.array([label_of[k] for k in keys])
    row = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(0)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 300)
    tables, titles = {}, {}

    # ---- external datasets, overlap with MSLD, skin tone
    print('\n--- external datasets ---')
    ext = external_datasets(folds_dir)
    msld_hash = np.stack([ahash(path_of[k]) for k in keys])
    rows = []
    for name, items in ext.items():
        seen = set()
        for p, y, c in items:
            h = md5(p)
            if h not in seen:
                seen.add(h)
                rows.append({'dataset': name, 'path': p, 'md5': h, 'y': y, 'class': c, 'exact': h in label_of})
        print(f'  {name}: {len(items)} images, {len(seen)} unique, classes {sorted({c for _, _, c in items})}')
    E = pd.DataFrame(rows, columns=['dataset', 'path', 'md5', 'y', 'class', 'exact'])
    if E.empty:
        raise SystemExit('no external dataset found: check EXTERNAL_DIRS and the inputs of the notebook')
    E['near'] = ~E['exact'] & (min_hamming(np.stack([ahash(p) for p in E['path']]), msld_hash) <= NEAR_THRESHOLD)
    E['independent'] = ~E['exact'] & ~E['near']
    shipped, shipped_ita = tone_labels(list(E['path']))
    base = E['path'].map(lambda p: os.path.basename(p).lower())
    E['tone label'] = base.map(lambda b: shipped.get(b, ''))
    E['ITA authors'] = base.map(lambda b: shipped_ita.get(b, np.nan))
    print('\n--- skin tone (ITA) ---')
    t0 = time.time()
    ita_msld = np.array([attributes(path_of[k])['ITA'] for k in keys])
    E['ITA'] = [attributes(p)['ITA'] for p in E['path']]
    E['tone group'] = E['ITA'].map(ita_group)
    grp_msld = np.array([ita_group(v) for v in ita_msld])
    print(f'  {len(keys) + len(E)} images in {time.time() - t0:.0f}s')

    t1 = []
    for (name, c), g in E.groupby(['dataset', 'class'], sort=False):
        t1.append({'dataset': name, 'class': c, 'in MSLD classes': 'yes' if g['y'].iloc[0] >= 0 else 'no',
                   'unique images': len(g), 'exact copies of MSLD': int(g['exact'].sum()),
                   'near copies of MSLD': int(g['near'].sum()), 'independent': int(g['independent'].sum()),
                   **{f'independent {lab}': int((g['independent'] & (g['tone group'] == lab)).sum())
                      for _, lab in ITA_GROUPS}})
    tables['table1_external_data'] = pd.DataFrame(t1)
    titles['table1_external_data'] = ('external images: copies of the MSLD originals (MD5 / aHash Hamming <= '
                                      f'{NEAR_THRESHOLD}) and skin-tone groups (ITA) of the independent images')

    ok = E['ITA authors'].notna() & E['ITA'].notna()
    if ok.sum() >= 10:
        o = E[ok]
        ours = np.where(o['ITA'] > 10, 'light', 'dark')
        has = o['tone label'].isin(['light', 'dark'])
        tables['table2_ita_check'] = pd.DataFrame([{
            'dataset': ', '.join(sorted(set(o['dataset']))), 'images': int(ok.sum()),
            'Spearman rho (our ITA ~ authors ITA)': f'{spearmanr(o["ITA"], o["ITA authors"])[0]:.3f}',
            'median abs difference (degrees)': f'{np.median(np.abs(o["ITA"] - o["ITA authors"])):.1f}',
            'agreement light/dark (our ITA > 10 vs authors label)':
                f'{(ours[has] == o["tone label"][has]).mean() * 100:.1f}%' if has.any() else '',
            'Cohen kappa': f'{cohen_kappa_score(ours[has], o["tone label"][has]):.3f}' if has.any() else ''}])
        titles['table2_ita_check'] = 'our ITA estimate (YCrCb skin pixels) against the ITA of the dataset authors'

    # ---- features of every image variant
    print('\n--- features ---')
    feats = {}
    kinds_msld = sorted({k for v, a, _ in METHODS.values() for k in [v] + a} - {'original'})
    kinds_ext = sorted({v for v, _, _ in METHODS.values()} - {'original'})
    ext_keys = list(dict.fromkeys(E['md5']))
    ext_path = dict(zip(E['md5'], E['path']))
    for n in sorted({b for v in CONFIGS.values() for b in v}):
        feats[(n, 'msld', 'original')] = get_features(n, originals, device)
        sets = {'external original': [(h, (lambda p=ext_path[h]: variant(p, 'original'))) for h in ext_keys]}
        sets.update({f'msld {k}': [(h, (lambda p=path_of[h], k=k: variant(p, k))) for h in keys] for k in kinds_msld})
        sets.update({f'external {k}': [(h, (lambda p=ext_path[h], k=k: variant(p, k))) for h in ext_keys]
                     for k in kinds_ext})
        for tag, f in features_for(n, sets, device).items():
            src, kind = tag.split(' ', 1)
            feats[(n, src, kind)] = f
    X = lambda src, kind, bs, ks: np.concatenate([np.stack([feats[(n, src, kind)][k] for k in ks]) for n in bs], 1)

    # ---- MSLD cross-validation, official folds
    print('\n--- folds ---')
    combos = [(c, m) for c in CONFIGS for m in METHODS]
    P = {cm: [] for cm in combos}
    y_all, g_all, t0 = [], [], time.time()
    for f in folds:
        idx = {s: np.array([row[md5(p)] for p, _ in f[s]]) for s in ('train', 'test')}
        ktr, kte = [keys[i] for i in idx['train']], [keys[i] for i in idx['test']]
        ytr = labels[idx['train']]
        for c, m in combos:
            kind, aug, rw = METHODS[m]
            bs = CONFIGS[c]
            w = tone_weights(ytr, grp_msld[idx['train']]) if rw else None
            (p,), _ = probe7(X('msld', kind, bs, ktr), ytr, [X('msld', kind, bs, kte)],
                             aug=[X('msld', a, bs, ktr) for a in aug], w=w)
            P[(c, m)].append(p)
        y_all.append(labels[idx['test']])
        g_all.append(grp_msld[idx['test']])
        print(f'  {f["name"]} done ({time.time() - t0:.0f}s)', flush=True)
    y, g = np.concatenate(y_all), np.concatenate(g_all)
    t3 = []
    for (c, m), ps in P.items():
        p = np.concatenate(ps)
        pred = p.argmax(1)
        r = {'model': c, 'method': m, 'acc': f'{(pred == y).mean() * 100:.1f}',
             'bal_acc': f'{balanced_accuracy_score(y, pred) * 100:.1f}',
             'mpox AUC': f'{safe_auc(y == MPOX, p[:, MPOX]):.3f}'}
        for _, lab in ITA_GROUPS:
            s = (y == MPOX) & (g == lab)
            r[f'mpox sens, {lab.split(" (")[0]}'] = fmt_rate(int((pred[s] == MPOX).sum()), int(s.sum()))
        for _, lab in ITA_GROUPS:
            s = (y == CHICKENPOX) & (g == lab)
            r[f'chickenpox recall, {lab.split(" (")[0]}'] = fmt_rate(int((pred[s] == CHICKENPOX).sum()), int(s.sum()))
        t3.append(r)
    tables['table3_msld_cv'] = pd.DataFrame(t3)
    titles['table3_msld_cv'] = ('MSLD, pooled test predictions of the 5 official folds: accuracy and recall per '
                                'skin-tone group (ITA), rate [95% Wilson CI] (correct/n)')

    # ---- external test: models trained on all MSLD images
    print('\n--- external validation (models trained on all MSLD images) ---')
    w_all = tone_weights(labels, grp_msld)
    Ei = E[E['independent']].reset_index(drop=True)
    subsets = [(n, Ei['dataset'] == n) for n in ext] + ([('all external', np.ones(len(Ei), bool))]
                                                          if len(ext) > 1 else [])
    t4, t5, fig_rows, base_correct = [], [], [], {}
    for c, m in combos:
        kind, aug, rw = METHODS[m]
        bs = CONFIGS[c]
        (p,), oof = probe7(X('msld', kind, bs, keys), labels, [X('external', kind, bs, list(Ei['md5']))],
                           aug=[X('msld', a, bs, keys) for a in aug], w=w_all if rw else None)
        thr = threshold_for_sensitivity(labels, oof, SENS_TARGET)
        yy = Ei['y'].to_numpy(int)
        flag = p.argmax(1) == MPOX
        for name, s in subsets:
            s = np.asarray(s)
            r, _ = evaluate(yy[s], p[s], rng, thr=thr)
            known = s & (yy >= 0)
            r = {'subset': name, 'model': c, 'method': m, **r,
                 'acc 6-way': f'{(p[known].argmax(1) == yy[known]).mean() * 100:.1f}'}
            t4.append(r)
            sources = [('ITA (ours)', Ei['tone group'].to_numpy())]
            if (Ei.loc[s, 'tone label'] != '').any():
                sources.append(('label of the dataset authors', Ei['tone label'].to_numpy()))
            for src, grp in sources:
                for lab in [l for _, l in ITA_GROUPS] if src == 'ITA (ours)' else ['light', 'dark']:
                    pos = s & (yy == MPOX) & (grp == lab)
                    neg = s & (yy != MPOX) & (grp == lab)
                    k = int(flag[pos].sum())
                    key = (name, src, lab, c)
                    if m == 'baseline':
                        base_correct[key] = flag[pos]
                    mc = ''
                    if m != 'baseline' and pos.any():
                        mc = f'p={mcnemar(flag[pos], base_correct[key]):.3g}'
                    t5.append({'subset': name, 'skin tone from': src, 'group': lab, 'model': c, 'method': m,
                               'mpox sensitivity': fmt_rate(k, int(pos.sum())),
                               'McNemar vs baseline': mc,
                               'non-mpox called mpox': fmt_rate(int(flag[neg].sum()), int(neg.sum()))})
                    if name == subsets[-1][0] and src == 'ITA (ours)':
                        fig_rows.append({'model': c, 'method': m, 'group': lab, 'k': k, 'n': int(pos.sum())})
        print(f'  {c} / {m} done', flush=True)
    tables['table4_external'] = pd.DataFrame(t4).fillna('')
    titles['table4_external'] = ('independent external images, models trained on all MSLD images (no re-training); '
                                 'acc: MSLD classes, prediction restricted to the classes of the dataset; acc 6-way: '
                                 'all six MSLD classes allowed; mpox metrics: all images (other classes = non-mpox)')
    tables['table5_external_by_skin_tone'] = pd.DataFrame(t5)
    titles['table5_external_by_skin_tone'] = ('independent external images: mpox sensitivity and false-positive rate '
                                              'per skin-tone group, rate [95% Wilson CI] (k/n); McNemar on the mpox '
                                              'images of the group against the baseline of the same model')

    for name, t in tables.items():
        print(f'\n=== Pilot 7, {name.split("_")[0]}: {titles[name]} ===')
        print(t.to_string(index=False))
        t.to_csv(os.path.join(OUT, f'pilot7_{name}.csv'), index=False)
    E.drop(columns='path').to_csv(os.path.join(OUT, 'pilot7_external_images.csv'), index=False)
    figure7(fig_rows)
    print(f'\nFiles in {OUT}: pilot7_table*.csv, pilot7_external_images.csv, fig_mpox_sensitivity_by_skin_tone.png')


main()
