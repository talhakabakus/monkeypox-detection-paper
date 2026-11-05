"""
Pilot: do foundation-model image features beat the ImageNet DenseNet201 on MSLD v2.0?

Every model is used as a FROZEN feature extractor; a logistic-regression head is trained on the training split of each
official, patient-independent fold, its regularisation is chosen on the validation split, and the test split is used
once. Exact and near duplicates of training images are removed from validation/test exactly as in run_experiments.py.
The point is a cheap ranking of representations (about 1 GPU hour), not final results.

Models (all downloaded from Hugging Face; the notebook needs Internet = On):
  densenet201_imagenet  timm densenet201.tv_in1k               reference: the backbone of the paper
  dinov2_vitl14         timm vit_large_patch14_dinov2.lvd142m  self-supervised, general images
  monet                 suinleelab/monet                       CLIP ViT-L/14 trained on dermatology images and text
  siglip_so400m         google/siglip-so400m-patch14-384       vision-language, general images

Kaggle: Accelerator = GPU T4, Internet = On, input = the MSLD v2.0 dataset. Paste the whole file into ONE cell and
start it with "Save Version -> Save & Run All" so that /kaggle/working/pilot is kept with the notebook version.
"""
import glob
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# Settings
# =============================================================================
MODELS = ['densenet201_imagenet', 'dinov2_vitl14', 'monet', 'siglip_so400m']
REFERENCE = 'densenet201_imagenet'
USE_AUG = False            # True: also extract features of the fold-matched augmented training images (~37k, slower)
FOLDS_DIR = None           # None = search /kaggle/input
AUG_DIR = None
OUT = '/kaggle/working/pilot'
NEAR_THRESHOLD = 10        # average-hash Hamming distance (of 256 bits), as in the paper
C_GRID = [0.001, 0.01, 0.1, 1.0]
BATCH = 32
SEED = 42
# Fine-tuned results already reported in the paper (accuracy, mean ± SD over folds), for context only
PAPER = {'DenseNet201 fine-tuned, original images': (79.01, 4.66),
         'DenseNet201 fine-tuned, + augmented images': (81.55, 5.10),
         'Hybrid (proposed), + augmented images': (79.66, 5.16)}

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
ALIASES = {'monkeypox': 'Mpox', 'normal': 'Healthy'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)
AUG_FOLD_RE = re.compile(r'^fold[\s_-]*(\d+)[\s_-]*aug$', re.IGNORECASE)


# =============================================================================
# Data
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


def discover_aug(aug_dir):
    out = {}
    for d in glob.glob(os.path.join(aug_dir, '*')):
        m = AUG_FOLD_RE.match(os.path.basename(d))
        if m:
            train = [s for s in glob.glob(os.path.join(d, '*')) if SPLITS.get(os.path.basename(s).lower()) == 'train']
            out[int(m.group(1))] = list_images(train[0])
    return out


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
    """drop exact and near duplicates of training images from val/test (as in the paper)"""
    train_h = {md5(p) for p, _ in fold['train']}
    report = {'fold': fold['name']}
    seen = set(train_h)
    for split in ('val', 'test'):
        exact = [md5(p) in seen for p, _ in fold[split]]
        kept = [x for x, e in zip(fold[split], exact) if not e]
        near = min_hamming(np.stack([ahash(p) for p, _ in kept]), np.stack([ahash(p) for p, _ in fold['train']]))
        fold[split] = [x for x, n in zip(kept, near) if n > NEAR_THRESHOLD]
        report[f'{split}_exact_removed'], report[f'{split}_near_removed'] = int(sum(exact)), int((near <= NEAR_THRESHOLD).sum())
        seen |= {md5(p) for p, _ in fold[split]}
    report.update({f'n_{s}': len(fold[s]) for s in ('train', 'val', 'test')})
    print(json.dumps(report))
    return fold


# =============================================================================
# Feature extractors
# =============================================================================
def as_tensor(out):
    """get_image_features returns a tensor in most transformers versions, a model output in some newer ones"""
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    return out if hasattr(out, 'float') else out[0]


def load_extractor(name, device):
    import torch
    if name in ('densenet201_imagenet', 'dinov2_vitl14'):
        import timm
        arch = {'densenet201_imagenet': 'densenet201.tv_in1k', 'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m'}[name]
        kw = {'img_size': 224} if 'dinov2' in arch else {}
        model = timm.create_model(arch, pretrained=True, num_classes=0, **kw).eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        cfg.update(input_size=(3, 224, 224), crop_pct=1.0)  # whole image, no centre crop (lesions reach the borders)
        tf = timm.data.create_transform(**cfg, is_training=False)
        return lambda imgs: model(torch.stack([tf(i) for i in imgs]).to(device))
    if name == 'monet':
        from transformers import AutoProcessor, CLIPModel
        model = CLIPModel.from_pretrained('suinleelab/monet').eval().to(device)
        proc = AutoProcessor.from_pretrained('suinleelab/monet')
        return lambda imgs: as_tensor(model.get_image_features(**proc(images=imgs, return_tensors='pt').to(device)))
    if name == 'siglip_so400m':
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained('google/siglip-so400m-patch14-384').eval().to(device)
        proc = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')
        return lambda imgs: as_tensor(model.get_image_features(**proc(images=imgs, return_tensors='pt').to(device)))
    raise ValueError(name)


def extract(name, paths, device):
    """features for every unique image (keyed by MD5), cached on disk"""
    import torch
    cache_file = os.path.join(OUT, f'features_{name}.npz')
    feats = {}
    if os.path.exists(cache_file):
        z = np.load(cache_file)
        feats = dict(zip(z['keys'], z['feats']))
    todo = {}
    for p in paths:
        if md5(p) not in feats:
            todo.setdefault(md5(p), p)
    if todo:
        fn = load_extractor(name, device)
        keys, t0 = list(todo), time.time()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            for i in range(0, len(keys), BATCH):
                batch = keys[i:i + BATCH]
                out = fn([Image.open(todo[k]).convert('RGB') for k in batch]).float().cpu().numpy()
                feats.update(zip(batch, out))
        print(f'  {name}: {len(keys)} images in {time.time() - t0:.0f}s, dim {out.shape[1]}')
        np.savez(cache_file, keys=np.array(list(feats)), feats=np.stack(list(feats.values())))
        del fn
        torch.cuda.empty_cache()
    return feats


# =============================================================================
# Linear probe and metrics
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


def probe(feats, fold, aug_rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    def xy(rows):
        return np.stack([feats[md5(p)] for p, _ in rows]), np.array([c for _, c in rows])
    Xtr, ytr = xy(fold['train'] + aug_rows)
    Xva, yva = xy(fold['val'])
    Xte, yte = xy(fold['test'])
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced', random_state=SEED).fit(Xtr, ytr)
        score = balanced_accuracy_score(yva, clf.predict(Xva))
        if best is None or score > best[0]:
            best = (score, C, clf)
    probs = np.zeros((len(yte), len(CLASSES)))
    probs[:, best[2].classes_] = best[2].predict_proba(Xte)
    return metrics(yte, probs), best[1], probs, yte


def corrected_ttest(a, b, ratio):
    from scipy import stats
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * var)
    return d.mean(), 2 * stats.t.sf(abs(t), len(d) - 1)


# =============================================================================
def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}; torch {torch.__version__}')
    folds = [screen(f) for f in discover_folds(FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True))]
    aug = discover_aug(AUG_DIR or find_dir(AUG_FOLD_RE, exclude_aug=False)) if USE_AUG else {}
    if USE_AUG:
        for i, f in enumerate(folds, 1):
            held = {md5(p) for s in ('val', 'test') for p, _ in f[s]}
            assert not any(md5(p) in held for p, _ in aug[i]), f'{f["name"]}: augmented image in val/test'
    paths = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]}
                   | {p for rows in aug.values() for p, _ in rows})
    ratio = np.mean([len(f['test']) / len(f['train']) for f in folds])

    rows, preds = [], []
    for name in MODELS:
        try:
            feats = extract(name, paths, device)
        except Exception as e:  # e.g. no internet, model not available
            print(f'  {name}: SKIPPED ({type(e).__name__}: {e})')
            continue
        for i, f in enumerate(folds, 1):
            m, C, probs, y = probe(feats, f, aug.get(i, []))
            rows.append(dict(model=name, fold=f['name'], C=C, **m))
            preds.append(pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES]).assign(
                model=name, fold=f['name'], y_true=y, y_pred=probs.argmax(1),
                path=[p for p, _ in f['test']]))
            print(f'  {name} | {f["name"]} | C={C} | acc={m["accuracy"]:.4f} f1={m["f1_macro"]:.4f} '
                  f'ece={m["ece"]:.3f}')
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'pilot_folds.csv'), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, 'pilot_predictions.csv'), index=False)

    print('\n=== Linear probe on frozen features: mean ± SD over the 5 official folds (%) ===')
    table = []
    for name, g in res.groupby('model', sort=False):
        row = {'model': name}
        for k in ('accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_macro_ovr', 'ece'):
            row[k] = f'{g[k].mean() * 100:.2f} ± {g[k].std(ddof=1) * 100:.2f}'
        if name != REFERENCE and REFERENCE in set(res['model']):
            ref = res[res['model'] == REFERENCE].set_index('fold').loc[g['fold']]
            diff, p = corrected_ttest(g['accuracy'].to_numpy(), ref['accuracy'].to_numpy(), ratio)
            row['acc_vs_reference'] = f'{diff * 100:+.2f} (p={p:.3f})'
        table.append(row)
    print(pd.DataFrame(table).to_string(index=False))
    print('\nFor context, fine-tuned results reported in the paper (accuracy):')
    for k, (m, s) in PAPER.items():
        print(f'  {k}: {m:.2f} ± {s:.2f}')
    print(f'\nUSE_AUG = {USE_AUG}. Files: {OUT}/pilot_folds.csv, pilot_predictions.csv, features_*.npz')


main()
