"""
Pilot 2 on MSLD v2.0 (official patient-independent folds, duplicates screened as in the paper), building on pilot 1
where frozen MONET and SigLIP features beat the ImageNet DenseNet201.

  A  fusion: logistic regression on concatenated MONET + SigLIP features, and the average of the two calibrated probes
  B  the same probes trained with the fold-matched augmented training images added (USE_AUG)
  C  attention heads on the frozen PATCH TOKENS (SigLIP 27x27, MONET 16x16), trained with 3 seeds:
       mean    mean of the tokens -> linear classifier          (control)
       attn    one learned query attends over the tokens
       region  as attn, plus the region-aware log prior of the paper: r_j = sigmoid(w^T t_j + b), logits + log r_j
  D  temperature scaling on the validation split; ECE before and after

Kaggle: Accelerator = GPU T4, Internet = On, input = the MSLD v2.0 dataset. Paste the whole file into ONE cell and
start it with "Save Version -> Save & Run All". Expected time: about 1.5-2 h (most of it extracting the ~37k
augmented images; set USE_AUG = False for a ~40 min run without part B).
"""
import glob
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# Settings
# =============================================================================
BACKBONES = ['monet', 'siglip_so400m']
USE_AUG = True
FOLDS_DIR = None            # None = search /kaggle/input
AUG_DIR = None
OUT = '/kaggle/working/pilot2'
NEAR_THRESHOLD = 10
C_GRID = [0.001, 0.01, 0.1, 1.0]
HEAD_SEEDS = [0, 1, 2]
HEAD = dict(dim=256, heads=4, dropout=0.1, head_dropout=0.3, lr=1e-3, weight_decay=0.05, batch=32,
            max_epochs=100, patience=12, token_keep=0.75)
BATCH = 32
REFERENCE = 'LP siglip_so400m'     # best model of pilot 1
PAPER = {'pilot 1, LP DenseNet201 (ImageNet)': (77.42, 7.44), 'pilot 1, LP MONET': (84.47, 4.99),
         'pilot 1, LP SigLIP': (86.17, 7.22), 'paper, DenseNet201 fine-tuned + augmented': (81.55, 5.10),
         'Ali et al. 2024 benchmark': (83.59, 2.11)}

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
ALIASES = {'monkeypox': 'Mpox', 'normal': 'Healthy'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)
AUG_FOLD_RE = re.compile(r'^fold[\s_-]*(\d+)[\s_-]*aug$', re.IGNORECASE)


# =============================================================================
# Data (identical to pilot 1)
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
# Features: pooled embedding for every image, patch tokens for the original images
# =============================================================================
def load_backbone(name, device):
    """returns f(list of PIL) -> (pooled embedding, patch tokens), identical to pilot 1 for the pooled part"""
    from transformers import AutoProcessor
    if name == 'monet':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('suinleelab/monet').eval().to(device)
        proc = AutoProcessor.from_pretrained('suinleelab/monet')

        def f(imgs):
            out = model.vision_model(pixel_values=proc(images=imgs, return_tensors='pt')['pixel_values'].to(device))
            return model.visual_projection(out.pooler_output), out.last_hidden_state[:, 1:]  # drop the CLS token
        return f
    if name == 'siglip_so400m':
        from transformers import AutoModel
        model = AutoModel.from_pretrained('google/siglip-so400m-patch14-384').eval().to(device)
        proc = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')

        def f(imgs):
            out = model.vision_model(pixel_values=proc(images=imgs, return_tensors='pt')['pixel_values'].to(device))
            return out.pooler_output, out.last_hidden_state
        return f
    raise ValueError(name)


def extract(name, paths, token_keys, device):
    import torch
    fn = load_backbone(name, device)
    pooled, tokens = {}, {}
    todo = {}
    for p in paths:
        todo.setdefault(md5(p), p)
    keys, t0 = list(todo), time.time()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
        for i in range(0, len(keys), BATCH):
            batch = keys[i:i + BATCH]
            pe, tk = fn([Image.open(todo[k]).convert('RGB') for k in batch])
            pe, tk = pe.float().cpu().numpy(), tk.half().cpu().numpy()
            for j, k in enumerate(batch):
                pooled[k] = pe[j]
                if k in token_keys:
                    tokens[k] = tk[j]
    print(f'  {name}: {len(keys)} images in {time.time() - t0:.0f}s; pooled dim {pe.shape[1]}, '
          f'tokens {tk.shape[1]}x{tk.shape[2]} for {len(tokens)} original images')
    np.savez(os.path.join(OUT, f'pooled_{name}.npz'), keys=np.array(list(pooled)), feats=np.stack(list(pooled.values())))
    del fn
    torch.cuda.empty_cache()
    return pooled, tokens


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


# =============================================================================
# A, B: linear probes
# =============================================================================
def linear_probe(Xtr, ytr, Xva, yva, Xte):
    """C chosen on validation balanced accuracy, then temperature fitted on validation; returns raw/calibrated test
    probabilities and the calibrated validation probabilities"""
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
    T = fit_temperature(full_logits(clf, Xva), yva)
    lte = full_logits(clf, Xte)
    return softmax(lte), softmax(lte / T), best[1], T


# =============================================================================
# C: attention heads on frozen patch tokens
# =============================================================================
def make_head(d_in, mode):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            d, h = HEAD['dim'], HEAD['heads']
            self.h, self.dh = h, d // h
            self.norm, self.proj = nn.LayerNorm(d_in), nn.Linear(d_in, d)
            if mode != 'mean':
                self.query = nn.Parameter(torch.randn(h, self.dh) * 0.02)
                self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
                self.attn_drop = nn.Dropout(HEAD['dropout'])
            if mode == 'region':
                self.region = nn.Linear(d, 1)
            self.drop, self.out = nn.Dropout(HEAD['head_dropout']), nn.Linear(d, len(CLASSES))

        def forward(self, x):                                   # x: (B, N, d_in)
            t = self.proj(self.norm(x))
            if mode == 'mean':
                z = t.mean(1)
            else:
                B, N, _ = t.shape
                k = self.k(t).view(B, N, self.h, self.dh)
                v = self.v(t).view(B, N, self.h, self.dh)
                s = torch.einsum('hd,bnhd->bhn', self.query, k) / math.sqrt(self.dh)
                if mode == 'region':                            # r_j = sigmoid(w^T t_j + b), logits + log r_j
                    s = s + torch.log(torch.sigmoid(self.region(t)).squeeze(-1) + 1e-6)[:, None, :]
                a = self.attn_drop(s.softmax(-1))
                z = self.o(torch.einsum('bhn,bnhd->bhd', a, v).reshape(B, -1))
            return self.out(self.drop(z))
    return Head()


def train_head(mode, Ttr, ytr, Tva, yva, Tte, seed, device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_head(Ttr.shape[-1], mode).to(device)
    w = len(ytr) / (len(CLASSES) * np.maximum(np.bincount(ytr, minlength=len(CLASSES)), 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=HEAD['lr'], weight_decay=HEAD['weight_decay'])
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Tva_t, yva_t, Tte_t = to(Tva), torch.tensor(yva, device=device), to(Tte)
    best, best_state, wait = np.inf, None, 0
    n_keep = max(1, int(round(Ttr.shape[1] * HEAD['token_keep'])))
    for epoch in range(HEAD['max_epochs']):
        model.train()
        for idx in np.array_split(np.random.permutation(len(ytr)), math.ceil(len(ytr) / HEAD['batch'])):
            keep = np.sort(np.random.choice(Ttr.shape[1], n_keep, replace=False))  # token dropout as augmentation
            xb = to(Ttr[idx][:, keep])
            loss = loss_fn(model(xb), torch.tensor(ytr[idx], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Tva_t), yva_t).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= HEAD['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lva = model(Tva_t).float().cpu().numpy()
        lte = model(Tte_t).float().cpu().numpy()
    T = fit_temperature(lva, yva)
    return softmax(lte), softmax(lte / T), epoch + 1


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
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}; torch {torch.__version__}')
    folds = [screen(f) for f in discover_folds(FOLDS_DIR or find_dir(FOLD_RE, exclude_aug=True))]
    aug = discover_aug(AUG_DIR or find_dir(AUG_FOLD_RE, exclude_aug=False)) if USE_AUG else {}
    for i, f in enumerate(folds, 1):
        held = {md5(p) for s in ('val', 'test') for p, _ in f[s]}
        assert not any(md5(p) in held for p, _ in aug.get(i, [])), f'{f["name"]}: augmented image in val/test'
    originals = sorted({p for f in folds for s in ('train', 'val', 'test') for p, _ in f[s]})
    token_keys = {md5(p) for p in originals}
    all_paths = originals + sorted({p for rows in aug.values() for p, _ in rows})
    ratio = np.mean([len(f['test']) / len(f['train']) for f in folds])

    pooled, tokens = {}, {}
    for name in BACKBONES:
        pooled[name], tokens[name] = extract(name, all_paths, token_keys, device)

    rows, preds = [], []

    def record(config, fold, y, probs_raw, probs_cal, paths, **extra):
        m = metrics(y, probs_cal)
        m['ece_uncalibrated'] = ece(probs_raw, y)
        rows.append(dict(config=config, fold=fold, **m, **extra))
        preds.append(pd.DataFrame(probs_cal, columns=[f'p_{c}' for c in CLASSES]).assign(
            config=config, fold=fold, y_true=y, y_pred=probs_cal.argmax(1), path=paths))
        print(f'  {config:34s} | {fold} | acc={m["accuracy"]:.4f} f1={m["f1_macro"]:.4f} '
              f'ece {m["ece_uncalibrated"]:.3f}->{m["ece"]:.3f}')

    for i, f in enumerate(folds, 1):
        y = {s: np.array([c for _, c in f[s]]) for s in ('train', 'val', 'test')}
        test_paths = [p for p, _ in f['test']]
        for with_aug in ([False, True] if USE_AUG else [False]):
            extra_rows = aug.get(i, []) if with_aug else []
            ytr = np.concatenate([y['train'], np.array([c for _, c in extra_rows], dtype=int)])
            suffix = ' +aug' if with_aug else ''

            def X(name, rows_):
                return np.stack([pooled[name][md5(p)] for p, _ in rows_])
            cal = {}
            for name in BACKBONES + ['concat']:
                get = (lambda r: np.concatenate([X(b, r) for b in BACKBONES], 1)) if name == 'concat' \
                    else (lambda r, n=name: X(n, r))
                raw, calp, C, T = linear_probe(get(f['train'] + extra_rows), ytr, get(f['val']), y['val'],
                                               get(f['test']))
                cal[name] = calp
                record(f'LP {name}{suffix}', f['name'], y['test'], raw, calp, test_paths, C=C, T=T)
            ens = np.mean([cal[b] for b in BACKBONES], axis=0)
            record(f'ensemble {"+".join(BACKBONES)}{suffix}', f['name'], y['test'], ens, ens, test_paths)

        for name in BACKBONES:
            T = {s: np.stack([tokens[name][md5(p)] for p, _ in f[s]]).astype(np.float32) for s in ('train', 'val', 'test')}
            for mode in ('mean', 'attn', 'region'):
                raw_s, cal_s, ep = zip(*[train_head(mode, T['train'], y['train'], T['val'], y['val'], T['test'],
                                                    seed, device) for seed in HEAD_SEEDS])
                # metrics averaged over seeds (not an ensemble), so the head is compared like for like
                ms = [metrics(y['test'], c) for c in cal_s]
                m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
                m['ece_uncalibrated'] = float(np.mean([ece(r, y['test']) for r in raw_s]))
                rows.append(dict(config=f'head-{mode} {name}', fold=f['name'], epochs=float(np.mean(ep)), **m))
                preds.append(pd.DataFrame(cal_s[0], columns=[f'p_{c}' for c in CLASSES]).assign(
                    config=f'head-{mode} {name}', fold=f['name'], y_true=y['test'],
                    y_pred=cal_s[0].argmax(1), path=test_paths))
                print(f'  {"head-" + mode + " " + name:34s} | {f["name"]} | acc={m["accuracy"]:.4f} '
                      f'f1={m["f1_macro"]:.4f} (mean of {len(HEAD_SEEDS)} seeds, ~{np.mean(ep):.0f} epochs)')

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'pilot2_folds.csv'), index=False)
    pd.concat(preds).to_csv(os.path.join(OUT, 'pilot2_predictions.csv'), index=False)

    table = []
    if REFERENCE not in set(res['config']):
        raise RuntimeError(f'{REFERENCE} missing; results are in {OUT}/pilot2_folds.csv')
    ref = res[res['config'] == REFERENCE].set_index('fold')
    for config, g in res.groupby('config', sort=False):
        row = {'config': config}
        for k in ('accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_macro_ovr'):
            row[k] = f'{g[k].mean() * 100:.2f} ± {g[k].std(ddof=1) * 100:.2f}'
        row['ece raw->cal'] = f'{g["ece_uncalibrated"].mean() * 100:.1f} -> {g["ece"].mean() * 100:.1f}'
        if config != REFERENCE:
            row['_diff'], row['_p'] = corrected_ttest(g['accuracy'].to_numpy(),
                                                      ref.loc[g['fold'], 'accuracy'].to_numpy(), ratio)
        table.append(row)
    tab = pd.DataFrame(table)
    has_p = tab['_p'].notna()
    tab.loc[has_p, 'p_holm'] = holm(tab.loc[has_p, '_p'].to_numpy())
    tab['acc vs ' + REFERENCE] = [f'{d * 100:+.2f} (p={p:.3f}, Holm {ph:.3f})' if pd.notna(p) else ''
                                  for d, p, ph in zip(tab['_diff'], tab['_p'], tab.get('p_holm'))]
    print('\n=== Pilot 2: mean ± SD over the 5 official folds (%); probes and heads calibrated on validation ===')
    print(tab.drop(columns=['_diff', '_p', 'p_holm']).to_string(index=False))
    print('\nFor context (accuracy):')
    for k, (m, s) in PAPER.items():
        print(f'  {k}: {m:.2f} ± {s:.2f}')
    print(f'\nFiles: {OUT}/pilot2_folds.csv, pilot2_predictions.csv, pooled_*.npz')


main()
