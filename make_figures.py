"""
Figures and supplementary tables for the revised manuscript, built from the outputs of run_experiments.py.
CPU only (no GPU quota needed). Paste the whole file into ONE notebook cell and run it.

Finds every results folder (<model>/<fold>_seed<k>/metrics.json) under /kaggle/working and /kaggle/input.
Results of a previous notebook version are visible only after adding that version's output as input
("Add Input" -> "Your Work" -> noteboo sk -> version). Experiment 2 runs are recognised by "aug_train": true.

Writes to /kaggle/working/figures (PNG 300 dpi + PDF) and figures.zip:
  fig2_class_distribution     class counts of the 755 original images
  fig3_architecture           block diagram of the proposed model
  fig4_leak_examples          test/training copies under the leaky random split (only if the pkl is attached)
  fig5_fold_accuracy          per-fold accuracy of every model, mean +/- SD
  fig6_confusion_matrices     pooled, row-normalised confusion matrices, DenseNet201 vs. hybrid
  table_per_fold_accuracy.csv, table_per_class.csv (pooled per-class precision/recall/F1)
"""
import glob
import hashlib
import json
import logging
import os
import pickle
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

SEARCH_ROOTS = ['/kaggle/working', '/kaggle/input']
OUT = '/kaggle/working/figures'
PKL_X = '/kaggle/input/monkeypox/monkeypox_X_128.pkl'
PKL_Y = '/kaggle/input/monkeypox/monkeypox_y.pkl'

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']
LABELS = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Mpox']
ORIGINAL_COUNTS = {'Chickenpox': 75, 'Cowpox': 66, 'HFMD': 161, 'Healthy': 114, 'Measles': 55, 'Monkeypox': 284}
DISPLAY = {'densenet201': 'DenseNet201', 'resnet50v2': 'ResNet50V2', 'mobilenetv2': 'MobileNetV2',
           'efficientnetv2b0': 'EfficientNetV2-B0', 'convnexttiny': 'ConvNeXt-Tiny', 'hybrid_plain': 'hybrid_plain',
           'hybrid_no_region': 'hybrid_no_region', 'hybrid_no_feedback': 'hybrid_no_feedback',
           'hybrid_no_fusion': 'hybrid_no_fusion', 'hybrid_full': 'Hybrid (full)'}
EXPERIMENTS = {1: 'Experiment 1: original training images', 2: 'Experiment 2: + augmented training images'}

# Print palette: blue = DenseNet201, orange = proposed hybrid, gray = all other models; ink for text.
BLUE, ORANGE, GRAY, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#9a9893', '#0b0b0b', '#52514e', '#e4e3df'
COLOR = {'densenet201': BLUE, 'hybrid_full': ORANGE}

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
# STIX ships with matplotlib, so the layout is identical on Kaggle and locally (Times-like, matches the math)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'font.size': 9,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.spines.top': False, 'mathtext.fontset': 'stix', 'axes.spines.right': False, 'figure.dpi': 110,
})


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}.png / .pdf')


# =============================================================================
# Collect runs
# =============================================================================
def find_result_dirs():
    dirs = set()
    for root in SEARCH_ROOTS:
        for f in glob.glob(os.path.join(root, '**', 'metrics.json'), recursive=True):
            run_dir = os.path.dirname(f)
            if '_seed' in os.path.basename(run_dir):
                dirs.add(os.path.dirname(os.path.dirname(run_dir)))
    return sorted(d for d in dirs if not d.startswith(OUT))


def load_runs():
    metrics = []
    for res_dir in find_result_dirs():
        files = glob.glob(os.path.join(res_dir, '*', '*_seed*', 'metrics.json'))
        for f in files:
            m = json.load(open(f))
            m['experiment'] = 2 if m.get('aug_train') else 1
            m['run_dir'] = os.path.dirname(f)
            metrics.append(m)
        print(f'{res_dir}: {len(files)} runs')
    if not metrics:
        return pd.DataFrame(), pd.DataFrame()
    # the same run can be visible twice (e.g. working copy and notebook-output input): keep the first
    key = ['experiment', 'model', 'fold', 'seed']
    metrics = pd.DataFrame(metrics).drop_duplicates(key).reset_index(drop=True)
    preds = []
    for r in metrics.itertuples():
        p = os.path.join(r.run_dir, 'predictions.csv')
        if os.path.exists(p):
            preds.append(pd.read_csv(p, usecols=['y_true', 'y_pred']).assign(
                experiment=r.experiment, model=r.model, fold=r.fold, seed=r.seed))
    preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=key + ['y_true', 'y_pred'])
    return metrics, preds


# =============================================================================
# Figures
# =============================================================================
def fig_class_distribution():
    counts = pd.Series(ORIGINAL_COUNTS).rename(index=dict(zip(CLASSES, LABELS))).sort_values()
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.6)
    for y, v in enumerate(counts.values):
        ax.text(v + 4, y, f'{v} ({v / counts.sum():.1%})', va='center', fontsize=8, color=INK)
    ax.set_xlabel('Number of original images (total 755)')
    ax.set_xlim(0, counts.max() * 1.3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    save(fig, 'fig2_class_distribution')


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#ffffff', ec=MUTED, bold=False, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2',
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=INK, linespacing=1.25)

    def arrow(p, q, style='-|>', ls='-', color=MUTED, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=0.9, color=color,
                                     linestyle=ls, connectionstyle=f'arc3,rad={rad}'))

    light_blue, light_orange = '#e3eefb', '#fde9e0'
    y = 25
    box(0.5, y, 9, 9, 'Input\n224×224×3', fs=7.0)
    box(12, y, 12, 9, 'DenseNet201\n(ImageNet)\n7×7×1920', fc=light_blue)
    box(27, y, 12, 9, 'Reshape +\nlinear proj.\n$F$: 49×256')
    box(42.5, y - 1.5, 22, 12, 'Region-aware\nTransformer block\n(4 heads, pre-LN)\n'
        r'$r_j = \sigma(w_r^{\top} t_j + b_r)$', fc=light_orange)
    box(67.5, y, 17, 9, 'Gated fusion\n' r'$z = a \odot l + (1-a) \odot g$', fs=7.2)
    box(88, y, 11.5, 9, 'LN, dropout,\nsoftmax\n(6 classes)')
    box(42.5, 4, 22, 9, 'Feedback (FiLM-style)\n' r'$F_k = F \odot (1+\tanh W_\gamma g) + W_\beta g$',
        fc=light_orange, fs=7.0)
    box(27, 38, 12, 6, '+ positional\nembedding $P$')

    arrow((9.8, y + 4.5), (11.7, y + 4.5))
    arrow((24.3, y + 4.5), (26.7, y + 4.5))
    arrow((39.3, y + 4.5), (42.2, y + 4.5))
    arrow((33, 37.7), (33, 34.3))
    arrow((64.8, y + 4.5), (67.2, y + 4.5))
    ax.text(66.0, y + 1.6, '$g$', fontsize=8, ha='center')
    arrow((84.8, y + 4.5), (87.7, y + 4.5))
    # feedback loop: block output -> feedback -> block input (K = 2 iterations, shared weights)
    arrow((59, y - 1.8), (59, 13.3), color=ORANGE)
    arrow((48, 13.3), (48, y - 1.8), color=ORANGE)
    ax.text(60, 18, '$g_{k-1}$', fontsize=8, color=ORANGE)
    ax.text(47, 18, '$F_k$ (k = 1, 2)', fontsize=7.5, color=ORANGE, ha='right')
    ax.text(53.5, 1.0, 'shared block weights, applied K + 1 = 3 times', fontsize=7, color=MUTED,
            ha='center', style='italic')
    arrow((33, y - 0.3), (42.2, 8.5), ls='--', rad=0.25)
    ax.text(27.5, 12, 'CNN tokens $F$', fontsize=7.5, color=MUTED, ha='right')
    # local descriptor l = mean(F) to the fusion
    ax.add_patch(FancyArrowPatch((36, y + 9.3), (76, y + 9.4), arrowstyle='-|>', mutation_scale=8, lw=0.9,
                                 color=MUTED, linestyle='--', connectionstyle='arc3,rad=-0.18'))
    ax.text(56, 42.3, '$l$ = mean($F$)', fontsize=7.5, color=MUTED, ha='center')
    save(fig, 'fig3_architecture')


def fig_leak_examples():
    if not (os.path.exists(PKL_X) and os.path.exists(PKL_Y)):
        print('  fig4 skipped: attach the monkeypox pkl dataset to draw it')
        return
    from sklearn.model_selection import train_test_split
    X = pickle.load(open(PKL_X, 'rb'))
    y = np.asarray(pickle.load(open(PKL_Y, 'rb')))
    X = np.asarray(X)
    if X.dtype != np.uint8:
        X = np.clip(np.rint(X * (255.0 if X.max() <= 1.0 + 1e-6 else 1.0)), 0, 255).astype(np.uint8)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=25)
    first_train = {}
    for i in idx_tr:
        first_train.setdefault(hashlib.md5(X[i].tobytes()).hexdigest(), i)
    pairs, seen_cls = [], set()
    for i in idx_te:  # one example per class
        j = first_train.get(hashlib.md5(X[i].tobytes()).hexdigest())
        if j is not None and y[i] not in seen_cls:
            pairs.append((i, j))
            seen_cls.add(y[i])
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.25 * len(pairs), 2.9))
    for c, (i, j) in enumerate(pairs):
        for r, k in enumerate((i, j)):
            axes[r, c].imshow(X[k])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
        axes[0, c].set_title(LABELS[int(y[i])], fontsize=8)
    axes[0, 0].set_ylabel('Test image', fontsize=8)
    axes[1, 0].set_ylabel('Training copy', fontsize=8)
    save(fig, 'fig4_leak_examples')


def fig_fold_accuracy(metrics):
    exps = sorted(metrics['experiment'].unique())
    order = (metrics[metrics['experiment'] == exps[0]].groupby('model')['accuracy'].mean()
             .sort_values(ascending=True).index.tolist())
    widths = [max(1, metrics[metrics['experiment'] == e]['model'].nunique()) for e in exps]
    fig, axes = plt.subplots(1, len(exps), figsize=(7.2, 0.32 * max(widths) + 1.0), sharex=True,
                             gridspec_kw={'width_ratios': [1] * len(exps)}, squeeze=False)
    for ax, e in zip(axes[0], exps):
        sub = metrics[metrics['experiment'] == e]
        models = [m for m in order if m in set(sub['model'])] + sorted(set(sub['model']) - set(order))
        for yi, m in enumerate(models):
            acc = sub.loc[sub['model'] == m, 'accuracy'].to_numpy() * 100
            col = COLOR.get(m, GRAY)
            mean, sd = acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0
            ax.plot([mean - sd, mean + sd], [yi, yi], color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.scatter(acc, np.full(len(acc), yi) + np.linspace(-0.12, 0.12, len(acc)), s=12, color=col,
                       alpha=0.55, lw=0, zorder=3)
            ax.scatter([mean], [yi], s=36, color=col, edgecolor='white', lw=1.2, zorder=4)
            ax.text(101, yi, f'{mean:.1f}', va='center', fontsize=7.5, color=INK)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([DISPLAY.get(m, m) for m in models])
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(EXPERIMENTS[e], fontsize=8.5, loc='left')
        ax.set_xlabel('Test accuracy per fold (%)')
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    lo = np.floor(metrics['accuracy'].min() * 100 / 5) * 5
    for ax in axes[0]:
        ax.set_xlim(lo, 100)
    fig.text(0.01, -0.02, 'Dots: individual folds; large dot: mean; bar: mean ± SD. Right: mean accuracy.',
             fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, 'fig5_fold_accuracy')


def fig_confusion(preds):
    combos = [(e, m) for e in sorted(preds['experiment'].unique()) for m in ('densenet201', 'hybrid_full')
              if len(preds[(preds['experiment'] == e) & (preds['model'] == m)])]
    if not combos:
        print('  fig6 skipped: no predictions of densenet201 / hybrid_full')
        return
    ncol = 2
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 3.3 * nrow), squeeze=False)
    for ax, (e, m) in zip(axes.ravel(), combos):
        p = preds[(preds['experiment'] == e) & (preds['model'] == m)]
        cm = confusion_matrix(p['y_true'], p['y_pred'], labels=range(len(CLASSES)))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, f'{norm[i, j] * 100:.0f}\n({cm[i, j]})', ha='center', va='center', fontsize=6.3,
                            color='white' if norm[i, j] > 0.55 else INK, linespacing=1.0)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=7)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(LABELS, fontsize=7)
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')
        acc = (p['y_true'] == p['y_pred']).mean() * 100
        ax.set_title(f'{DISPLAY[m]} – Exp. {e} (pooled n = {len(p)}, acc. {acc:.1f}%)', fontsize=8, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for ax in axes.ravel()[len(combos):]:
        ax.axis('off')
    fig.text(0.01, -0.01, 'Cells: row-normalised % (count). Predictions pooled over the five folds.',
             fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, 'fig6_confusion_matrices')


# =============================================================================
# Tables
# =============================================================================
def tables(metrics, preds):
    per_fold = metrics.pivot_table(index=['experiment', 'model'], columns='fold', values='accuracy') * 100
    per_fold['mean'] = per_fold.mean(axis=1)
    per_fold['sd'] = per_fold.drop(columns='mean').std(axis=1, ddof=1)
    per_fold = per_fold.round(2).sort_values(['experiment', 'mean'], ascending=[True, False])
    per_fold.to_csv(os.path.join(OUT, 'table_per_fold_accuracy.csv'))
    print('\n=== Accuracy per fold (%) ===')
    print(per_fold.to_string())

    rows = []
    for (e, m), p in preds.groupby(['experiment', 'model']):
        prec, rec, f1, sup = precision_recall_fscore_support(p['y_true'], p['y_pred'], labels=range(len(CLASSES)),
                                                             zero_division=0)
        for c in range(len(CLASSES)):
            rows.append({'experiment': e, 'model': m, 'class': LABELS[c], 'n': int(sup[c]),
                         'precision': round(prec[c] * 100, 2), 'recall': round(rec[c] * 100, 2),
                         'f1': round(f1[c] * 100, 2)})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(os.path.join(OUT, 'table_per_class.csv'), index=False)
    focus = per_class[per_class['model'].isin(['densenet201', 'hybrid_full'])]
    if len(focus):
        print('\n=== Pooled per-class results, DenseNet201 vs. hybrid (%) ===')
        print(focus.pivot_table(index=['experiment', 'class'], columns='model', values=['recall', 'f1']).to_string())


def main():
    os.makedirs(OUT, exist_ok=True)
    metrics, preds = load_runs()
    print('\nFigures:')
    fig_class_distribution()
    fig_architecture()
    fig_leak_examples()
    if metrics.empty:
        print('No experiment results found; fig5/fig6 and the tables need the results folders as input.')
    else:
        print('Runs per experiment and model:')
        print(metrics.groupby(['experiment', 'model']).size().unstack(0).fillna(0).astype(int).to_string())
        fig_fold_accuracy(metrics)
        fig_confusion(preds)
        tables(metrics, preds)
    shutil.make_archive(OUT, 'zip', OUT)
    print(f'\nAll files in {OUT}/ and {OUT}.zip')


main()
