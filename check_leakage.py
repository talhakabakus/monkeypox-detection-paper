"""
Duplicate and train/test leakage check for the MSLD v2.0 pickles used in the paper.

Reproduces the exact split of the notebook
(train_test_split(X, y, test_size=0.3, random_state=25)) and reports:
  1. How many unique images the 3,775-image pickle actually contains
     (exact duplicates, after quantising back to uint8).
  2. Per-class unique counts vs. the original MSLD v2.0 counts.
  3. How many test images have an exact / near duplicate in the training set
     (near = perceptual hash, also checked against the 8 flip/rotation variants
     so that augmented copies are caught).
  4. Duplicate groups that carry conflicting labels.
  5. (optional) Where the duplicates come from in the raw dataset folders.

Usage on Kaggle (default paths match the notebook):
    python check_leakage.py
    python check_leakage.py --raw-dir /kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20
Inside a notebook cell, paste the file and call main() directly, e.g.:
    main(['--raw-dir', '/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20'])
"""
import argparse
import hashlib
import os
import pickle
from collections import Counter, defaultdict
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']
# Number of original images per class in MSLD v2.0 (Ali et al., 2024)
MSLD_V2_ORIGINAL = {'Chickenpox': 75, 'Cowpox': 66, 'HFMD': 161, 'Healthy': 114, 'Measles': 55, 'Monkeypox': 284}
RANDOM_STATE = 25
TRAIN_TEST_SPLIT = .3
HASH_GRID = 16  # perceptual hash resolution -> 16 x 16 = 256 bits


def to_uint8(X):
    X = np.asarray(X)
    if X.dtype != np.uint8:
        scale = 255.0 if X.max() <= 1.0 + 1e-6 else 1.0
        X = np.clip(np.rint(X * scale), 0, 255).astype(np.uint8)
    return X


def exact_hashes(X_u8):
    return np.array([hashlib.md5(img.tobytes()).hexdigest() for img in X_u8])


def average_hash(X_u8, grid=HASH_GRID):
    """Block-mean average hash on the grayscale image -> (n, grid*grid) bool."""
    gray = X_u8.astype(np.float32).mean(axis=-1)
    n, h, w = gray.shape
    bh, bw = h // grid, w // grid
    gray = gray[:, :bh * grid, :bw * grid].reshape(n, grid, bh, grid, bw).mean(axis=(2, 4))
    return (gray > gray.mean(axis=(1, 2), keepdims=True)).reshape(n, -1)


def dihedral_variants(X_u8):
    """The 8 flip/rotation variants of each image (identity first)."""
    for k in range(4):
        rot = np.rot90(X_u8, k, axes=(1, 2))
        yield rot
        yield rot[:, :, ::-1]


def hamming_matrix(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    return A @ (1 - B).T + (1 - A) @ B.T


def nearest_train_distance(X_test_u8, train_hash):
    """Minimum Hamming distance of each test image (any dihedral variant) to any training image."""
    best = np.full(len(X_test_u8), np.inf, dtype=np.float32)
    best_idx = np.zeros(len(X_test_u8), dtype=int)
    for variant in dihedral_variants(X_test_u8):
        d = hamming_matrix(average_hash(np.ascontiguousarray(variant)), train_hash)
        idx = d.argmin(axis=1)
        dist = d[np.arange(len(d)), idx]
        better = dist < best
        best[better] = dist[better]
        best_idx[better] = idx[better]
    return best, best_idx


def report_raw_dir(raw_dir):
    """Count .jpg files per folder prefix and how often the same file content repeats."""
    print(f'\n=== Raw dataset folder: {raw_dir} ===')
    files = glob(f'{raw_dir}/**/*.jpg', recursive=True)
    print(f'Total .jpg files: {len(files)}')
    if not files:
        return
    per_prefix = Counter(os.path.relpath(os.path.dirname(os.path.dirname(f)), raw_dir) for f in files)
    print('Files per folder (parent of the class folder):')
    for prefix, count in sorted(per_prefix.items()):
        print(f'  {prefix}: {count}')
    content = Counter(hashlib.md5(open(f, 'rb').read()).hexdigest() for f in files)
    mult = Counter(content.values())
    print(f'Unique file contents: {len(content)}')
    print(f'Copies per unique file -> number of files: {dict(sorted(mult.items()))}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', default='/kaggle/input/monkeypox/monkeypox_X_128.pkl')
    parser.add_argument('--y', default='/kaggle/input/monkeypox/monkeypox_y.pkl')
    parser.add_argument('--raw-dir', default=None, help='optional: raw MSLD v2.0 folder to inspect')
    parser.add_argument('--threshold', type=int, default=10,
                        help='max Hamming distance (out of 256 bits) to count as a near duplicate')
    parser.add_argument('--out', default='leakage_report')
    # parse_known_args: ignore the '-f kernel.json' argument Jupyter/Kaggle passes when pasted into a notebook cell
    args, _ = parser.parse_known_args(argv)
    os.makedirs(args.out, exist_ok=True)

    with open(args.x, 'rb') as f:
        X = pickle.load(f)
    with open(args.y, 'rb') as f:
        y = np.asarray(pickle.load(f))
    X_u8 = to_uint8(X)
    print(f'X: {X_u8.shape}, y: {y.shape}')

    # 1-2. Exact duplicates in the whole pickle
    hashes = exact_hashes(X_u8)
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    print(f'\n=== Exact duplicates in the whole dataset ===')
    print(f'Total images: {len(hashes)}, unique images: {len(groups)}')
    print(f'Copies per unique image -> number of unique images: '
          f'{dict(sorted(Counter(len(g) for g in groups.values()).items()))}')

    rows = []
    for c, name in enumerate(TARGET_CLASSES):
        mask = y == c
        rows.append({'Class': name, 'Images': int(mask.sum()), 'Unique': len(set(hashes[mask])),
                     'MSLD v2.0 original': MSLD_V2_ORIGINAL[name]})
    per_class = pd.DataFrame(rows)
    per_class['Images / original'] = (per_class['Images'] / per_class['MSLD v2.0 original']).round(2)
    print(per_class.to_string(index=False))

    conflicts = [g for g in groups.values() if len(set(y[g])) > 1]
    print(f'Duplicate groups with conflicting labels: {len(conflicts)}')

    # 3. Reproduce the notebook split (indices instead of arrays)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)
    train_hash_set = set(hashes[idx_train])
    exact_leak = np.array([hashes[i] in train_hash_set for i in idx_test])

    train_ahash = average_hash(X_u8[idx_train])
    near_dist, near_idx = nearest_train_distance(X_u8[idx_test], train_ahash)
    near_leak = near_dist <= args.threshold

    print(f'\n=== Train/test leakage (test_size={TRAIN_TEST_SPLIT}, random_state={RANDOM_STATE}) ===')
    print(f'Train: {len(idx_train)}, test: {len(idx_test)}')
    print(f'Test images with an EXACT copy in train: {exact_leak.sum()} ({exact_leak.mean():.1%})')
    print(f'Test images with a NEAR copy in train (Hamming <= {args.threshold}, incl. flips/rotations): '
          f'{near_leak.sum()} ({near_leak.mean():.1%})')
    print(f'Test images with no exact or near copy in train: {(~(exact_leak | near_leak)).sum()}')

    leak_rows = []
    for c, name in enumerate(TARGET_CLASSES):
        mask = y[idx_test] == c
        leak_rows.append({'Class': name, 'Test images': int(mask.sum()),
                          'Exact leak': int(exact_leak[mask].sum()), 'Near leak': int(near_leak[mask].sum())})
    leak_df = pd.DataFrame(leak_rows)
    print(leak_df.to_string(index=False))

    # Distance histogram helps choosing / defending the threshold
    print('\nNearest-train Hamming distance of test images (bins of 8 bits):')
    hist, edges = np.histogram(near_dist, bins=np.arange(0, 264, 8))
    for count, lo in zip(hist, edges[:-1]):
        if count:
            print(f'  {int(lo):3d}-{int(lo) + 7:3d}: {count}')

    # Save evidence
    per_class.to_csv(f'{args.out}/per_class_counts.csv', index=False)
    leak_df.to_csv(f'{args.out}/leakage_per_class.csv', index=False)
    pd.DataFrame({'test_index': idx_test, 'label': [TARGET_CLASSES[c] for c in y[idx_test]],
                  'exact_copy_in_train': exact_leak, 'nearest_train_index': idx_train[near_idx],
                  'nearest_train_label': [TARGET_CLASSES[c] for c in y[idx_train[near_idx]]],
                  'hamming_distance': near_dist.astype(int)}).to_csv(f'{args.out}/test_vs_train.csv', index=False)
    # Unique indices (first occurrence) to rebuild a de-duplicated dataset
    np.save(f'{args.out}/unique_indices.npy', np.array(sorted(g[0] for g in groups.values())))
    save_pairs_figure(X_u8, idx_test, idx_train[near_idx], near_dist, f'{args.out}/leak_examples.png')
    print(f'\nReport files written to {args.out}/')

    if args.raw_dir:
        report_raw_dir(args.raw_dir)


def save_pairs_figure(X_u8, test_idx, train_idx, dist, path, n=6):
    """Side-by-side test image / nearest training image, as visual evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = np.argsort(dist)[:n]
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    for col, k in enumerate(order):
        axes[0, col].imshow(X_u8[test_idx[k]])
        axes[0, col].set_title(f'test #{test_idx[k]}', fontsize=8)
        axes[1, col].imshow(X_u8[train_idx[k]])
        axes[1, col].set_title(f'train #{train_idx[k]} (d={int(dist[k])})', fontsize=8)
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
