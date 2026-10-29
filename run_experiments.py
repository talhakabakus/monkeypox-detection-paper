"""
Leakage-free experiments for the revised Mpox skin lesion paper.

What changed with respect to the original notebook
---------------------------------------------------
* Data: images are read directly from the official, patient-independent MSLD
  v2.0 fold folders (FOLDS/foldK/{Train,Valid,Test}/<class>/*.jpg), as in the
  benchmark of Ali et al. (2024). The test splits of different folds overlap,
  so results are reported as mean +/- std over folds. Exact and near
  (perceptual hash) duplicates of training images are removed from the
  valid/test splits of each fold and reported.
* Augmentation is applied to training batches only (Keras preprocessing layers
  that are inactive at inference).
* Optional (--aug-dir): the fold-matched, pre-augmented training images of MSLD
  v2.0 (FOLDS_AUG/foldK_AUG/Train) are added to the training split of fold K,
  as in the benchmark. Validation/test stay original. Training is aborted if an
  augmented image is an exact copy of a val/test image or belongs to a patient
  outside the fold's training split.
* Proposed model: the attention now operates on the h x w spatial tokens of the
  CNN feature map (not on a single pooled vector, where softmax is always 1),
  and region weighting, iterative feedback and gated fusion are implemented as
  defined below.
* Baselines are trained with the same resolution, augmentation, class weights,
  two-phase fine-tuning and early-stopping protocol.
* Results are written per (model, fold, seed) so that an interrupted Kaggle
  session can be resumed; --summarize computes mean +/- std, 95% CIs, the
  corrected resampled t-test (Nadeau & Bengio, 2003) on fold-level scores, and
  McNemar tests / paired bootstrap CIs on the predictions pooled over folds.

Proposed hybrid model (notation used in the paper)
--------------------------------------------------
  F  = W_p * reshape(CNN(x))                    F in R^{N x d}, N = h*w tokens
  T0 = Block(F + P)                             P: learnable positional embedding
  Region-aware attention inside Block, per head:
      r_j = sigmoid(w_r^T t_j + b_r)            region (lesion) score of token j
      A   = softmax(Q K^T / sqrt(d_h) + 1 log(r)^T)
  Feedback, k = 1..K (Block and feedback weights shared across iterations):
      g_{k-1} = mean_i T_{k-1,i}
      F_k = F * (1 + tanh(W_g g_{k-1})) + W_b g_{k-1}   (W_g, W_b initialised to 0)
      T_k = Block(F_k + P)
  Gated fusion of the local CNN descriptor l = mean_i F_i and the global
  descriptor g = mean_i T_{K,i}:
      a = sigmoid(W_a [l; g] + b_a),  z = a * l + (1 - a) * g
  y = softmax(W_c LN(z) + b_c)

Usage (Kaggle, GPU T4)
----------------------
Notebook: paste the whole file into ONE cell, set MODE (and optionally the other
settings) in the "Notebook settings" block at the end, and run the cell:
    MODE = 'check'      -> data integrity report only
    MODE = 'train'      -> all experiments (resumes finished runs), then the summary
    MODE = 'summarize'  -> tables and statistical tests from existing results
Command line:
    python run_experiments.py --folds-dir "<.../FOLDS>" --check-only
    python run_experiments.py --folds-dir "<.../FOLDS>"
    python run_experiments.py --summarize
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, ops
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']
CLASS_ALIASES = {'mpox': 'Monkeypox', 'monkeypox': 'Monkeypox', 'normal': 'Healthy'}
SPLIT_ALIASES = {'train': 'train', 'valid': 'val', 'val': 'val', 'validation': 'val', 'test': 'test'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_FOLDS_DIR = '/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20/Original Images/Original Images/FOLDS'

BACKBONES = {
    'densenet201': (keras.applications.DenseNet201, keras.applications.densenet.preprocess_input),
    'resnet50v2': (keras.applications.ResNet50V2, keras.applications.resnet_v2.preprocess_input),
    'mobilenetv2': (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    'efficientnetv2b0': (keras.applications.EfficientNetV2B0, keras.applications.efficientnet_v2.preprocess_input),
    'convnexttiny': (keras.applications.ConvNeXtTiny, keras.applications.convnext.preprocess_input),
}

# Baselines: backbone + GAP + linear head. Hybrids: ablations of the proposed model (leave-one-out).
MODEL_SPECS = {
    'densenet201': {'type': 'baseline', 'backbone': 'densenet201'},
    'resnet50v2': {'type': 'baseline', 'backbone': 'resnet50v2'},
    'mobilenetv2': {'type': 'baseline', 'backbone': 'mobilenetv2'},
    'efficientnetv2b0': {'type': 'baseline', 'backbone': 'efficientnetv2b0'},
    'convnexttiny': {'type': 'baseline', 'backbone': 'convnexttiny'},
    'hybrid_plain': {'type': 'hybrid', 'region': False, 'feedback_iters': 0, 'fusion': False},
    'hybrid_no_region': {'type': 'hybrid', 'region': False, 'feedback_iters': 2, 'fusion': True},
    'hybrid_no_feedback': {'type': 'hybrid', 'region': True, 'feedback_iters': 0, 'fusion': True},
    'hybrid_no_fusion': {'type': 'hybrid', 'region': True, 'feedback_iters': 2, 'fusion': False},
    'hybrid_full': {'type': 'hybrid', 'region': True, 'feedback_iters': 2, 'fusion': True},
}
REFERENCE_MODEL = 'hybrid_full'


# =============================================================================
# Data
# =============================================================================
def _match_class(folder_name):
    name = folder_name.strip()
    for c in CLASSES:
        if c.lower() == name.lower():
            return c
    return CLASS_ALIASES.get(name.lower())


def list_images(split_dir):
    rows = []
    for class_dir in sorted(glob.glob(os.path.join(split_dir, '*'))):
        cls = _match_class(os.path.basename(class_dir)) if os.path.isdir(class_dir) else None
        if cls is None:
            continue
        for f in sorted(glob.glob(os.path.join(class_dir, '*'))):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                rows.append({'path': f, 'label': CLASSES.index(cls)})
    return pd.DataFrame(rows, columns=['path', 'label'])


FOLD_RE = re.compile(r'^fold[\s_-]*\d+$', re.IGNORECASE)


def discover_folds(folds_dir):
    fold_dirs = sorted((d for d in glob.glob(os.path.join(folds_dir, '*'))
                        if os.path.isdir(d) and FOLD_RE.match(os.path.basename(d))),
                       key=lambda d: int(re.sub(r'\D', '', os.path.basename(d)) or 0))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold* folders under {folds_dir}')
    folds = []
    for fd in fold_dirs:
        fold = {'name': os.path.basename(fd)}
        for sd in glob.glob(os.path.join(fd, '*')):
            key = SPLIT_ALIASES.get(os.path.basename(sd).lower())
            if key and os.path.isdir(sd):
                fold[key] = list_images(sd)
        missing = {'train', 'val', 'test'} - set(fold)
        if missing:
            raise FileNotFoundError(f'{fd}: missing split folder(s) {missing}')
        folds.append(fold)
    return folds


class ImageCache:
    """Reads every file once, keyed by the MD5 of its bytes (the same image appears in all 5 folds)."""

    def __init__(self, size):
        self.size = size
        self.path_md5 = {}
        self.images = {}

    def md5(self, path):
        if path not in self.path_md5:
            with open(path, 'rb') as f:
                raw = f.read()
            h = hashlib.md5(raw).hexdigest()
            self.path_md5[path] = h
            if h not in self.images:
                img = tf.io.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, (self.size, self.size), antialias=True)
                self.images[h] = np.clip(np.rint(img.numpy()), 0, 255).astype(np.uint8)
        return self.path_md5[path]

    def load(self, df):
        hashes = [self.md5(p) for p in df['path']]
        X = np.stack([self.images[h] for h in hashes]) if hashes else np.zeros((0, self.size, self.size, 3), np.uint8)
        return X, np.array(hashes)


def average_hash(X_u8, grid=16):
    gray = X_u8.astype(np.float32).mean(axis=-1)
    n, h, w = gray.shape
    bh, bw = h // grid, w // grid
    gray = gray[:, :bh * grid, :bw * grid].reshape(n, grid, bh, grid, bw).mean(axis=(2, 4))
    return (gray > gray.mean(axis=(1, 2), keepdims=True)).reshape(n, -1).astype(np.float32)


def min_hamming(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf)
    return (A @ (1 - B).T + (1 - A) @ B.T).min(axis=1)


PATIENT_RE = re.compile(r'^([A-Za-z]+)_(\d+)_\d+')


def patient_id(path):
    m = PATIENT_RE.match(os.path.basename(path))
    return f'{m.group(1).lower()}_{m.group(2)}' if m else None


def prepare_fold(fold, cache, near_threshold, log, drop_near_dups=True):
    """Load a fold, drop exact (and by default near) duplicates of training images from val/test, log the checks."""
    data = {}
    for split in ('train', 'val', 'test'):
        X, h = cache.load(fold[split])
        data[split] = {'X': X, 'y': fold[split]['label'].to_numpy(), 'hash': h,
                       'path': fold[split]['path'].to_numpy()}

    report = {'fold': fold['name']}
    train_set = set(data['train']['hash'])
    for split, earlier in (('val', train_set), ('test', train_set | set(data['val']['hash']))):
        leak = np.array([h in earlier for h in data[split]['hash']], dtype=bool)
        report[f'{split}_exact_dups_removed'] = int(leak.sum())
        for k in ('X', 'y', 'hash', 'path'):
            data[split][k] = data[split][k][~leak]

    ahash_train = average_hash(data['train']['X'])
    for split in ('val', 'test'):
        near = min_hamming(average_hash(data[split]['X']), ahash_train) <= near_threshold
        report[f'{split}_near_dups_in_train'] = int(near.sum())
        if drop_near_dups:
            for k in ('X', 'y', 'hash', 'path'):
                data[split][k] = data[split][k][~near]

    pids = {s: {patient_id(p) for p in data[s]['path']} - {None} for s in data}
    if pids['train']:
        report['patient_overlap_train_test'] = len(pids['train'] & pids['test'])
        report['patient_overlap_train_val'] = len(pids['train'] & pids['val'])
    for s in data:
        report[f'n_{s}'] = len(data[s]['y'])
    log(json.dumps(report))
    return data, report


AUG_FOLD_RE = re.compile(r'^fold[\s_-]*(\d+)[\s_-]*aug$', re.IGNORECASE)


def fold_number(name):
    return int(re.sub(r'\D', '', name) or 0)


def discover_aug_train(aug_dir):
    """Fold number -> DataFrame of that fold's pre-augmented training images (FOLDS_AUG/foldK_AUG/Train/<class>)."""
    out = {}
    for d in glob.glob(os.path.join(aug_dir, '*')):
        m = AUG_FOLD_RE.match(os.path.basename(d))
        if not (m and os.path.isdir(d)):
            continue
        train_dirs = [s for s in glob.glob(os.path.join(d, '*'))
                      if os.path.isdir(s) and SPLIT_ALIASES.get(os.path.basename(s).lower()) == 'train']
        if len(train_dirs) != 1:
            raise FileNotFoundError(f'{d}: expected exactly one Train folder, found {train_dirs}')
        out[int(m.group(1))] = list_images(train_dirs[0])
    if not out:
        raise FileNotFoundError(f'No fold*_AUG folders under {aug_dir}')
    return out


def add_aug_train(data, aug_df, img_size, near_threshold, log, fold_name):
    """Append a fold's pre-augmented training images after checking they cannot come from its val/test split."""
    X, h = ImageCache(img_size).load(aug_df)  # separate cache: augmented images are not shared across folds
    paths = aug_df['path'].to_numpy()
    train_set = set(data['train']['hash'])
    held_out = set(data['val']['hash']) | set(data['test']['hash'])
    pids = {s: {patient_id(p) for p in data[s]['path']} - {None} for s in ('train', 'val', 'test')}
    aug_pid = [patient_id(p) for p in paths]
    aug_patients = set(aug_pid) - {None}

    report = {'fold': fold_name, 'n_aug_files': len(h),
              'aug_exact_in_val_test': int(sum(x in held_out for x in h)),
              'aug_exact_copies_of_train': int(sum(x in train_set for x in h)),
              'aug_patient_id_parsed': int(sum(p is not None for p in aug_pid)),
              'aug_patients': len(aug_patients),
              'aug_patients_not_in_train': len(aug_patients - pids['train']),
              'aug_patient_overlap_val_test': len(aug_patients & (pids['val'] | pids['test']))}
    ahash_aug = average_hash(X)
    for split in ('val', 'test'):
        report[f'{split}_near_dups_in_aug'] = int((min_hamming(average_hash(data[split]['X']), ahash_aug)
                                                   <= near_threshold).sum())
    log(json.dumps(report))
    if report['aug_exact_in_val_test'] or report['aug_patient_overlap_val_test'] or report['aug_patients_not_in_train']:
        raise RuntimeError(f'{fold_name}: augmented training images are not confined to the training split '
                           f'(see the report above); refusing to train on them.')
    if report['aug_patient_id_parsed'] < len(h):
        log(f'WARNING {fold_name}: patient id could not be read from {len(h) - report["aug_patient_id_parsed"]} '
            f'augmented file names; only the exact/near duplicate checks apply to them.')

    keep = np.array([x not in train_set for x in h], dtype=bool)  # originals are already in the training split
    tr = data['train']
    train = {'X': np.concatenate([tr['X'], X[keep]]),
             'y': np.concatenate([tr['y'], aug_df['label'].to_numpy()[keep]]),
             'hash': np.concatenate([tr['hash'], h[keep]]),
             'path': np.concatenate([tr['path'], paths[keep]]),
             'n_original': len(tr['y'])}
    return dict(data, train=train), report


def check_folds_partition(folds, cache, log):
    """How the test splits of the folds cover the dataset (in MSLD v2.0 they overlap, see the module docstring)."""
    test_sets = [set(cache.md5(p) for p in f['test']['path']) for f in folds]
    all_hashes = set().union(*[set(cache.md5(p) for s in ('train', 'val', 'test') for p in f[s]['path'])
                               for f in folds])
    union = set().union(*test_sets)
    overlaps = sum(len(a & b) for i, a in enumerate(test_sets) for b in test_sets[i + 1:])
    log(f'Unique images over all folds: {len(all_hashes)}; covered by a test split: {len(union)}; '
        f'images in more than one test split: {overlaps}')


# =============================================================================
# Model
# =============================================================================
class PositionalEmbedding(layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, input_shape[1], input_shape[2]),
                                   initializer=keras.initializers.TruncatedNormal(stddev=0.02), name='pos')

    def call(self, x):
        return x + self.pos


class RegionAwareAttention(layers.Layer):
    """Multi-head self-attention with an additive log region prior on the keys."""

    def __init__(self, dim, num_heads, dropout=0.0, use_region=True, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.use_region = use_region
        self.qkv = layers.Dense(3 * dim)
        self.proj = layers.Dense(dim)
        self.attn_drop = layers.Dropout(dropout)
        self.region = layers.Dense(1) if use_region else None

    def call(self, x, training=None):
        n = x.shape[1]
        qkv = ops.reshape(self.qkv(x), (-1, n, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = ops.einsum('bhid,bhjd->bhij', q, k) / np.sqrt(self.head_dim).astype('float32')
        if self.use_region:
            r = ops.sigmoid(self.region(x))                                  # (b, n, 1)
            logits = logits + ops.log(ops.transpose(r, (0, 2, 1)) + 1e-6)[:, None]  # (b, 1, 1, n)
        attn = self.attn_drop(ops.softmax(logits, axis=-1), training=training)
        out = ops.transpose(ops.einsum('bhij,bhjd->bhid', attn, v), (0, 2, 1, 3))
        return self.proj(ops.reshape(out, (-1, n, self.dim)))


class TransformerBlock(layers.Layer):
    """Pre-LN Transformer encoder block built on RegionAwareAttention."""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, use_region=True, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = RegionAwareAttention(dim, num_heads, dropout, use_region)
        self.drop1 = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = layers.Dense(mlp_dim, activation='gelu')
        self.mlp2 = layers.Dense(dim)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        x = x + self.drop1(self.attn(self.ln1(x), training=training), training=training)
        return x + self.drop2(self.mlp2(self.mlp1(self.ln2(x))), training=training)


class FeedbackModulation(layers.Layer):
    """F_k = F * (1 + tanh(W_g g)) + W_b g with g the mean Transformer token; identity at initialisation."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.gamma = layers.Dense(dim, kernel_initializer='zeros')
        self.beta = layers.Dense(dim, kernel_initializer='zeros')

    def call(self, features, tokens):
        g = ops.mean(tokens, axis=1, keepdims=True)
        return features * (1.0 + ops.tanh(self.gamma(g))) + self.beta(g)


class GatedFusion(layers.Layer):
    """z = a * local + (1 - a) * global, a = sigmoid(W [local; global] + b)."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(dim, activation='sigmoid')

    def call(self, local, global_):
        a = self.gate(ops.concatenate([local, global_], axis=-1))
        return a * local + (1.0 - a) * global_


def augmentation_block(seed):
    return keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical', seed=seed),
        layers.RandomRotation(0.1, seed=seed),
        layers.RandomZoom(0.1, seed=seed),
        layers.RandomTranslation(0.05, 0.05, seed=seed),
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
    ], name='augmentation')


def build_model(name, args, seed):
    spec = MODEL_SPECS[name]
    backbone_name = spec.get('backbone', args.hybrid_backbone)
    backbone_fn, preprocess = BACKBONES[backbone_name]
    shape = (args.img_size, args.img_size, 3)

    inputs = keras.Input(shape)
    x = augmentation_block(seed)(inputs)
    x = layers.Lambda(preprocess, name='preprocess')(x)
    base = backbone_fn(include_top=False, weights=None if args.no_pretrained else 'imagenet', input_shape=shape)
    base.trainable = False
    fmap = base(x, training=False)  # BatchNorm stays in inference mode also while fine-tuning

    if spec['type'] == 'baseline':
        z = layers.GlobalAveragePooling2D()(fmap)
    else:
        h, w, c = fmap.shape[1:]
        tokens = layers.Dense(args.embed_dim, name='token_projection')(layers.Reshape((h * w, c))(fmap))
        pos = PositionalEmbedding(name='pos_embedding')
        block = TransformerBlock(args.embed_dim, args.num_heads, args.mlp_dim, args.dropout,
                                 use_region=spec['region'], name='region_aware_transformer')
        t = block(pos(tokens))
        if spec['feedback_iters']:
            feedback = FeedbackModulation(args.embed_dim, name='feedback')
            for _ in range(spec['feedback_iters']):
                t = block(pos(feedback(tokens, t)))
        global_desc = layers.GlobalAveragePooling1D()(t)
        if spec['fusion']:
            z = GatedFusion(args.embed_dim, name='fusion')(layers.GlobalAveragePooling1D()(tokens), global_desc)
        else:
            z = global_desc
        z = layers.LayerNormalization(epsilon=1e-6)(z)

    z = layers.Dropout(args.head_dropout)(z)
    outputs = layers.Dense(len(CLASSES), activation='softmax', name='classifier')(z)
    return keras.Model(inputs, outputs, name=name), base


def unfreeze_top(base, fraction):
    base.trainable = True
    cut = int(len(base.layers) * (1 - fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cut and not isinstance(layer, layers.BatchNormalization)


# =============================================================================
# Training / evaluation
# =============================================================================
def make_dataset(X, y, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, keras.utils.to_categorical(y, len(CLASSES))))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(tf.data.AUTOTUNE)


def compute_metrics(y, probs):
    pred = probs.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    m = {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision_macro': precision_score(y, pred, labels=labels, average='macro', zero_division=0),
        'recall_macro': recall_score(y, pred, labels=labels, average='macro', zero_division=0),
        'f1_macro': f1_score(y, pred, labels=labels, average='macro', zero_division=0),
        'f1_weighted': f1_score(y, pred, labels=labels, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y, pred),
        'mcc': matthews_corrcoef(y, pred),
    }
    try:
        present = sorted(set(y))
        m['auc_macro_ovr'] = roc_auc_score(y, probs[:, present] / probs[:, present].sum(1, keepdims=True),
                                           multi_class='ovr', average='macro', labels=present) \
            if len(present) > 2 else float('nan')
    except ValueError:
        m['auc_macro_ovr'] = float('nan')
    return {k: float(v) for k, v in m.items()}


def measure_latency(model, img_size, runs=100):
    """Median single-image latency (batch 1, after warm-up) and throughput at batch 32."""
    infer = tf.function(lambda t: model(t, training=False))
    result = {}
    for batch, key in ((1, 'latency_ms_batch1'), (32, 'throughput_img_per_s_batch32')):
        x = tf.random.uniform((batch, img_size, img_size, 3), 0, 255)
        for _ in range(10):
            infer(x).numpy()
        times = []
        for _ in range(runs if batch == 1 else max(runs // 5, 5)):
            t0 = time.perf_counter()
            infer(x).numpy()
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        result[key] = med * 1000 if batch == 1 else batch / med
    return result


def fit_phase(model, train_ds, val_ds, lr, epochs, class_weight, patience):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience // 2), verbose=1),
    ]
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                  callbacks=callbacks, verbose=2)
    return pd.DataFrame(h.history)


def run_dir_of(args, name, fold_name, seed):
    return os.path.join(args.out, name, f'{fold_name}_seed{seed}')


def run_one(name, fold_idx, fold_name, data, seed, args, external, log):
    run_dir = run_dir_of(args, name, fold_name, seed)
    if os.path.exists(os.path.join(run_dir, 'metrics.json')):
        log(f'[skip] {name} / {fold_name} / seed {seed} already done')
        return
    os.makedirs(run_dir, exist_ok=True)
    log(f'\n===== {name} | {fold_name} | seed {seed} =====')

    keras.utils.set_random_seed(seed)
    model, base = build_model(name, args, seed)
    tr, va, te = data['train'], data['val'], data['test']
    train_ds = make_dataset(tr['X'], tr['y'], args.batch_size, True, seed)
    val_ds = make_dataset(va['X'], va['y'], args.batch_size, False, seed)
    present = np.unique(tr['y'])
    weights = compute_class_weight('balanced', classes=present, y=tr['y'])
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    t0 = time.time()
    hist = [fit_phase(model, train_ds, val_ds, args.lr_head, args.epochs_head, class_weight, args.patience)
            .assign(phase='head')]
    if args.finetune_fraction > 0 and args.epochs_finetune > 0:
        unfreeze_top(base, args.finetune_fraction)
        hist.append(fit_phase(model, train_ds, val_ds, args.lr_finetune, args.epochs_finetune, class_weight,
                              args.patience).assign(phase='finetune'))
    train_seconds = time.time() - t0
    pd.concat(hist, ignore_index=True).to_csv(os.path.join(run_dir, 'history.csv'), index_label='epoch')

    probs = model.predict(te['X'].astype(np.float32), batch_size=args.batch_size, verbose=0)
    metrics = compute_metrics(te['y'], probs)
    metrics.update({'model': name, 'fold': fold_name, 'seed': seed, 'n_test': int(len(te['y'])),
                    'n_train': int(tr.get('n_original', len(tr['y']))), 'n_train_images': int(len(tr['y'])),
                    'aug_train': 'n_original' in tr,
                    'train_seconds': train_seconds, 'params': int(model.count_params())})
    pred_df = pd.DataFrame(probs, columns=[f'p_{c}' for c in CLASSES])
    pred_df.insert(0, 'y_pred', probs.argmax(1))
    pred_df.insert(0, 'y_true', te['y'])
    pred_df.insert(0, 'path', te['path'])
    pred_df.insert(0, 'fold', fold_name)
    pred_df.to_csv(os.path.join(run_dir, 'predictions.csv'), index=False)
    cm = confusion_matrix(te['y'], probs.argmax(1), labels=list(range(len(CLASSES))))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(run_dir, 'confusion_matrix.csv'))
    with open(os.path.join(run_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(te['y'], probs.argmax(1), labels=list(range(len(CLASSES))),
                                      target_names=CLASSES, digits=4, zero_division=0))

    if fold_idx == 0:
        metrics.update(measure_latency(model, args.img_size))
    if external is not None and len(external['y']):
        ext_probs = model.predict(external['X'].astype(np.float32), batch_size=args.batch_size, verbose=0)
        metrics.update({f'external_{k}': v for k, v in compute_metrics(external['y'], ext_probs).items()})
    if args.save_weights:
        model.save_weights(os.path.join(run_dir, 'model.weights.h5'))

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'{name} | {fold_name} | acc={metrics["accuracy"]:.4f} f1_macro={metrics["f1_macro"]:.4f} '
        f'auc={metrics["auc_macro_ovr"]:.4f} ({train_seconds:.0f}s)')
    keras.backend.clear_session()


def load_external(ext_dir, cache, msld_hashes_ahash, near_threshold, log):
    """External test set with <class>/ sub-folders; images that (near-)duplicate MSLD v2.0 are removed."""
    df = list_images(ext_dir)
    X, _ = cache.load(df)
    y = df['label'].to_numpy()
    d = min_hamming(average_hash(X), msld_hashes_ahash)
    keep = d > near_threshold
    log(f'External set {ext_dir}: {len(y)} images, removed {int((~keep).sum())} near-duplicates of MSLD v2.0, '
        f'class counts kept: {dict(Counter(CLASSES[c] for c in y[keep]))}')
    return {'X': X[keep], 'y': y[keep]}


def write_run_info(args, log):
    info = {'args': vars(args), 'tensorflow': tf.__version__, 'keras': keras.__version__,
            'python': platform.python_version(), 'cpu': platform.processor() or platform.machine(), 'gpus': []}
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            info['gpus'].append(tf.config.experimental.get_device_details(gpu).get('device_name', gpu.name))
        except Exception:  # noqa: BLE001 - details are best-effort
            info['gpus'].append(gpu.name)
    with open(os.path.join(args.out, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    log(f'Environment: {json.dumps({k: v for k, v in info.items() if k != "args"})}')


# =============================================================================
# Summary and statistics
# =============================================================================
def summarize(out_dir, reference=REFERENCE_MODEL, n_boot=2000, seed=0):
    from scipy import stats

    metric_files = glob.glob(os.path.join(out_dir, '*', '*', 'metrics.json'))
    if not metric_files:
        print(f'No results under {out_dir}')
        return
    res = pd.DataFrame([json.load(open(f)) for f in metric_files])
    metric_cols = [c for c in res.columns
                   if c not in ('model', 'fold', 'seed', 'n_test', 'n_train', 'n_train_images', 'aug_train') and
                   pd.api.types.is_numeric_dtype(res[c])]

    rows = []
    for name, g in res.groupby('model'):
        row = {'model': name, 'runs': len(g)}
        for c in metric_cols:
            v = g[c].dropna()
            if v.empty:
                continue
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row[f'{c}_mean'], row[f'{c}_std'] = v.mean(), v.std(ddof=1)
            row[f'{c}_ci95'] = half
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values('accuracy_mean', ascending=False)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

    pretty = summary[['model', 'runs']].copy()
    for c in ('accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'auc_macro_ovr',
              'mcc', 'latency_ms_batch1', 'params'):
        if f'{c}_mean' in summary:
            pretty[c] = [f'{m * 100:.2f} ± {s * 100:.2f}' if c not in ('latency_ms_batch1', 'params') else
                         f'{m:.2f}' if c == 'latency_ms_batch1' else f'{m / 1e6:.1f}M'
                         for m, s in zip(summary[f'{c}_mean'], summary[f'{c}_std'].fillna(0))]
    print('\n=== Mean ± std over folds (and seeds) ===')
    print(pretty.to_string(index=False))
    try:
        table = pretty.to_markdown(index=False)
    except ImportError:  # to_markdown needs the optional 'tabulate' package
        table = pretty.to_string(index=False)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(table)

    # Corrected resampled t-test on fold-level scores (accounts for overlapping training sets across folds)
    if reference in set(res['model']):
        ref_res = res[res['model'] == reference]
        cv_rows = []
        for name in sorted(set(res['model']) - {reference}):
            merged = ref_res.merge(res[res['model'] == name], on=['fold', 'seed'], suffixes=('_ref', '_oth'))
            if len(merged) < 2:
                continue
            ratio = (merged['n_test_ref'] / merged['n_train_ref']).mean() if 'n_train_ref' in merged else 1 / 4
            row = {'reference': reference, 'compared_to': name, 'n_folds': len(merged)}
            for c in ('accuracy', 'f1_macro'):
                d = merged[f'{c}_ref'] - merged[f'{c}_oth']
                var = d.var(ddof=1)
                t = d.mean() / np.sqrt((1 / len(d) + ratio) * var) if var > 0 else np.nan
                row[f'{c}_diff_mean'] = d.mean()
                row[f'{c}_p'] = 2 * stats.t.sf(abs(t), len(d) - 1) if np.isfinite(t) else np.nan
            cv_rows.append(row)
        cv_tests = pd.DataFrame(cv_rows)
        cv_tests.to_csv(os.path.join(out_dir, 'corrected_ttest.csv'), index=False)
        print(f'\n=== Corrected resampled t-test on fold-level scores against {reference} ===')
        print(cv_tests.round(4).to_string(index=False))

    # Predictions pooled over folds (the same image can appear in the test split of several folds)
    preds = []
    for f in glob.glob(os.path.join(out_dir, '*', '*', 'predictions.csv')):
        df = pd.read_csv(f)
        df['model'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        df['seed'] = int(os.path.basename(os.path.dirname(f)).rsplit('seed', 1)[1])
        preds.append(df)
    preds = pd.concat(preds, ignore_index=True)
    preds['correct'] = (preds['y_true'] == preds['y_pred']).astype(int)
    if reference not in set(preds['model']):
        print(f'\nReference model {reference} has no predictions; skipping statistical tests.')
        return

    rng = np.random.default_rng(seed)
    test_rows = []
    ref = preds[preds['model'] == reference]
    for name in sorted(set(preds['model']) - {reference}):
        other = preds[preds['model'] == name]
        merged = ref.merge(other, on=['seed', 'fold', 'path'], suffixes=('_ref', '_oth'))
        if merged.empty:
            continue
        a, b = merged['correct_ref'].to_numpy(), merged['correct_oth'].to_numpy()
        n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p_mcnemar = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        idx = rng.integers(0, len(a), (n_boot, len(a)))
        diffs = a[idx].mean(1) - b[idx].mean(1)
        test_rows.append({'reference': reference, 'compared_to': name, 'n_pairs': len(a),
                          'acc_reference': a.mean(), 'acc_other': b.mean(), 'acc_diff': a.mean() - b.mean(),
                          'diff_ci95_low': np.percentile(diffs, 2.5), 'diff_ci95_high': np.percentile(diffs, 97.5),
                          'ref_only_correct': n01, 'other_only_correct': n10, 'mcnemar_p': p_mcnemar})
    tests = pd.DataFrame(test_rows)
    tests.to_csv(os.path.join(out_dir, 'statistical_tests.csv'), index=False)
    print(f'\n=== Predictions pooled over folds vs {reference} (McNemar exact, paired bootstrap) ===')
    print(tests.round(4).to_string(index=False))


# =============================================================================
# Entry point
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folds-dir', default=DEFAULT_FOLDS_DIR, help='folder that contains fold1..fold5')
    p.add_argument('--out', default='results')
    p.add_argument('--models', default=','.join(MODEL_SPECS), help=f'comma separated subset of {list(MODEL_SPECS)}')
    p.add_argument('--folds', default='all', help='e.g. "1,2" (1-based) or "all"')
    p.add_argument('--seeds', default='42')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs-head', type=int, default=30)
    p.add_argument('--epochs-finetune', type=int, default=30)
    p.add_argument('--lr-head', type=float, default=1e-3)
    p.add_argument('--lr-finetune', type=float, default=1e-5)
    p.add_argument('--finetune-fraction', type=float, default=0.2, help='top fraction of backbone layers to unfreeze')
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--hybrid-backbone', default='densenet201', choices=list(BACKBONES))
    p.add_argument('--embed-dim', type=int, default=256)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--mlp-dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--head-dropout', type=float, default=0.3)
    p.add_argument('--near-threshold', type=int, default=10, help='aHash Hamming distance (of 256) for near duplicates')
    p.add_argument('--keep-near-dups', action='store_true',
                   help='keep val/test images that near-duplicate a training image (removed by default)')
    p.add_argument('--aug-dir', default=None,
                   help='folder with fold*_AUG/Train (MSLD v2.0 FOLDS_AUG); adds them to each fold\'s training split')
    p.add_argument('--external-dir', default=None, help='optional external test set with <class>/ sub-folders')
    p.add_argument('--save-weights', action='store_true')
    p.add_argument('--check-only', action='store_true', help='only run the data integrity checks')
    p.add_argument('--summarize', action='store_true', help='only aggregate existing results')
    p.add_argument('--no-pretrained', action='store_true', help='random init (smoke tests only)')
    # parse_known_args: ignore the '-f kernel.json' argument Jupyter/Kaggle passes in notebook cells
    args, _ = p.parse_known_args(argv)
    unknown = set(args.models.split(',')) - set(MODEL_SPECS)
    if unknown:
        p.error(f'unknown model(s): {unknown}')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.summarize:
        summarize(args.out)
        return

    log_file = open(os.path.join(args.out, 'log.txt'), 'a')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    write_run_info(args, log)
    cache = ImageCache(args.img_size)
    folds = discover_folds(args.folds_dir)
    log(f'Found {len(folds)} folds: {[f["name"] for f in folds]}')
    check_folds_partition(folds, cache, log)

    fold_ids = range(len(folds)) if args.folds == 'all' else [int(i) - 1 for i in args.folds.split(',')]
    prepared = {}
    reports = []
    for i in fold_ids:
        prepared[i], rep = prepare_fold(folds[i], cache, args.near_threshold, log,
                                        drop_near_dups=not args.keep_near_dups)
        reports.append(rep)
    pd.DataFrame(reports).to_csv(os.path.join(args.out, 'data_integrity.csv'), index=False)

    aug = discover_aug_train(args.aug_dir) if args.aug_dir else None
    if aug is not None:
        missing = [folds[i]['name'] for i in fold_ids if fold_number(folds[i]['name']) not in aug]
        if missing:
            raise FileNotFoundError(f'No augmented training folder for {missing} under {args.aug_dir}')
    if args.check_only:
        if aug is not None:  # loads every augmented fold once, so the checks can be read before training
            aug_reports = [add_aug_train(prepared[i], aug[fold_number(folds[i]['name'])], args.img_size,
                                         args.near_threshold, log, folds[i]['name'])[1] for i in fold_ids]
            pd.DataFrame(aug_reports).to_csv(os.path.join(args.out, 'aug_integrity.csv'), index=False)
        return

    external = None
    if args.external_dir:
        msld_ahash = average_hash(np.stack(list(cache.images.values())))
        external = load_external(args.external_dir, cache, msld_ahash, args.near_threshold, log)

    seeds = [int(s) for s in args.seeds.split(',')]
    models = args.models.split(',')
    for i in fold_ids:
        fold_name = folds[i]['name']
        if all(os.path.exists(os.path.join(run_dir_of(args, n, fold_name, s), 'metrics.json'))
               for s in seeds for n in models):
            log(f'[skip] all runs of {fold_name} already done')
            continue
        data = prepared[i]
        if aug is not None:  # one fold's augmented images at a time (~1 GB each at 224 px)
            data, _ = add_aug_train(data, aug[fold_number(fold_name)], args.img_size, args.near_threshold, log,
                                    fold_name)
        for seed in seeds:
            for name in models:
                run_one(name, i, fold_name, data, seed, args, external, log)
        del data

    summarize(args.out)
    log_file.close()


def find_folds_dir(root='/kaggle/input'):
    """Locate the folder holding fold1..fold5 of the ORIGINAL images (augmented copies are skipped)."""
    candidates = []
    for dirpath, dirnames, _ in os.walk(root):
        if 'aug' in dirpath.lower():
            continue
        if any(FOLD_RE.match(d) for d in dirnames):
            candidates.append(dirpath)
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected exactly one fold folder under {root}, found: {candidates}. '
                                f'Set FOLDS_DIR manually.')
    return candidates[0]


def find_aug_dir(root='/kaggle/input'):
    """Locate the folder holding fold1_AUG..fold5_AUG (MSLD v2.0 'Augmented Images/.../FOLDS_AUG')."""
    candidates = [dirpath for dirpath, dirnames, _ in os.walk(root) if any(AUG_FOLD_RE.match(d) for d in dirnames)]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected exactly one fold*_AUG folder under {root}, found: {candidates}. '
                                f'Set AUG_DIR manually.')
    return candidates[0]


# =============================================================================
# Notebook settings: edit these and re-run the whole cell
# =============================================================================
MODE = 'train'            # 'check' -> data integrity only | 'train' -> run experiments | 'summarize'
AUG_TRAIN = True          # True -> add the fold-matched MSLD v2.0 augmented training images (val/test stay original)
FOLDS_DIR = None          # None = search /kaggle/input automatically
AUG_DIR = None            # None = search /kaggle/input automatically (only used when AUG_TRAIN = True)
OUT_DIR = '/kaggle/working/results_aug' if AUG_TRAIN else '/kaggle/working/results'
RESUME_FROM = None        # e.g. '/kaggle/input/<previous-notebook-output>/results' to continue an earlier version
EXTRA_ARGS = ['--models', 'densenet201,hybrid_full', '--seeds', '42',
              '--epochs-head', '10', '--epochs-finetune', '10', '--patience', '3']  # e.g. ['--models', 'densenet201,hybrid_full', '--img-size', '128', '--seeds', '42,7,123']


def notebook_argv():
    if RESUME_FROM:
        import shutil
        shutil.copytree(RESUME_FROM, OUT_DIR, dirs_exist_ok=True)
        print(f'Copied previous results from {RESUME_FROM} to {OUT_DIR}')
    argv = ['--out', OUT_DIR]
    if MODE != 'summarize':
        argv += ['--folds-dir', FOLDS_DIR or find_folds_dir()]
        if AUG_TRAIN:
            argv += ['--aug-dir', AUG_DIR or find_aug_dir()]
    argv += {'check': ['--check-only'], 'train': [], 'summarize': ['--summarize']}[MODE]
    return argv + list(EXTRA_ARGS)


if __name__ == '__main__':
    import sys
    # Command line (python run_experiments.py --...) uses its own arguments; a notebook cell uses the settings above
    main() if any(a.startswith('--') for a in sys.argv[1:]) else main(notebook_argv())
