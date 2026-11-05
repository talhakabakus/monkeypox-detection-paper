"""
Step 1 of the extended study: list every dataset attached to this Kaggle notebook and its folder structure,
so that the audit and the experiments can be pointed at the right folders. CPU only, takes a few seconds.

Paste the whole file into ONE notebook cell and run it. Send the complete printed output back.
"""
import os
from collections import Counter

ROOT = '/kaggle/input'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
MAX_DEPTH = 9       # /kaggle/input/datasets/<owner>/<name>/... plus e.g. FOLDS_AUG/fold1_AUG/Train/<class>
MAX_CHILDREN = 12   # folders listed per level before the rest are summarised


def image_counts(root):
    """number of images directly in, and anywhere below, every folder"""
    direct, total = Counter(), Counter()
    for dirpath, _, files in os.walk(root):
        n = sum(f.lower().endswith(IMAGE_EXTENSIONS) for f in files)
        direct[dirpath] = n
        path = dirpath
        while True:
            total[path] += n
            if path == root:
                break
            path = os.path.dirname(path)
    return direct, total


def show(path, depth, direct, total):
    children = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    children = [c for c in children if total[os.path.join(path, c)]]
    for i, child in enumerate(children):
        if i == MAX_CHILDREN:
            rest = children[MAX_CHILDREN:]
            print('  ' * depth + f'... and {len(rest)} more folders '
                  f'({sum(total[os.path.join(path, c)] for c in rest)} images)')
            break
        full = os.path.join(path, child)
        extra = f', {direct[full]} directly' if direct[full] and direct[full] != total[full] else ''
        print('  ' * depth + f'{child}/  ({total[full]} images{extra})')
        if depth < MAX_DEPTH:
            show(full, depth + 1, direct, total)


def main():
    if not os.path.isdir(ROOT):
        print(f'{ROOT} not found: run this cell on Kaggle')
        return
    datasets = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    print(f'Attached inputs under {ROOT}: {datasets}\n')
    direct, total = image_counts(ROOT)
    for d in datasets:
        full = os.path.join(ROOT, d)
        print(f'=== {d}/  ({total[full]} images)')
        show(full, 1, direct, total)
        other = Counter(os.path.splitext(f)[1].lower() for _, _, fs in os.walk(full) for f in fs
                        if not f.lower().endswith(IMAGE_EXTENSIONS))
        if other:
            print(f'  other files: {dict(other.most_common(8))}')
        print()


main()
