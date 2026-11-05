# Frozen vision–language foundation models for mpox skin lesion classification

Code for the paper *Frozen vision–language foundation models for mpox skin lesion classification: leakage-free
benchmarking, external validation and a skin-tone audit* (T. Muezzinoglu, A. T. Kabakus; submitted to
*Biomedical Signal Processing and Control*).

## What the code does

`final_experiments.py` reproduces every table of the paper and of the Supplementary Material in one run:

1. **Duplicate screening.** It removes exact (MD5) and near (16 × 16 average hash, Hamming ≤ 10) copies of training
   images from the validation and test splits of the official MSLD v2.0 folds. It also flags copies of MSLD images in
   the external datasets.
2. **Feature extraction and probes.** It extracts frozen features from seven encoders (DenseNet201, DINOv2, DINOv3,
   BiomedCLIP, SigLIP, SigLIP2 and MONET). It then trains linear probes, with C and the temperature selected inside the
   training data of each fold.
3. **Benchmark statistics.** It runs the benchmark with the corrected resampled t-test, Holm adjustment, McNemar tests
   and bootstrap confidence intervals.
4. **Reliability analyses.** It computes calibration, selective prediction and split conformal prediction.
5. **Zero-shot concept scores.** These come from the MONET and SigLIP2 text encoders.
6. **Shortcut audit.** It covers image attributes, skin tone measured with the individual typology angle (ITA),
   masked images and attribute-only models.
7. **External validation.** It tests on the independent images of MSID and MCSI.
8. **Skin-tone analysis.** It compares reweighting, grey-scale conversion and tone augmentation, and reports the AUC
   and sensitivity within each skin-tone group.

`figures/make_figures.py` draws Figs. 1–5 and Fig. S1 from the CSV files written by `final_experiments.py`. It saves
each figure as a 600-DPI PNG and a vector PDF.

## Data

| Dataset | Source |
|---|---|
| MSLD v2.0 (official five folds) | Ali et al., *Biomed. Signal Process. Control* 98 (2024) 106742 |
| MSID, version 6 | Mendeley Data, https://doi.org/10.17632/r9bfpnvyxr.6 |
| MCSI, version 2 | Zenodo, https://doi.org/10.5281/zenodo.8360076 |

The datasets are not redistributed here.

## How to run

1. **Create a notebook.** Create a Kaggle notebook with Accelerator = GPU (T4) and Internet = On.
2. **Add the datasets.** Add the three datasets as inputs. If the folder names differ, set `FOLDS_DIR` and
   `EXTERNAL_DIRS` at the top of `final_experiments.py`. `discover_datasets.py` lists the folders that are attached.
3. **Run the experiments.** Paste the whole of `final_experiments.py` into **one** cell and run it. The notebook must
   contain only this cell.
   - A run takes about 2 hours, or about 1–1.5 hours when cached features are reused.
   - The CSV tables are written to `/kaggle/working/final`.
4. **Copy the outputs.** Copy the CSV files to `pilot_outputs/final/`.
5. **Draw the figures.** Run:

   ```bash
   python figures/make_figures.py
   ```

The probes, the bootstrap and the statistical tests are seeded. Features are extracted in half precision on a GPU, so
runs on other hardware may differ in the last digit.

Main dependencies: `torch`, `timm`, `open_clip_torch`, `transformers`, `scikit-learn`, `scipy`, `pandas`, `numpy`,
`opencv-python`, `Pillow`, `matplotlib`.

## Other files

- `check_leakage.py` shows that pooling the five fold directories of MSLD v2.0 yields 3,775 files in which every
  original image occurs five times. It also shows that a random 70/30 split then places copies of almost all test
  images in the training set. This was the starting point of the study.
- `run_experiments.py` and `pilot*.py` are earlier experiments: the fine-tuned CNN and hybrid baselines and the pilot
  studies that led to `final_experiments.py`. They are not needed to reproduce the paper.

## Use of AI-assisted tools

Claude (Anthropic) was used to help write and debug the code. All experiments were run by the authors.

## Licence and citation

Please cite the paper if you use this code. Licence: to be added by the authors.
