# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSc dissertation project measuring how an XGBoost Speech Emotion Recognition (SER) model degrades under white noise at varying Signal-to-Noise Ratios (SNRs). Trained on clean speech from the RAVDESS dataset, evaluated on clean and synthetically noised test data, for four emotions: **angry**, **calm**, **happy**, **sad**.

## Setup

Dependency management is via `uv` — `pyproject.toml` declares deps, `uv.lock` pins exact resolved versions (committed to git for reproducibility). `requirements.txt`/pip are no longer used.

```bash
uv sync
```

Requires a Kaggle API token (`~/.kaggle/kaggle.json`) — `data_preparation.py` downloads the RAVDESS dataset via `kagglehub`. XGBoost's `device` (`config.DEVICE`) defaults to `'cuda'`; override with `XGB_DEVICE=cpu` env var to run on CPU-only machines.

There is no test suite, linter, or build step in this repository. `main.py` at the repo root is an unused stub left over from `uv init` — it is not part of the pipeline.

## Running the pipeline

Scripts must be run from `src/`, in this order — each stage reads pickled/joblib artifacts produced by the previous one from `data/`:

```bash
cd src
uv run python data_preparation.py      # Download RAVDESS, parse filenames, speaker-independent train/val/test split
uv run python feature_extraction.py    # Extract acoustic features from clean audio
uv run python noisy_data_generator.py  # Generate noisy test sets at each SNR level in config.SNR_LEVELS
uv run python model_trainer.py         # Bayesian hyperparameter search (BayesSearchCV) + train XGBoost; XGB_DEVICE=cuda|cpu (default cuda)
uv run python model_evaluator.py       # Evaluate on clean + all noisy test sets, with bootstrap CIs
uv run python results_visualiser.py    # Generate and save all plots
```

`noisy_data_generator.py` imports `extract_global_features` directly from the `feature_extraction` module (`from feature_extraction import extract_global_features`).

There's no CLI/entrypoint layer — each script's `main()` is invoked via `if __name__ == '__main__'`, and every script hardcodes its input/output paths from `config.py`.

## Reproducibility

`config.RANDOM_SEED` (42) is the single seed used across the pipeline — `add_noise` (in `utils.py`) takes a seeded `np.random.default_rng(RANDOM_SEED)` passed in by `noisy_data_generator.py` and `results_visualiser.py` rather than drawing from unseeded global `np.random` state, and the bootstrap resampling in `model_evaluator.get_bootstrap_metrics` uses `random_state=i` per iteration. This means noisy-test-set generation and CI numbers are now stable across reruns — don't reintroduce unseeded `np.random.randn`/`resample` calls, since that would make published numbers non-reproducible from run to run.

One thing is *not* pinned and could still cause drift across reruns/machines: the Kaggle-hosted RAVDESS dataset itself (`kagglehub.dataset_download` has no version pin). `device` is now selectable via `XGB_DEVICE` env var (see Setup) instead of hardcoded.

## Architecture

Everything is configuration-driven through `src/config.py`, which centralizes all paths, dataset filtering (`EMOTIONS_TO_KEEP`), feature-extraction params (`TARGET_SR`, `N_FFT`, `HOP_LENGTH`, `N_MFCC`), SNR levels for noise injection, train/val/test split ratios, Bayesian search settings, fixed XGBoost params, and bootstrap CI settings. Change experiment parameters here rather than in the individual scripts.

Pipeline stages and their artifact hand-offs (all under `data/`, `models/`, `results/` — none of which are checked into git):

1. **`data_preparation.py`** — downloads RAVDESS via kagglehub, parses filenames with `utils.parse_ravdess_filename` (RAVDESS filenames encode modality/vocal-channel/emotion/actor as `--`-separated codes), does a **speaker-independent** split using `GroupShuffleSplit` grouped by `speaker_id` (critical: prevents the same actor appearing in both train and test). Outputs `metadata.pkl`, `train_df.pkl`, `val_df.pkl`, `test_df.pkl`.
2. **`feature_extraction.py`** — defines `extract_global_features()` (F0 via `librosa.pyin`, RMS energy, 13 MFCCs + delta + delta², ZCR, spectral centroid — all reduced to mean/std/min/max style scalars per clip, i.e. "global" utterance-level features, not framewise sequences). Fits `StandardScaler`/`LabelEncoder` **on training data only**, then applies them to val/test. NaN imputation uses the training feature means, saved separately for reuse in the noisy pipeline.
3. **`noisy_data_generator.py`** — reuses `extract_global_features` from step 2 and `utils.add_noise` (adds white noise scaled to hit a target SNR in dB) to build one noisy feature set per SNR level in `config.SNR_LEVELS`, transformed using the **scaler/encoder/feature-means fitted in step 2** (never refit on noisy data).
4. **`model_trainer.py`** — pools train+val (cross-validation replaces a fixed validation set), uses `StratifiedGroupKFold` grouped by `speaker_id` with gender-based stratification (derived from `speaker_id % 2`, per RAVDESS convention), and runs `BayesSearchCV` over XGBoost hyperparameters. Saves the single best estimator.
5. **`model_evaluator.py`** — scores the saved model on clean and every noisy test set, computes bootstrap confidence intervals per emotion (`get_bootstrap_metrics`), and separately identifies a "representative speaker" (`find_representative_speaker`) whose per-SNR accuracy trend best matches the overall average trend (lowest MSE) — used later for qualitative plots.
6. **`results_visualiser.py`** — consumes `model_evaluator.py`'s saved results to plot per-emotion F1 degradation with CIs across SNR levels, normalised confusion matrices at selected SNRs, and waveform/mel-spectrogram/MFCC visualisations for the representative speaker using `utils.plot_audio_ser`.

Key invariant across the whole pipeline: anything that transforms features (scaler, label encoder, imputation means) is fit exactly once on the clean training split and reused everywhere else — never refit on validation, test, or noisy data.
