# Enhancing Speech Emotion Recognition: The Impact of White Noise and Signal-to-Noise Ratios

This repository contains the source code and methodology for the MSc dissertation project investigating the performance of an XGBoost-based Speech Emotion Recognition (SER) model under varying levels of white noise.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Project Structure](#project-structure)
3.  [Setup and Installation](#setup-and-installation)
4.  [Dataset](#dataset)
5.  [Execution Workflow](#execution-workflow)
6.  [Citation](#citation)

## Project Overview
This project aims to quantify the degradation in performance of a SER system when subjected to white noise at different Signal-to-Noise Ratios (SNRs). The model is trained on clean speech from the RAVDESS dataset and then evaluated on both clean and noisy test data. The emotions under consideration are **angry**, **calm**, **happy**, and **sad**.

Acoustic features including Fundamental Frequency (F0), Root Mean Square (RMS) energy, Mel-Frequency Cepstral Coefficients (MFCCs) and their derivatives, Zero-Crossing Rate (ZCR), and Spectral Centroid are used to train an XGBoost classifier.

## Project Structure
The project is organised into a modular structure to ensure clarity and reproducibility.

```
SER_in_White_Noise_XGBoost/
├── data/                  # Stores datasets, features, and pre-processing objects.
├── models/                # Stores the final trained model.
├── results/               # Stores evaluation metrics (CSVs) and plots (PNGs).
├── src/                   # Contains all Python source code.
├── requirements.txt       # Project dependencies.
└── README.md              # This file.
```

## Setup and Installation

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd SER_in_White_Noise_XGBoost
```

**2. Set up a Python Environment**
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Kaggle API Credentials**
This project uses the `kagglehub` library to download the dataset automatically. You need to have your Kaggle API token (`kaggle.json`) set up. Please follow the instructions [here](https://www.kaggle.com/docs/api) to get your token and place it in the `~/.kaggle/` directory.

## Dataset
The project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. The `src/data_preparation.py` script will automatically download and extract the dataset using the Kaggle API. Only audio-only speech files for the four target emotions are used.

## Execution Workflow
To run the full pipeline from data download to result visualisation, execute the scripts from the `src/` directory in the following order. It is recommended to run them from the root directory of the project.

```bash
# Make sure your virtual environment is activated
# Run from the root directory: SER_in_White_Noise_XGBoost/

# Step 1: Download, parse, and split the dataset
python src/data_preparation.py

# Step 2: Extract acoustic features from the clean audio splits
python src/feature_extraction.py

# Step 3: Generate noisy test sets for each SNR level
python src/noisy_data_generator.py

# Step 4: Train the XGBoost model with Bayesian hyperparameter optimisation
# This step is computationally intensive and requires a CUDA-enabled GPU.
python src/model_trainer.py

# Step 5: Evaluate the trained model on all clean and noisy test sets
python src/model_evaluator.py

# Step 6: Generate and save all plots and visualisations
python src/results_visualiser.py
```

After running all scripts, the `results/` folder will contain all generated plots and metric files.

## Citation
If you use the RAVDESS dataset, please cite the original paper:

> Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391