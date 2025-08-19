# Speech Emotion Recognition in White Noise

This project implements a complete machine learning pipeline to investigate the impact of white noise on a Speech Emotion Recognition (SER) system. An XGBoost model is trained on the RAVDESS dataset to classify emotions, and its performance is evaluated on both clean and noisy audio to measure resilience.

The project is built with a modular, scalable, and reproducible architecture, making it suitable for further development and experimentation.

## Features

  - **Modular Codebase**: Housed in an installable `src` package, separating concerns like data processing, feature extraction, and modeling.
  - **Configuration Driven**: All parameters, paths, and settings are managed centrally via `config/config.yaml`, eliminating hardcoded values.
  - **End-to-End Pipeline**: Includes automated scripts for data download, preprocessing, training, evaluation, and visualization.
  - **Reproducibility**: Uses a fixed random seed for consistent results in data splitting and model training.
  - **Robust Logging**: Implements structured logging to a file (`project.log`) and the console, replacing simple print statements for better monitoring and debugging.

-----

## Getting Started

### Prerequisites

  - Python 3.8+
  - An NVIDIA GPU with CUDA installed (for `device="cuda"` in XGBoost). The configuration can be changed to use a CPU if needed.
  - Kaggle API credentials set up for dataset download.

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <your-repo-url>
    cd speech-emotion-recognition-white-noise
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the Project in Editable Mode**
    This command installs the dependencies from `requirements.txt` and makes the `src` folder available as a package throughout your environment.

    ```bash
    pip install -e .
    ```

-----

## ⚡️ How to Run

You can run the entire experiment—from data download to result generation—with a single command. The script will automatically create the necessary directories (`data/raw`, `data/processed`, etc.).

```bash
python scripts/run_experiment.py
```

This master script executes the training and evaluation pipelines in sequence.

### Running Individual Stages

For more granular control, you can run individual stages of the pipeline:

  - **1. Train the model:**
    This script will download the data, preprocess it, extract features, and run the hyperparameter tuning to train and save the best model.

    ```bash
    python scripts/train_model.py
    ```

  - **2. Evaluate the trained model:**
    This script loads the saved model and evaluates it against the clean and noisy test sets, generating metrics and visualizations.

    ```bash
    python scripts/evaluate_model.py
    ```

All outputs, including processed data, the trained model, and result plots, will be saved in the `data/` directory as specified in `config/config.yaml`.

-----

## Project Structure

```
speech-emotion-recognition-white-noise/
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── setup.py                  # Makes the `src` directory an installable package
├── .gitignore                # Specifies files for Git to ignore
├── config/
│   └── config.yaml           # Central configuration for all parameters and paths
├── src/                      # Source code for the installable package
│   ├── data/                 # Modules for data handling and preprocessing
│   ├── features/             # Feature extraction modules
│   ├── models/               # Model definition and training logic
│   ├── evaluation/           # Modules for model evaluation and visualization
│   └── utils/                # Helper functions (e.g., logging, config loading)
├── scripts/                  # High-level executable scripts
│   ├── train_model.py        # Runs the training pipeline
│   ├── evaluate_model.py     # Runs the evaluation pipeline
│   └── run_experiment.py     # Runs the full end-to-end experiment
└── data/                     # Directory for all data (created automatically)
    ├── raw/                  # Raw downloaded data
    ├── processed/            # Processed data and saved artifacts (scaler, encoder)
    ├── models/               # Saved model file
    └── results/              # Final metrics and plots
```

## Contributing

This is an MSc research project. No contributions accepted at this time.

---

## License

This project is licensed under the MIT License.

---

### Author

**Tatiana Limonova**  
MSc Language Sciences (Technology of Language and Speech) – UCL  
[GitHub Profile](https://github.com/kanincityy) • [LinkedIn](https://linkedin.com/in/tatianalimonova)  
