# Speech Emotion Recognition in White Noise: Impact Analysis

A comprehensive research implementation studying the robustness of speech emotion recognition systems to white noise degradation. This project provides a complete pipeline for training, evaluating, and analyzing XGBoost-based emotion recognition models under various noise conditions.

---

## Overview

This research investigates how white noise affects the performance of speech emotion recognition systems across different signal-to-noise ratios (SNRs). The study uses the RAVDESS dataset and implements a comprehensive evaluation framework with statistical confidence intervals and degradation analysis.

### Key Features

- **Comprehensive Feature Extraction**: F0, energy, MFCCs, spectral features with temporal derivatives
- **Noise Robustness Analysis**: Systematic evaluation across 7 SNR levels (20dB to -10dB)
- **Statistical Rigor**: Bootstrap confidence intervals and significance testing
- **Visualization Suite**: Publication-ready plots and audio degradation sequences
- **Reproducible Pipeline**: Complete experimental framework with configuration management

---

## Results Highlights

- **Clean Performance**: 75.0% accuracy on clean test data
- **Noise Degradation**: Performance drops to 25.0% at -10dB SNR
- **Feature Importance**: Energy features most robust to noise
- **Emotion-Specific**: Happy and angry emotions show better noise robustness

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/speech-emotion-recognition-white-noise.git
cd speech-emotion-recognition-white-noise

# Install dependencies
pip install -e .

# Create directory structure
make setup-dirs
```

### Running the Experiment

```bash
# Complete experiment with hyperparameter optimization
make run-experiment

# Quick run without optimization
make run-quick

# Evaluation only (requires trained model)
make evaluate-only
```

### Custom Configuration

```bash
python scripts/run_experiment.py --config config/custom_config.yaml --download --visualize
```

---

## Project Structure

```
speech-emotion-recognition-white-noise/
├── src/                          # Source code
│   ├── data/                     # Dataset handling
│   ├── features/                 # Feature extraction
│   ├── models/                   # Model implementations
│   ├── evaluation/               # Metrics and visualization
│   └── utils/                    # Utility functions
├── scripts/                      # Execution scripts
├── config/                       # Configuration files
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
└── data/                         # Data directory
    ├── raw/                      # Raw datasets
    ├── processed/                # Processed features
    ├── models/                   # Trained models
    └── results/                  # Experimental results
```

---

## Methodology

### Dataset
- **RAVDESS**: 768 emotional speech files (4 emotions: angry, calm, happy, sad)
- **Speaker-Independent**: Non-overlapping speakers in train/validation/test splits
- **Balanced**: Equal samples per emotion class

### Feature Extraction
- **Audio Features**: 92-dimensional feature vectors
- **F0 Features**: Fundamental frequency statistics
- **Energy Features**: RMS energy characteristics  
- **MFCCs**: 13 coefficients with delta and delta-delta
- **Spectral Features**: Zero-crossing rate and spectral centroid

### Model Architecture
- **XGBoost Classifier**: Gradient boosting with hyperparameter optimization
- **Bayesian Optimization**: 50 iterations with 3-fold cross-validation
- **GPU Acceleration**: CUDA-enabled training when available

### Evaluation Protocol
- **Bootstrap Confidence Intervals**: 1000 iterations, 95% confidence level
- **Multiple Metrics**: Accuracy, F1-score, precision, recall
- **Statistical Testing**: Paired t-tests for significance analysis

---

## Key Findings

### Overall Performance Degradation
- **Clean Baseline**: 75.0% ± 7.8% accuracy
- **High SNR (20dB)**: 47.7% accuracy (37% relative drop)
- **Low SNR (-10dB)**: 25.0% accuracy (67% relative drop)

### Per-Emotion Robustness
- **Most Robust**: Angry (94% → 38% recall)
- **Least Robust**: Sad (56% → 19% recall)
- **Calm Emotion**: Highest false positive rate under noise

### Feature Importance
1. **Energy Features**: Most discriminative and robust
2. **Delta-Delta MFCCs**: Strong emotional characterization
3. **Base MFCCs**: Fundamental but noise-sensitive
4. **F0 Features**: Variable importance across emotions

---

## Development

### Testing

```bash
# Run all tests
make test

# Run specific test module
python -m pytest tests/test_features.py -v
```


### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make type-check
```

--- 

## Documentation

- **API Documentation**: Available at `docs/api/`
- **User Guide**: See `docs/user_guide.md`
- **Research Paper**: `docs/paper.pdf`
- **Experimental Protocols**: `docs/protocols.md`

---

## Citation

If you use this work in your research, please cite:
**Coming soon**

---

## Acknowledgments

- **RAVDESS Dataset**: Livingstone & Russo (2018)
- **XGBoost**: Chen & Guestrin (2016)
- **Librosa**: McFee et al. (2015)
- **Scikit-learn**: Pedregosa et al. (2011)

---


## Contributing

This is an MSc research project. No contributions accepted at this time.

---

## License

This project is licensed under the MIT License.


# ==============================================================================
# LICENSE (MIT License)
# ==============================================================================

MIT License

Copyright (c) 2024 Speech Emotion Recognition Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### Author

**Tatiana Limonova**  
MSc Language Sciences (Technology of Language and Speech) – UCL  
[GitHub Profile](https://github.com/kanincityy) • [LinkedIn](https://linkedin.com/in/tatianalimonova)  
