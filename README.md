# Well Log Facies Prediction with Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive machine learning framework for **depositional facies classification** and **petrophysical property prediction** from well log data in the oil & gas industry. This project implements multiple ML approaches with robust **uncertainty quantification** to support reservoir characterization and geological interpretation.

## Project Overview

This project addresses the challenge of predicting geological facies (depositional environments) and petrophysical properties (porosity, water saturation) from well log measurements. Key innovations include:

- **Well-based data splitting** to prevent data leakage between training and testing
- **Uncertainty quantification** using ensemble methods and Bayesian approaches
- **Feature engineering** with geologically-meaningful derived features
- **Multiple ML approaches** for comparison and ensemble predictions

### Facies Classes (Equinor Interpretation)

| Code | Facies Name |
|------|-------------|
| 0 | Tidal Bar |
| 1 | Upper Shoreface |
| 2 | Offshore |
| 3 | Tidal Channel |
| 4 | Mouthbar |
| 5 | Lower Shoreface |
| 6 | Marsh |
| 7 | Tidal Flat Muddy |
| 8 | Tidal Flat Sandy |

## Features

### Input Well Log Data
- **Base Logs (10)**: GR, RT, RD, NPHI, RHOB, DT, VSH, KLOGH, PHIF, SW
- **Lithology Labels**: One-hot encoded (5 classes)

### Engineered Features (7)
| Feature | Description |
|---------|-------------|
| `GR_Slope` | GR derivative indicating coarsening/fining trends |
| `GR_Serration` | Rolling standard deviation of GR (heterogeneity indicator) |
| `RelPos` | Relative position within GR cycles |
| `OppositionIndex` | Local correlation between GR and porosity |
| `NTG_Slope` | Net-to-gross trend within depth window |
| `AI_Slope` | Acoustic impedance trend |
| `BaseSharpness` | Maximum GR gradient at cycle boundaries |

## Machine Learning Approaches

### 1. Random Forest Classification with HDG Uncertainty
**File:** `rf_classification_uncertainty_ensemble_v4_hdg.py`

Implements the **Halotel–Demyanov–Gardiner (HDG)** approach:
- Systematic train/validation splits using entire wells as folds
- Hyperparameter tuning on validation set
- Ensemble predictions with uncertainty through voting agreement
- Per-depth facies proportion analysis

### 2. Bayesian Neural Network with MC Dropout
**File:** `nn_facies_prediction_bayesian_dropout_well_split.py`

- Monte Carlo Dropout for uncertainty estimation
- Well-based train/validation/test splitting
- Calibrated confidence intervals
- Detailed uncertainty analysis per facies

### 3. Random Forest Regression for Property Prediction
**File:** `rf_regression_phif_sw_f14_imputation_v3_with_confidence.py`

Predicts missing PHIF (porosity) and SW (water saturation) for well F-14:
- Cross-validated model training
- Confidence intervals using tree variance
- Uncertainty-aware predictions

## Project Structure

```
├── Dataset_logs_core_v4_cleaned.csv          # Raw well log data
├── ML_ready_*.csv                            # Preprocessed ML-ready datasets
│
├── prepare_ml_ready_dataset_*.py             # Data preparation scripts
│   ├── prepare_ml_ready_dataset_10_base_plus_engineered_f14_special.py
│   ├── prepare_ml_ready_dataset_gr_vsh_rt_rd_nphi_rhob_dt_phif_sw_klogh_litho.py
│   └── ... (various feature combinations)
│
├── rf_classification_*.py                    # Random Forest classifiers
│   ├── rf_classification_uncertainty_ensemble_v4_hdg.py  # HDG approach
│   ├── rf_classification_uncertainty_ensemble_v3.py
│   └── rf_classification_simple_max_dataset.py
│
├── nn_facies_prediction_bayesian_dropout_well_split.py  # Neural Network
│
├── rf_regression_phif_sw_f14_imputation_v3_with_confidence.py  # Regression
│
├── RF_HDG_Uncertainty_*/                     # Output directories
├── NN_Bayesian_WellSplit_*/                  # Output directories
│
└── OLD/                                      # Previous versions
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Place your raw well log data as `Dataset_logs_core_v4_cleaned.csv`
2. Run the data preparation script:

```bash
python prepare_ml_ready_dataset_10_base_plus_engineered_f14_special.py
```

This will:
- Clean and validate well log measurements
- Apply per-well KNN imputation for missing values
- Calculate engineered features
- Handle special cases (e.g., F-14 missing PHIF/SW)
- Create normalized ML-ready dataset

### Training Models

#### Random Forest HDG Ensemble
```bash
python rf_classification_uncertainty_ensemble_v4_hdg.py
```

Interactive prompts will guide you through:
1. Selecting blind test well(s)
2. Configuring train/validation splits
3. Enabling hyperparameter tuning

#### Bayesian Neural Network
```bash
python nn_facies_prediction_bayesian_dropout_well_split.py
```

Interactive prompts will guide you through:
1. Selecting training, validation, and test wells
2. Model automatically trains with early stopping
3. MC Dropout uncertainty quantification

#### Property Prediction (PHIF/SW)
```bash
python rf_regression_phif_sw_f14_imputation_v3_with_confidence.py
```

## Output and Visualization

### Generated Outputs
- **Confusion matrices** (raw counts and normalized)
- **Feature importance plots** (color-coded by feature type)
- **Per-well accuracy analysis**
- **Facies proportion logs** (HDG depth profiles)
- **Uncertainty profiles** (depth vs. prediction confidence)
- **Calibration plots** (confidence vs. accuracy)
- **Monte Carlo analysis** (sample agreement, variance)
- **CSV exports** with predictions and uncertainty metrics

### Example Visualizations

The framework generates comprehensive visualizations including:
- 3-panel facies proportion logs
- 6-panel comprehensive HDG analysis
- Training history plots
- Well-based uncertainty analysis

## Configuration

### Hyperparameter Grid (RF)
```python
PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 8],
    'min_samples_leaf': [3, 5]
}
```

### Neural Network Architecture
- Input → BatchNorm → Dense(256) → MC Dropout → Dense(128) → MC Dropout → Dense(64) → MC Dropout → Dense(32) → MC Dropout → Output(Softmax)
- Dropout rate: 0.3
- L1/L2 regularization
- Adam optimizer with learning rate scheduling

## References

This implementation is inspired by:

1. **HDG Approach**: Halotel, Demyanov, Gardiner - "Value of information of time-lapse seismic data for prediction improvement of reservoir production" (SPE/Petroleum Geoscience)

2. **Bayesian Uncertainty**: [Uncertainty quantification in facies classification from well logs](https://www.sciencedirect.com/science/article/pii/S0920410521004770)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Well Log Analysis & ML Pipeline Development

## Acknowledgments

- Professor Vasily Demyanov, Heriot-Watt University
- PhD Student Farah Rabie, Heriot-Watt University
- Equinor for facies interpretation methodology
- Volve dataset contributors
- Open-source ML community

