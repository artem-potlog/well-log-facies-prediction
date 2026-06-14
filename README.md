# Well Log Facies Prediction (Volve Hugin)

Machine-learning workflow for predicting depositional facies and reservoir-quality tiers from wireline logs in the Equinor **Volve** field (**Middle Jurassic Hugin Formation**). The notebook is self-contained: open it, run top to bottom, and every transformation, model, and metric is defined inside.

## Repository contents

| File | Description |
|---|---|
| `main_workflow_v3.ipynb` | End-to-end facies prediction workflow (Steps 0–11, appendix) |
| `Dataset_logs_core_v4_cleaned.xlsx` | Well log data (~0.15 m sampling, 19 wells) |
| `requirements.txt` | Python dependencies |

## Problem

Core-derived facies labels are authoritative but sparse. Wireline logs are continuous yet ambiguous at a single depth — GR, porosity, and shale indicators overlap between tidal bars, mouth bars, and heterolithic packages. This workflow infers **5-class facies** and **economic tiers** (pay vs seal) along the full borehole, including in blind wells not used for training.

Evaluation uses **leave-one-well-out (LOWO)** cross-validation on five log-rich wells, simulating deployment on an unseen borehole.

## Headline results (pooled blind wells)

| Metric | Value |
|---|---|
| Facies accuracy | ~0.58 |
| Tier accuracy | ~0.65 |
| Macro-F1 | ~0.50 |
| Pay vs seal accuracy | ~0.96 |
| Reservoir recall (pay captured) | ~0.99 |
| Within-1-tier accuracy | ~0.98 |

## Quick start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/artem-potlog/well-log-facies-prediction.git
cd well-log-facies-prediction
pip install -r requirements.txt
jupyter notebook main_workflow_v3.ipynb
```

Run all cells in order. The notebook searches upward from the working directory to find `Dataset_logs_core_v4_cleaned.xlsx`.

## Workflow overview

0. **Why this workflow** — problem, motivation, blind-well protocol
1. **Setup** — imports and path resolution
2. **Load data** — 5 log-rich wells; impute F-14 PHIF/SW
3. **Merge facies** — 9 Equinor classes → 5 decision-useful classes (+ tiers)
4. **Feature engineering** — rank-normalized logs, multi-scale vertical context
5. **Model** — Random Forest + vertical probability smoothing
6. **Blind-well evaluation** — LOWO on five wells
7. **Results** — per-well and pooled metrics
8. **Diagnostics** — confusion matrices, feature importance, learning curve
9. **Ensemble** — multi-view soft voting + uncertainty signals
10. **Scenarios** — P1 / P2 / P3 alternative conditioning realizations
11. **Discrimination experts** — pair-aware re-weighting for confused facies

The appendix documents side experiments (baseline features, imputation alternatives, smoothing sweep, seal knobs).

## References

Facies scheme grounded in **Kieft et al. (2011)** — shallow-marine, tide-influenced Hugin deposition. Full citations are in the notebook References section.
