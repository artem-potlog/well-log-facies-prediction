# V7: Intelligent Feature Selection with Optuna

## Overview

**Problem**: You have 21 features → 2^21 = 2,097,152 possible combinations. Testing all is impossible.

**Solution**: Two-step approach combining V7 (discovery) + V6 (training):

### Step 1: V7 - Feature Discovery
Use Optuna's Bayesian optimization to discover best feature combinations:
- Each feature = binary decision (include/exclude)
- Objective = CV Score - Feature Penalty
- TPE learns which features lead to good performance
- Finds 30-40 best combinations quickly (1-2 hours)

### Step 2: V7 Final - Full Training (Optional)
Train discovered combinations with V6's full methodology:
- Load top N combinations from V7 results
- Full GridSearchCV hyperparameter tuning
- Multiple optimization metrics
- Comprehensive ensemble predictions and visualizations

**Key Innovation**: V7 discovers features (fast), V7 Final trains them properly (thorough)

---

## Quick Start

### Two-Step Workflow

#### Step 1: V7 - Feature Discovery (Required)

```bash
# Install dependencies
pip install optuna>=3.0.0

# Run V7 for feature discovery
python v7_multi_scenario_facies_cv.py
```

**Follow Prompts**
1. **Mode**: Choose number of trials
   - Quick Test: 50 trials (~30 min)
   - Standard: 100 trials (~1 hour) ⭐ RECOMMENDED  
   - Thorough: 200 trials (~2 hours)
   - Exhaustive: 500 trials (~5 hours)

2. **Metric**: Choose optimization metric
   - custom_sand: Reservoir-focused ⭐ RECOMMENDED
   - f1_weighted: Balanced across all facies

3. **Wells**: Select training and test wells

4. **Training**: Optionally train top 30 discovered models

**Expected Results**:
- Time: 1-2 hours for Standard (100 trials)
- Output: 30-40 best feature combinations
- Best CV: 0.48-0.50 (custom_sand metric)
- Optimal features: 6-12 features typically

---

#### Step 2: V7 Final - Full Training (Optional but Recommended)

```bash
# Train V7 discoveries with V6's full methodology
python v7_multi_scenario_facies_cv_final.py
```

**Follow Prompts**:
1. **V7 directory**: Path to V7 output (e.g., `v7_output`)
2. **Top N combinations**: How many to train (default: 20)
3. **Grid type**: Hyperparameter grid size (standard recommended)
4. **Wells**: Select training and test wells

**Expected Results**:
- Time: 3-5 hours for 20 combinations (standard grid)
- Output: Fully trained models with GridSearchCV
- Same ensemble methodology as V6
- Comprehensive visualizations

**Why use V7 Final?**
- V7 uses fixed RF params (fast discovery)
- V7 Final uses full GridSearchCV (optimal performance)
- Best of both worlds: intelligent discovery + thorough training

---

## How It Works

### 1. Feature Encoding
Each trial samples a subset of 21 features:
```python
Trial #42:
  GR: True
  VSH: True
  RT: False
  ... (21 total decisions)
```

### 2. Objective Function
```python
Objective = CV_Score - Feature_Penalty

where:
  CV_Score = GroupKFold cross-validation score
  Feature_Penalty = 0.01 × (n_features / 21)
```

This encourages:
- ✅ High performance (maximize CV score)
- ✅ Simplicity (minimize features → Occam's razor)

### 3. Intelligent Sampling (TPE)
Optuna's Tree-structured Parzen Estimator:
- Learns from previous trials
- Builds probabilistic models of good/bad feature sets
- Samples from promising regions
- Much smarter than random or grid search

### 4. Extract Best Combinations
- Rank all trials by objective value
- Extract top 30-40 unique feature subsets
- Train final models with full pipeline
- Generate ensemble predictions

---

## Understanding Output

### Files Generated

```
v7_output/
├── results/
│   ├── best_feature_combinations.csv    # Top 30-40 ranked
│   └── analysis_summary.txt             # Detailed report
├── models/
│   ├── rank_01_trial_042/
│   │   ├── model.pkl                    # Trained pipeline
│   │   ├── metadata.json                # Features & params
│   │   └── feature_importance.csv       # RF importances
│   └── ... (top 30 models)
├── optuna_history/
│   └── study.pkl                        # Full Optuna study
├── predictions/
│   └── {well}/ensemble_predictions.csv  # Facies + uncertainty
└── visualizations/
    ├── v7_optimization_history.png      # Trial progress
    ├── v7_feature_importance.png        # Feature analysis
    └── v7_model_comparison.png          # Top models
```

### Key Results File

`best_feature_combinations.csv`:
```csv
rank,trial_number,n_features,cv_mean,cv_std,objective_value
1,42,8,0.480,0.045,0.476
2,73,7,0.475,0.038,0.472
3,91,10,0.478,0.052,0.469
```

**Columns**:
- `rank`: Overall ranking (1 = best)
- `trial_number`: Optuna trial ID
- `n_features`: Feature count
- `cv_mean`: CV score
- `cv_std`: CV standard deviation
- `objective_value`: Score after penalty

---

## Interpreting Results

### Optimization Progress

Check `v7_optimization_history.png`:
- **Panel 1 (top-left)**: Objective value over trials
  - Should show upward trend
  - Plateau indicates convergence
- **Panel 2 (top-right)**: Feature count distribution
  - Optimal typically 6-12 features
- **Panel 3 (bottom-left)**: CV score vs feature count
  - Reveals performance-complexity trade-off
- **Panel 4 (bottom-right)**: Most frequent features
  - Shows which features are most important

### Feature Importance

Check console output and `analysis_summary.txt`:

**Essential features** (≥80% selection):
- Appear in most top models
- Critical for performance

**Important features** (50-79%):
- Frequently selected
- Strong contributors

**Optional features** (20-49%):
- Sometimes helpful
- Context-dependent

**Rare features** (<20%):
- Seldom selected
- Likely redundant

### Model Quality

Check `v7_model_comparison.png`:
- **Top-left**: CV scores should decay smoothly
- **Top-right**: Feature count variation shows diversity
- **Bottom-left**: Score distribution
- **Bottom-right**: Summary statistics

---

## Workflow Comparison

### Three Approaches Available

| Approach | What It Does | Time | Best For |
|----------|-------------|------|----------|
| **V6** | Trains 19 manual feature combinations | 5-8 hours | Hypothesis testing |
| **V7** | Discovers optimal features (fast) | 1-2 hours | Quick discovery |
| **V7 Final** | V7 discovery + V6 training | 4-6 hours total | Production (best performance) |

### Recommended Workflow

```
1. Run V7 (1-2 hours)
   ↓
2. Analyze discovered features
   ↓
3. Run V7 Final on top 20-30 (3-5 hours)
   ↓
4. Deploy best ensemble models
```

**Why this workflow?**
- V7: Fast feature discovery with intelligent sampling
- V7 Final: Thorough training of discovered features
- Result: Optimal features + optimal hyperparameters

---

## V7 vs V6 Comparison

| Aspect | V6 (Manual) | V7 (Discovery) | V7 Final (Best) |
|--------|-------------|----------------|-----------------|
| **Approach** | 19 hand-crafted | Optuna discovery | V7 + V6 training |
| **Feature combinations** | 19 manual | 30-40 discovered | Top 20-30 from V7 |
| **HP tuning** | Full GridSearchCV | Fixed params | Full GridSearchCV |
| **Search space** | 0.001% explored | 0.005-0.024% | Best of both |
| **Time** | 5-8 hours | 1-2 hours | 4-6 hours total |
| **Human bias** | High | Minimal | Minimal |
| **Discovery** | Limited | Excellent | Excellent |
| **Training quality** | Best per combo | Fast | Best per combo |
| **Best use** | Hypothesis testing | Quick discovery | Production |

### When to Use Each

#### Use V7 (Discovery Only)
- ✅ Quick feature discovery (1-2 hours)
- ✅ Want to see which features matter
- ✅ Prototyping/exploration phase
- ✅ Limited time budget

#### Use V7 Final (Discovery + Training)
- ✅ Production deployment (best performance)
- ✅ Want both optimal features AND hyperparameters
- ✅ Have 4-6 hours total
- ✅ Need comprehensive ensemble predictions
- ✅ **RECOMMENDED for final models**

#### Use V6 (Manual)
- ✅ Testing specific hypotheses
- ✅ Need interpretable feature groups
- ✅ Strong domain knowledge to leverage
- ✅ Want named combinations (e.g., "baseline", "lithology path")

### Best Practice: Two-Step Approach
1. **Run V7** (1-2 hours) → Discover optimal features
2. **Run V7 Final** (3-5 hours) → Train discoveries properly
3. **Result**: Best possible models for production

---

## Configuration Options

### Adjust Number of Trials

Edit `main()` function:
```python
n_trials = 200  # More exploration (instead of 100)
```

### Adjust Feature Penalty

Edit `_objective()` method:
```python
feature_penalty = 0.02 * (n_features / 21.0)  # Stronger penalty (instead of 0.01)
```
- Higher penalty → fewer features selected
- Lower penalty → more features selected

### Change Optimization Metric

Choose during interactive setup or edit:
```python
metric = 'f1_weighted'  # Instead of 'custom_sand'
```

### Parallel Processing

```python
selector.run_optimization(n_trials=100, n_jobs=4)  # 4 parallel jobs
```
⚠️ Caution: High memory usage with multiple jobs

---

## Expected Performance

### Computational Cost

| Mode | Trials | CV Folds | Total Fits | Time |
|------|--------|----------|------------|------|
| Quick | 50 | 4 | 200 | ~30 min |
| Standard | 100 | 4 | 400 | ~1 hour |
| Thorough | 200 | 4 | 800 | ~2 hours |
| Exhaustive | 500 | 4 | 2000 | ~5 hours |

### Expected Results (Standard Mode)

- **Best CV score**: 0.48-0.50 (custom_sand)
- **Optimal feature count**: 6-12 features
- **Top 10 CV range**: 0.46-0.49
- **Essential features found**: 3-5 features
- **Improvement over random**: 5-10%

### Comparison with V6

V6 best (FC_19): **0.480** (8 features, forward selection)

V7 expected:
- 50 trials: Match V6 (0.48)
- 100 trials: Beat V6 (0.48-0.50)
- 200 trials: Significantly beat V6 (0.49-0.51)

---

## Troubleshooting

### "No improvement after 50 trials"
- Increase n_trials to 200
- Check data quality (missing values, outliers)
- Try different metric (custom_sand vs f1_weighted)

### "All trials returning negative scores"
- Check feature availability (missing columns?)
- Verify data format
- Review optimization metric

### "Too many features selected (>15)"
- Increase feature penalty to 0.02 or 0.03
- Adjust in `_objective()` function

### "Out of memory"
- Reduce n_jobs to 1 (sequential)
- Use smaller RF: n_estimators=100
- Process test wells one at a time

### "Very slow progress"
- Reduce CV folds to 3 (in configure_wells)
- Use smaller RF
- Start with Quick mode (50 trials) for testing

---

## Technical Details

### TPE Algorithm

**Tree-structured Parzen Estimator** (Bergstra et al., 2011):

1. Observe trial results
2. Split into "good" (top 25%) and "poor" (bottom 75%)
3. Build probability models:
   - p(features | good results)
   - p(features | poor results)
4. Sample next trial to maximize: p(good) / p(poor)
5. Update models and repeat

**Why TPE?**
- Handles categorical choices (include/exclude features)
- Efficient for large discrete spaces (2^21)
- Balances exploration vs exploitation
- Works well with noisy objectives (CV variance)

### Feature Penalty Rationale

Without penalty: Optuna selects all 21 features (overfitting risk)

With penalty (0.01 per feature ratio):
```python
8 features, CV=0.480  → Objective = 0.480 - 0.0038 = 0.476
15 features, CV=0.485 → Objective = 0.485 - 0.0071 = 0.478
```
Winner: 8-feature model (simpler, nearly same performance)

### Fixed RF Hyperparameters

V7 uses fixed RF params (fast trials):
```python
n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=3
```

**Trade-off**: More feature exploration, less HP tuning per combination

V6 uses GridSearchCV (72 HP combos) but fewer feature combinations (19)

**Justification**: Feature selection matters more than fine-tuning n_estimators

---

## Advanced Usage

### Resume Interrupted Run

```python
import joblib

# Load existing study
study = joblib.load('v7_output/optuna_history/study.pkl')

# Continue optimization
study.optimize(objective_function, n_trials=50)  # 50 more trials
```

### Extract Features Programmatically

```python
import pandas as pd
import json

# Load best combinations
combos = pd.read_csv('v7_output/results/best_feature_combinations.csv')

# Get features from rank 1
with open('v7_output/models/rank_01_trial_042/metadata.json', 'r') as f:
    metadata = json.load(f)

best_features = metadata['features']
print(f"Best features: {best_features}")
```

### Feature Frequency Table

```python
import pandas as pd

combos = pd.read_csv('v7_output/results/best_feature_combinations.csv')

# Count occurrences in top 20
feature_counts = {}
for _, row in combos.head(20).iterrows():
    features = eval(row['features'])
    for feat in features:
        feature_counts[feat] = feature_counts.get(feat, 0) + 1

# Print sorted
for feat, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {count}/20 ({count/20*100:.0f}%)")
```

### Compare Multiple Runs

```bash
# Run with different metrics
python v7_multi_scenario_facies_cv.py  # custom_sand
python v7_multi_scenario_facies_cv.py  # f1_weighted

# Compare results
diff v7_custom_sand/results/best_feature_combinations.csv \
     v7_f1_weighted/results/best_feature_combinations.csv
```

---

## Best Practices

### For Quick Testing (30 min)
```python
n_trials = 50
use_gridsearch = False
```

### For Production (1-2 hours)
```python
n_trials = 100-200
use_gridsearch = False
metric = 'custom_sand'
```

### For Research (5+ hours)
```python
n_trials = 500
use_gridsearch = True  # Full HP search per combination
```

### Ensemble Strategy
- Use top 10-30 models for ensemble
- Weighted voting (by CV score) often better than equal
- More models = smoother predictions but slower

---

## References

### Optuna
- Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD.
- Documentation: https://optuna.org/

### TPE Algorithm
- Bergstra, J., et al. (2011). "Algorithms for Hyper-Parameter Optimization." NIPS.
- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR.

### Feature Selection
- Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." JMLR.
- Chandrashekar, G., & Sahin, F. (2014). "A survey on feature selection methods." Computers & Electrical Engineering.

---

## Summary

**V7 provides:**
- ✅ Intelligent exploration of 2^21 feature space
- ✅ Automatic discovery of optimal 6-12 features
- ✅ 30-40 best combinations in 1-2 hours
- ✅ Data-driven insights with minimal bias
- ✅ Rich visualizations and analysis
- ✅ Production-ready ensemble models

**Start now:**
```bash
pip install optuna>=3.0.0
python v7_multi_scenario_facies_cv.py
```

**Next steps after optimization:**
1. Check `v7_optimization_history.png` for progress
2. Review `best_feature_combinations.csv` for top models
3. Analyze feature frequency in console output
4. Compare with V6 results (if available)
5. Deploy ensemble for predictions

---

**Version**: 7.0  
**Date**: 2026-01-05  
**Status**: Production Ready ✅

