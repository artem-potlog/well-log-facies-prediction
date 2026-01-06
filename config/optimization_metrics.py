"""
Optimization Metric Definitions for V5 Multi-Scenario Approach

Three different optimization metrics to test sensitivity:
1. F1-weighted (standard balanced performance)
2. ROC-AUC OvR (probabilistic + uncertainty)
3. Custom Sand-Weighted (reservoir-focused)
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, make_scorer


def custom_sand_weighted_f1(y_true, y_pred):
    """
    Custom F1 metric emphasizing sand facies (reservoir quality)
    
    Weights reflect:
    - Reservoir importance (economic value)
    - Cost of misclassification
    - Geological interpretability
    
    Facies codes:
    0: Tidal Bar - excellent reservoir
    1: Upper Shoreface - excellent reservoir
    2: Offshore - seal/source, less critical
    3: Tidal Channel - excellent reservoir
    4: Mouthbar - good reservoir
    5: Lower Shoreface - good reservoir
    6: Marsh - non-reservoir
    7: Tidal Flat Muddy - cap/seal
    8: Tidal Flat Sandy - variable quality
    """
    # Facies importance weights
    facies_weights = {
        0: 2.0,  # Tidal Bar - excellent reservoir
        1: 2.0,  # Upper Shoreface - excellent reservoir
        2: 0.5,  # Offshore - seal/source, less critical
        3: 2.0,  # Tidal Channel - excellent reservoir
        4: 1.5,  # Mouthbar - good reservoir
        5: 1.5,  # Lower Shoreface - good reservoir
        6: 0.5,  # Marsh - non-reservoir
        7: 1.0,  # Tidal Flat Muddy - cap/seal
        8: 1.0   # Tidal Flat Sandy - variable quality
    }
    
    # Get per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Apply weights
    weighted_f1_list = []
    class_counts = []
    for facies_code in range(9):  # 9 facies classes
        if facies_code < len(f1_per_class):
            weight = facies_weights.get(facies_code, 1.0)
            count = np.sum(y_true == facies_code)
            weighted_f1_list.append(f1_per_class[facies_code] * weight * count)
            class_counts.append(count * weight)
    
    # Weighted average
    return np.sum(weighted_f1_list) / np.sum(class_counts) if np.sum(class_counts) > 0 else 0.0


def roc_auc_ovr_scorer_func(y_true, y_pred_proba):
    """
    Multi-class ROC-AUC using One-vs-Rest strategy
    Requires predict_proba output
    """
    try:
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    except ValueError:
        # Handle cases where some classes are missing
        return 0.0


# Create sklearn scorers
METRICS = {
    'f1_weighted': {
        'scorer': make_scorer(f1_score, average='weighted', zero_division=0),
        'description': 'F1-Weighted (Standard)',
        'purpose': 'Balanced performance across all facies',
        'requires_proba': False
    },
    
    'roc_auc_ovr': {
        'scorer': make_scorer(roc_auc_ovr_scorer_func, response_method='predict_proba', greater_is_better=True),
        'description': 'ROC-AUC One-vs-Rest',
        'purpose': 'Probabilistic evaluation with uncertainty quantification',
        'requires_proba': True
    },
    
    'custom_sand': {
        'scorer': make_scorer(custom_sand_weighted_f1, greater_is_better=True),
        'description': 'Custom Sand-Weighted F1',
        'purpose': 'Petroleum engineering focus - emphasize reservoir facies',
        'requires_proba': False
    }
}


def get_scorer(metric_name):
    """Get sklearn scorer for a given metric name"""
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRICS.keys())}")
    return METRICS[metric_name]['scorer']


def get_all_metric_names():
    """Get list of all metric names"""
    return list(METRICS.keys())


def print_metric_summary():
    """Print summary of all optimization metrics"""
    print("\n" + "="*80)
    print("OPTIMIZATION METRICS SUMMARY")
    print("="*80)
    
    for metric_name, metric_info in METRICS.items():
        print(f"\n{metric_name}: {metric_info['description']}")
        print(f"  Purpose: {metric_info['purpose']}")
        print(f"  Requires probabilities: {metric_info['requires_proba']}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    print_metric_summary()

