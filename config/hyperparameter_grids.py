"""
Hyperparameter Grid Definitions for V5 Multi-Scenario Approach

Multiple grid sizes for different phases:
- Quick: Fast validation (2-3 hours for 30 models)
- Standard: Balanced performance (6-8 hours for 30 models)
- Full: Exhaustive search (10-14 hours for 30 models)
"""

# Quick grid for Phase 1 validation
PARAM_GRID_QUICK = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [15, 25],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [3]
}

# Standard grid for Phase 2 core analysis
PARAM_GRID_STANDARD = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [15, 20, 25],
    'classifier__min_samples_split': [5, 8],
    'classifier__min_samples_leaf': [2, 3, 5],
    'classifier__max_features': ['sqrt', 'log2']
}

# Full grid for Phase 3 production
PARAM_GRID_FULL = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [15, 20, 25, None],
    'classifier__min_samples_split': [5, 8, 10],
    'classifier__min_samples_leaf': [2, 3, 5],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Minimal grid for debugging
PARAM_GRID_DEBUG = {
    'classifier__n_estimators': [50],
    'classifier__max_depth': [15],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [3]
}


def get_param_grid(grid_type='standard'):
    """
    Get hyperparameter grid by type
    
    Parameters:
    -----------
    grid_type : str
        'debug': Minimal grid for testing (< 1 min per scenario)
        'quick': Fast validation (4-6 min per scenario)
        'standard': Balanced (10-15 min per scenario)
        'full': Exhaustive (20-30 min per scenario)
    
    Returns:
    --------
    dict : Parameter grid for GridSearchCV
    """
    grids = {
        'debug': PARAM_GRID_DEBUG,
        'quick': PARAM_GRID_QUICK,
        'standard': PARAM_GRID_STANDARD,
        'full': PARAM_GRID_FULL
    }
    
    if grid_type not in grids:
        raise ValueError(f"Unknown grid type: {grid_type}. Available: {list(grids.keys())}")
    
    return grids[grid_type]


def count_combinations(param_grid):
    """Count total number of hyperparameter combinations"""
    count = 1
    for param_values in param_grid.values():
        count *= len(param_values)
    return count


def print_grid_summary():
    """Print summary of all hyperparameter grids"""
    print("\n" + "="*80)
    print("HYPERPARAMETER GRIDS SUMMARY")
    print("="*80)
    
    grids = {
        'debug': ('Debug (Testing)', PARAM_GRID_DEBUG),
        'quick': ('Quick (Phase 1 Validation)', PARAM_GRID_QUICK),
        'standard': ('Standard (Phase 2 Core)', PARAM_GRID_STANDARD),
        'full': ('Full (Phase 3 Production)', PARAM_GRID_FULL)
    }
    
    for grid_name, (description, grid) in grids.items():
        n_combos = count_combinations(grid)
        print(f"\n{grid_name.upper()}: {description}")
        print(f"  Combinations: {n_combos}")
        print(f"  With 4-fold CV: {n_combos * 4} fits per scenario")
        print(f"  Estimated time (30 scenarios):")
        
        if grid_name == 'debug':
            print(f"    ~{0.5 * 30:.0f}-{1 * 30:.0f} minutes total")
        elif grid_name == 'quick':
            print(f"    ~{4 * 30 / 60:.1f}-{6 * 30 / 60:.1f} hours total")
        elif grid_name == 'standard':
            print(f"    ~{10 * 30 / 60:.1f}-{15 * 30 / 60:.1f} hours total")
        elif grid_name == 'full':
            print(f"    ~{20 * 30 / 60:.1f}-{30 * 30 / 60:.1f} hours total")
        
        print(f"  Parameters:")
        for param, values in grid.items():
            param_short = param.replace('classifier__', '')
            print(f"    {param_short}: {values}")
    
    print("\n" + "="*80)
    print("\nRECOMMENDATION:")
    print("  - Use 'debug' for code testing (< 1 hour)")
    print("  - Use 'quick' for Phase 1 validation (2-3 hours)")
    print("  - Use 'standard' for Phase 2 core analysis (5-8 hours)")
    print("  - Use 'full' for Phase 3 production run (10-15 hours)")
    print("="*80)


if __name__ == '__main__':
    print_grid_summary()

