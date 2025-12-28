"""
Random Forest Classification with Uncertainty Quantification - Version 4
Based on Halotel–Demyanov–Gardiner (HDG) Approach

Key Features:
- Fixed blind test well(s) kept completely separate
- Systematic train/validation splits using entire wells as folds
- Hyperparameter tuning on validation set for each combination
- Comprehensive uncertainty quantification through ensemble predictions
- Per-depth facies proportion analysis (similar to their Fig. 9)

Reference: Halotel, Demyanov, Gardiner - "Value of information of time-lapse seismic 
data for prediction improvement of reservoir production" (SPE/Petroleum Geoscience)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Define facies mapping
FACIES_NAMES = {
    0: 'TIDAL BAR',
    1: 'UPPER SHOREFACE', 
    2: 'OFFSHORE',
    3: 'TIDAL CHANNEL',
    4: 'MOUTHBAR',
    5: 'LOWER SHOREFACE',
    6: 'MARSH',
    7: 'TIDAL FLAT MUDDY',
    8: 'TIDAL FLAT SANDY'
}

# Define all features to use
ALL_FEATURES = ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH',
                'GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                'NTG_Slope', 'AI_Slope', 'BaseSharpness',
                'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4']

# Hyperparameter grid for tuning (simplified for speed)
PARAM_GRID = {
    'n_estimators': [50, 100],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 8],
    'min_samples_leaf': [3, 5]
}

def load_data():
    """Load dataset and prepare for HDG ensemble training"""
    print("="*80)
    print("RANDOM FOREST ENSEMBLE - HDG APPROACH (V4)")
    print("Halotel–Demyanov–Gardiner Method with Train/Val/Test Splits")
    print("="*80)
    
    # Load the dataset
    dataset_path = 'ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv'
    print(f"\nLoading dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{dataset_path}' not found!")
        return None, None
    
    # Filter to only samples with facies labels
    df_with_facies = df[df['Equinor facies'].notna()].copy()
    print(f"Total samples with facies labels: {len(df_with_facies)} / {len(df)}")
    
    # Get available features
    features_to_use = [f for f in ALL_FEATURES if f in df.columns]
    print(f"Using {len(features_to_use)} features")
    
    # Handle NaN values
    for col in features_to_use:
        if col in df_with_facies.columns:
            median_val = df_with_facies[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_with_facies[col] = df_with_facies[col].fillna(median_val)
    
    return df_with_facies, features_to_use

def select_hdg_configuration(df):
    """Select test well and validation configuration for HDG approach"""
    
    wells = sorted(df['Well'].unique())
    print(f"\nAvailable wells with facies labels ({len(wells)}):")
    for i, well in enumerate(wells):
        sample_count = len(df[df['Well'] == well])
        well_name = well.replace('15_9-', '')
        if 'F-14' in well:
            print(f"  {i+1}. {well_name} ({sample_count} samples) <- F-14 with predicted PHIF/SW")
        else:
            print(f"  {i+1}. {well_name} ({sample_count} samples)")
    
    print("\n" + "="*60)
    print("HDG CONFIGURATION")
    print("="*60)
    
    print("\nStep 1: Select BLIND TEST WELL(S) (kept completely separate):")
    print("Enter comma-separated numbers (e.g., 5 for just F-14):")
    test_indices = input("Test wells: ").strip().split(',')
    test_wells = []
    for idx in test_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                test_wells.append(wells[well_idx])
    
    # Remaining wells for train/validation splits
    remaining_wells = [w for w in wells if w not in test_wells]
    
    print(f"\n✓ Blind test well(s): {[w.replace('15_9-', '') for w in test_wells]}")
    print(f"✓ Remaining wells for train/val splits: {len(remaining_wells)}")
    
    print("\nStep 2: Configure train/validation splits:")
    print(f"Available wells for train/val: {[w.replace('15_9-', '') for w in remaining_wells]}")
    
    # Show possible configurations
    print("\nPossible validation set sizes:")
    max_val = min(3, len(remaining_wells) - 1)  # Need at least 1 for training
    for k in range(1, max_val + 1):
        n_combinations = len(list(combinations(remaining_wells, k)))
        print(f"  {k} validation well(s) -> {n_combinations} combinations")
    
    print(f"\nHow many wells to use for VALIDATION in each split?")
    print(f"(Recommended: 1 for <6 wells, 2 for 6-10 wells)")
    n_val_wells = int(input("Validation wells per split: ").strip())
    
    # Generate all train/validation combinations
    val_combinations = list(combinations(remaining_wells, n_val_wells))
    train_val_splits = []
    
    for val_wells in val_combinations:
        train_wells = [w for w in remaining_wells if w not in val_wells]
        train_val_splits.append({
            'train': train_wells,
            'validation': list(val_wells)
        })
    
    print("\n" + "="*60)
    print(f"HDG ENSEMBLE STRUCTURE")
    print("="*60)
    
    print(f"\n✓ Blind test well(s): {[w.replace('15_9-', '') for w in test_wells]}")
    print(f"✓ Number of train/val combinations: {len(train_val_splits)}")
    print(f"✓ Each split: {len(remaining_wells) - n_val_wells} training, {n_val_wells} validation")
    
    # Show all splits
    print("\n" + "-"*60)
    print("TRAIN/VALIDATION SPLITS:")
    print("-"*60)
    for i, split in enumerate(train_val_splits):
        train_names = [w.replace('15_9-', '') for w in split['train']]
        val_names = [w.replace('15_9-', '') for w in split['validation']]
        print(f"  Split {i+1:2d}: Train={train_names}, Val={val_names}")
    
    # Ask user to confirm
    print("\n" + "-"*60)
    confirm = input("\nProceed with this HDG configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled. Please restart.")
        return None, None, None
    
    return test_wells, train_val_splits, remaining_wells

def visualize_hdg_splits(train_val_splits, test_wells, remaining_wells):
    """Create visualization of HDG train/val/test splits"""
    
    n_splits = len(train_val_splits)
    n_wells = len(remaining_wells)
    
    # Create matrix: 0=not used, 1=train, 2=validation
    split_matrix = np.zeros((n_splits, n_wells))
    
    for i, split in enumerate(train_val_splits):
        for j, well in enumerate(remaining_wells):
            if well in split['train']:
                split_matrix[i, j] = 1  # Training
            elif well in split['validation']:
                split_matrix[i, j] = 2  # Validation
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n_splits*0.3)))
    
    # Custom colormap: white=not used, blue=train, orange=validation
    from matplotlib.colors import ListedColormap
    colors = ['white', 'steelblue', 'orange']
    cmap = ListedColormap(colors)
    
    # Plot 1: Split matrix
    well_names = [w.replace('15_9-', '') for w in remaining_wells]
    
    im = ax1.imshow(split_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax1.set_yticks(range(n_splits))
    ax1.set_yticklabels([f'Split {i+1}' for i in range(n_splits)])
    ax1.set_xticks(range(n_wells))
    ax1.set_xticklabels(well_names, rotation=45, ha='right')
    ax1.set_xlabel('Wells')
    ax1.set_ylabel('Train/Val Splits')
    ax1.set_title(f'HDG Train/Validation Splits\n(Blue=Train, Orange=Val)')
    
    # Add grid
    for i in range(n_splits + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(n_wells + 1):
        ax1.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # Plot 2: Well usage frequency
    train_usage = (split_matrix == 1).sum(axis=0)
    val_usage = (split_matrix == 2).sum(axis=0)
    
    x = np.arange(n_wells)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, train_usage, width, label='Training', color='steelblue')
    bars2 = ax2.bar(x + width/2, val_usage, width, label='Validation', color='orange')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(well_names, rotation=45, ha='right')
    ax2.set_ylabel('Number of Splits')
    ax2.set_title('Well Usage Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add test wells information
    test_names = [w.replace('15_9-', '') for w in test_wells]
    fig.text(0.5, 0.02, f'Blind Test Wells: {", ".join(test_names)}', 
             ha='center', fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    return fig

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Tune hyperparameters on validation set"""
    
    best_score = -1
    best_params = None
    
    # Grid search (simplified for speed)
    for params in ParameterGrid(PARAM_GRID):
        rf = RandomForestClassifier(
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        rf.fit(X_train, y_train)
        y_pred_val = rf.predict(X_val)
        
        # Use F1 score for imbalanced classes
        score = f1_score(y_val, y_pred_val, average='weighted')
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score

def train_hdg_ensemble(df, train_val_splits, test_wells, features, use_tuning=True):
    """Train HDG ensemble with train/validation splits"""
    
    print("\n" + "="*60)
    print("TRAINING HDG ENSEMBLE")
    print("="*60)
    
    # Prepare test data once
    test_data = df[df['Well'].isin(test_wells)].copy()
    X_test = test_data[features].values
    y_test = test_data['Equinor facies'].astype(int).values
    
    print(f"Test set size: {len(X_test)} samples from {len(test_wells)} well(s)")
    
    if use_tuning:
        print("Hyperparameter tuning: ENABLED (slower but better)")
    else:
        print("Hyperparameter tuning: DISABLED (faster)")
    
    # Store results
    ensemble_predictions = []
    ensemble_probabilities = []
    model_performances = []
    
    print(f"\nTraining {len(train_val_splits)} models with train/val splits...")
    print("-"*60)
    
    for i, split in enumerate(train_val_splits):
        # Prepare data
        train_data = df[df['Well'].isin(split['train'])].copy()
        val_data = df[df['Well'].isin(split['validation'])].copy()
        
        X_train = train_data[features].values
        y_train = train_data['Equinor facies'].astype(int).values
        X_val = val_data[features].values
        y_val = val_data['Equinor facies'].astype(int).values
        
        train_names = [w.replace('15_9-', '') for w in split['train']]
        val_names = [w.replace('15_9-', '') for w in split['validation']]
        
        print(f"Split {i+1:2d}/{len(train_val_splits)}: Train={train_names}, Val={val_names}")
        
        if use_tuning:
            # Tune hyperparameters on validation set
            best_params, val_score = tune_hyperparameters(X_train, y_train, X_val, y_val)
            print(f"  Best params: {best_params}")
            print(f"  Val F1 score: {val_score:.3f}")
            
            # Retrain on combined train+val with best params
            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.hstack([y_train, y_val])
            
            rf = RandomForestClassifier(
                max_features='sqrt',
                class_weight='balanced',
                random_state=42 + i,
                n_jobs=-1,
                **best_params
            )
        else:
            # Use default parameters and train on train+val
            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.hstack([y_train, y_val])
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42 + i,
                n_jobs=-1
            )
            
            # Still calculate validation score for reporting
            rf_val = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42 + i,
                n_jobs=-1
            )
            rf_val.fit(X_train, y_train)
            y_pred_val = rf_val.predict(X_val)
            val_score = f1_score(y_val, y_pred_val, average='weighted')
            print(f"  Val F1 score: {val_score:.3f}")
        
        # Train final model
        rf.fit(X_train_full, y_train_full)
        
        # Predict on test set
        y_pred_test = rf.predict(X_test)
        y_proba_test = rf.predict_proba(X_test)
        
        # Calculate test accuracy
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy: {test_acc:.3f}")
        
        # Store results
        ensemble_predictions.append(y_pred_test)
        ensemble_probabilities.append(y_proba_test)
        
        model_performances.append({
            'split_id': i + 1,
            'train_wells': train_names,
            'val_wells': val_names,
            'val_score': val_score,
            'test_accuracy': test_acc,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val)
        })
        
        print("-"*60)
    
    print(f"\n✓ All {len(train_val_splits)} models trained successfully")
    
    # Convert to numpy array
    ensemble_predictions = np.array(ensemble_predictions)
    
    # Summary statistics
    test_accs = [m['test_accuracy'] for m in model_performances]
    val_scores = [m['val_score'] for m in model_performances]
    
    print("\nMODEL PERFORMANCE SUMMARY:")
    print(f"  Validation F1 - Mean: {np.mean(val_scores):.4f} (±{np.std(val_scores):.4f})")
    print(f"  Test Accuracy - Mean: {np.mean(test_accs):.4f} (±{np.std(test_accs):.4f})")
    print(f"  Test Accuracy - Best: {np.max(test_accs):.4f} (Split {np.argmax(test_accs)+1})")
    print(f"  Test Accuracy - Worst: {np.min(test_accs):.4f} (Split {np.argmin(test_accs)+1})")
    
    return ensemble_predictions, ensemble_probabilities, model_performances, test_data, y_test

def calculate_hdg_uncertainty(ensemble_predictions, test_data):
    """Calculate uncertainty metrics using HDG approach"""
    
    print("\n" + "="*60)
    print("HDG UNCERTAINTY QUANTIFICATION")
    print("="*60)
    
    n_models, n_samples = ensemble_predictions.shape
    
    # Get all possible facies values
    all_facies = np.unique(ensemble_predictions)
    n_facies = len(all_facies)
    
    print(f"Number of models: {n_models}")
    print(f"Number of test samples: {n_samples}")
    print(f"Facies classes: {all_facies}")
    
    # Create proportion matrix
    facies_proportions = np.zeros((n_samples, n_facies))
    
    # For each sample, count predictions from all models
    for i in range(n_samples):
        predictions_at_i = ensemble_predictions[:, i]
        for j, facies in enumerate(all_facies):
            count = np.sum(predictions_at_i == facies)
            facies_proportions[i, j] = count / n_models
    
    # Create DataFrame with proportions
    proportion_columns = [f'Proportion_Facies_{int(f)}' for f in all_facies]
    hdg_results = pd.DataFrame(facies_proportions, columns=proportion_columns)
    
    # Add metadata
    hdg_results['Well'] = test_data['Well'].values
    hdg_results['Depth'] = test_data['Depth'].values
    hdg_results['True_Facies'] = test_data['Equinor facies'].astype(int).values
    
    # Most likely facies (ensemble vote)
    hdg_results['HDG_Prediction'] = all_facies[np.argmax(facies_proportions, axis=1)]
    
    # Uncertainty metrics specific to HDG
    hdg_results['Max_Proportion'] = np.max(facies_proportions, axis=1)
    hdg_results['Vote_Agreement'] = hdg_results['Max_Proportion'] * 100
    
    # Entropy-based uncertainty
    entropy = -np.sum(facies_proportions * np.log(facies_proportions + 1e-10), axis=1)
    hdg_results['Prediction_Entropy'] = entropy
    hdg_results['HDG_Uncertainty'] = 1 - hdg_results['Max_Proportion']
    
    # Calculate range of predictions (HDG specific)
    prediction_range = np.zeros(n_samples)
    for i in range(n_samples):
        unique_preds = np.unique(ensemble_predictions[:, i])
        prediction_range[i] = len(unique_preds)
    hdg_results['Prediction_Range'] = prediction_range
    hdg_results['Normalized_Range'] = prediction_range / n_facies
    
    # Calculate ensemble accuracy
    ensemble_acc = accuracy_score(hdg_results['True_Facies'], hdg_results['HDG_Prediction'])
    
    print(f"\n✓ HDG Ensemble accuracy: {ensemble_acc:.4f}")
    
    # HDG-specific statistics
    print("\nHDG UNCERTAINTY METRICS:")
    print(f"  Mean prediction range: {hdg_results['Prediction_Range'].mean():.2f} facies")
    print(f"  Samples with full agreement: {(hdg_results['Vote_Agreement'] == 100).sum()}")
    print(f"  Samples with >3 different predictions: {(hdg_results['Prediction_Range'] > 3).sum()}")
    print(f"  Mean entropy: {hdg_results['Prediction_Entropy'].mean():.3f}")
    
    return hdg_results, all_facies

def create_facies_proportion_logs(hdg_results, all_facies, test_wells, output_dir):
    """Create individual 3-panel facies proportion logs for each test well (FROM V3)"""
    
    print("\n" + "="*60)
    print("CREATING INDIVIDUAL WELL FACIES PROPORTION LOGS")
    print("="*60)
    
    # Create a log plot for each test well
    for well in test_wells:
        well_name = well.replace('15_9-', '')
        well_data = hdg_results[hdg_results['Well'] == well].sort_values('Depth')
        
        if len(well_data) == 0:
            continue
        
        print(f"Creating facies proportion log for well {well_name}...")
        
        # Create figure for this well (3-panel format like V3)
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        
        depths = well_data['Depth'].values
        
        # Define consistent colors for facies
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_facies)))
        facies_colors = {f: colors[i] for i, f in enumerate(all_facies)}
        
        # Get proportion columns
        prop_cols = [col for col in hdg_results.columns if col.startswith('Proportion_Facies_')]
        
        # 1. Stacked area plot of facies proportions
        ax = axes[0]
        
        # Create stacked data
        bottom = np.zeros(len(depths))
        
        for i, (col, facies) in enumerate(zip(prop_cols, all_facies)):
            props = well_data[col].values
            facies_name = FACIES_NAMES.get(facies, f'F{facies}')
            ax.fill_betweenx(depths, bottom, bottom + props, 
                            alpha=0.7, label=facies_name, color=facies_colors[facies])
            bottom += props
        
        ax.set_xlabel('Cumulative Proportion')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Facies Proportions (HDG Ensemble)')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty profile
        ax = axes[1]
        uncertainty = well_data['HDG_Uncertainty'].values
        ax.plot(uncertainty, depths, 'r-', linewidth=1)
        ax.fill_betweenx(depths, 0, uncertainty, alpha=0.3, color='red')
        ax.set_xlabel('HDG Uncertainty')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Uncertainty Profile')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_unc = uncertainty.mean()
        std_unc = uncertainty.std()
        ax.axvline(mean_unc, color='blue', linestyle='--', 
                  label=f'Mean: {mean_unc:.3f}')
        ax.axvline(mean_unc + std_unc, color='green', linestyle=':', 
                  label=f'+1σ: {mean_unc + std_unc:.3f}')
        ax.axvline(max(0, mean_unc - std_unc), color='green', linestyle=':', 
                  label=f'-1σ: {max(0, mean_unc - std_unc):.3f}')
        ax.legend()
        
        # 3. True vs HDG Predicted
        ax = axes[2]
        
        true_facies = well_data['True_Facies'].values
        hdg_pred = well_data['HDG_Prediction'].values
        
        # Plot as colored strips
        for i in range(len(depths)-1):
            # True facies (left half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0, 0.45,
                           color=facies_colors.get(true_facies[i], 'gray'), alpha=0.7)
            # HDG prediction (right half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0.55, 1,
                           color=facies_colors.get(hdg_pred[i], 'gray'), alpha=0.7)
        
        ax.set_xlim([0, 1])
        ax.set_xticks([0.225, 0.775])
        ax.set_xticklabels(['True', 'HDG Predicted'])
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - True vs HDG Ensemble')
        ax.invert_yaxis()
        
        # Calculate accuracy for this well
        from sklearn.metrics import accuracy_score
        well_acc = accuracy_score(true_facies, hdg_pred)
        ax.text(0.5, 0.02, f'Accuracy: {well_acc:.3f}', 
               transform=ax.transAxes, ha='center', fontweight='bold')
        
        # Add legend for facies colors
        legend_elements = []
        for facies in np.unique(np.concatenate([true_facies, hdg_pred])):
            facies_name = FACIES_NAMES.get(facies, f'F{facies}')
            legend_elements.append(mpatches.Rectangle((0,0),1,1, fc=facies_colors.get(facies, 'gray'), 
                                               alpha=0.7, label=facies_name))
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        
        plt.suptitle(f'HDG Facies Proportion Log - Well {well_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save individual well plot
        well_filename = os.path.join(output_dir, f'HDG_Well_{well_name}_facies_log.png')
        plt.savefig(well_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved: {well_filename}")
    
    print(f"\n✓ Created individual facies proportion logs for {len(test_wells)} test well(s)")

def create_hdg_depth_proportions(hdg_results, all_facies, test_wells, output_dir):
    """Create comprehensive HDG analysis plots (6-panel format)"""
    
    print("\nCreating comprehensive HDG analysis visualizations...")
    
    for well in test_wells:
        well_name = well.replace('15_9-', '')
        well_data = hdg_results[hdg_results['Well'] == well].sort_values('Depth')
        
        if len(well_data) == 0:
            continue
        
        # Create figure with multiple panels
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        depths = well_data['Depth'].values
        
        # Define consistent colors for facies
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_facies)))
        facies_colors = {f: colors[i] for i, f in enumerate(all_facies)}
        
        # 1. Stacked facies proportions (HDG style)
        ax = axes[0, 0]
        prop_cols = [col for col in hdg_results.columns if col.startswith('Proportion_Facies_')]
        
        # Create stacked area plot
        bottom = np.zeros(len(depths))
        for i, (col, facies) in enumerate(zip(prop_cols, all_facies)):
            props = well_data[col].values
            facies_name = FACIES_NAMES.get(facies, f'F{facies}')
            ax.fill_betweenx(depths, bottom, bottom + props, 
                            alpha=0.8, label=facies_name, color=facies_colors[facies])
            bottom += props
        
        ax.set_xlabel('Facies Proportion')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - HDG Facies Proportions')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # 2. HDG Uncertainty profile
        ax = axes[0, 1]
        uncertainty = well_data['HDG_Uncertainty'].values
        ax.plot(uncertainty, depths, 'r-', linewidth=1.5)
        ax.fill_betweenx(depths, 0, uncertainty, alpha=0.3, color='red')
        ax.set_xlabel('HDG Uncertainty')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Uncertainty Profile')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axvline(uncertainty.mean(), color='blue', linestyle='--', 
                  label=f'Mean: {uncertainty.mean():.3f}')
        ax.axvline(uncertainty.std(), color='green', linestyle='--', 
                  label=f'Std: {uncertainty.std():.3f}')
        ax.legend()
        
        # 3. Prediction Range (HDG specific)
        ax = axes[0, 2]
        pred_range = well_data['Prediction_Range'].values
        ax.plot(pred_range, depths, 'b-', linewidth=1.5)
        ax.fill_betweenx(depths, 0, pred_range, alpha=0.3, color='blue')
        ax.set_xlabel('Number of Different Predictions')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Prediction Diversity')
        ax.invert_yaxis()
        ax.set_xlim([0, len(all_facies)])
        ax.grid(True, alpha=0.3)
        
        # 4. True vs HDG Predicted (colored strips)
        ax = axes[1, 0]
        
        true_facies = well_data['True_Facies'].values
        hdg_pred = well_data['HDG_Prediction'].values
        
        # Plot as colored strips
        for i in range(len(depths)-1):
            # True facies (left half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0, 0.45,
                           color=facies_colors.get(true_facies[i], 'gray'), alpha=0.8)
            # HDG prediction (right half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0.55, 1,
                           color=facies_colors.get(hdg_pred[i], 'gray'), alpha=0.8)
        
        ax.set_xlim([0, 1])
        ax.set_xticks([0.225, 0.775])
        ax.set_xticklabels(['True', 'HDG Predicted'])
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - True vs HDG Ensemble')
        ax.invert_yaxis()
        
        # Calculate accuracy for this well
        well_acc = accuracy_score(true_facies, hdg_pred)
        ax.text(0.5, 0.02, f'Accuracy: {well_acc:.3f}', 
               transform=ax.transAxes, ha='center', fontweight='bold')
        
        # 5. Vote Agreement Heatmap
        ax = axes[1, 1]
        
        # Create a 2D representation of vote agreement over depth
        agreement_matrix = well_data['Vote_Agreement'].values.reshape(-1, 1)
        im = ax.imshow(agreement_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_yticks([])
        ax.set_xlabel('Depth Index')
        ax.set_title(f'{well_name} - Model Agreement Heatmap')
        plt.colorbar(im, ax=ax, label='Agreement (%)')
        
        # 6. Entropy over depth
        ax = axes[1, 2]
        entropy = well_data['Prediction_Entropy'].values
        ax.plot(entropy, depths, 'g-', linewidth=1.5)
        ax.fill_betweenx(depths, 0, entropy, alpha=0.3, color='green')
        ax.set_xlabel('Prediction Entropy')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Information Entropy')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Add reference line for maximum entropy
        max_entropy = np.log(len(all_facies))
        ax.axvline(max_entropy, color='red', linestyle='--', 
                  label=f'Max: {max_entropy:.2f}')
        ax.legend()
        
        plt.suptitle(f'HDG Comprehensive Analysis - Well {well_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_filename = os.path.join(output_dir, f'HDG_Well_{well_name}_comprehensive_analysis.png')
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved: {fig_filename}")

def create_hdg_summary_visualization(model_performances, hdg_results, all_facies, output_dir, split_fig):
    """Create comprehensive HDG summary visualization"""
    
    # Save split visualization
    split_fig.savefig(os.path.join(output_dir, 'hdg_train_val_splits.png'), dpi=150, bbox_inches='tight')
    plt.close(split_fig)
    
    # Create summary figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Validation vs Test Performance
    ax1 = plt.subplot(2, 3, 1)
    val_scores = [m['val_score'] for m in model_performances]
    test_accs = [m['test_accuracy'] for m in model_performances]
    
    ax1.scatter(val_scores, test_accs, alpha=0.6, s=50)
    ax1.set_xlabel('Validation F1 Score')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Validation vs Test Performance')
    
    # Add trend line
    z = np.polyfit(val_scores, test_accs, 1)
    p = np.poly1d(z)
    ax1.plot(val_scores, p(val_scores), "r--", alpha=0.5)
    
    # Add correlation
    corr = np.corrcoef(val_scores, test_accs)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax1.transAxes, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(test_accs, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(test_accs), color='red', linestyle='--', 
               label=f'Mean: {np.mean(test_accs):.3f}')
    ax2.axvline(np.median(test_accs), color='orange', linestyle='--', 
               label=f'Median: {np.median(test_accs):.3f}')
    ax2.set_xlabel('Test Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of {len(test_accs)} Model Accuracies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. HDG Uncertainty Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(hdg_results['HDG_Uncertainty'], bins=30, 
             color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(hdg_results['HDG_Uncertainty'].mean(), color='red', 
               linestyle='--', label=f"Mean: {hdg_results['HDG_Uncertainty'].mean():.3f}")
    ax3.set_xlabel('HDG Uncertainty')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Prediction Uncertainty')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix for HDG Ensemble
    ax4 = plt.subplot(2, 3, 4)
    from sklearn.metrics import confusion_matrix
    
    true_facies = hdg_results['True_Facies'].values
    hdg_pred = hdg_results['HDG_Prediction'].values
    
    unique_facies = sorted(np.unique(np.concatenate([true_facies, hdg_pred])))
    facies_names = [FACIES_NAMES.get(f, f'F{f}') for f in unique_facies]
    
    cm = confusion_matrix(true_facies, hdg_pred, labels=unique_facies)
    
    # Normalize for colors
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=facies_names, yticklabels=facies_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    
    ensemble_acc = accuracy_score(true_facies, hdg_pred)
    ax4.set_title(f'HDG Ensemble Confusion Matrix\nAccuracy: {ensemble_acc:.3f}')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Prediction Range Analysis
    ax5 = plt.subplot(2, 3, 5)
    range_counts = hdg_results['Prediction_Range'].value_counts().sort_index()
    ax5.bar(range_counts.index, range_counts.values, color='steelblue')
    ax5.set_xlabel('Number of Different Predictions per Sample')
    ax5.set_ylabel('Count')
    ax5.set_title('Prediction Diversity Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "HDG ENSEMBLE CONFIGURATION\n" + "="*40 + "\n\n"
    summary_text += f"Train/Val Splits: {len(model_performances)}\n"
    summary_text += f"Test Samples: {len(hdg_results)}\n\n"
    
    summary_text += "MODEL PERFORMANCE:\n"
    summary_text += f"  Val F1 - Mean: {np.mean(val_scores):.4f} (±{np.std(val_scores):.4f})\n"
    summary_text += f"  Test Acc - Mean: {np.mean(test_accs):.4f} (±{np.std(test_accs):.4f})\n"
    summary_text += f"  HDG Ensemble Acc: {ensemble_acc:.4f}\n"
    summary_text += f"  Improvement: {(ensemble_acc - np.mean(test_accs))*100:.1f}%\n\n"
    
    summary_text += "UNCERTAINTY METRICS:\n"
    summary_text += f"  Mean HDG Uncertainty: {hdg_results['HDG_Uncertainty'].mean():.3f}\n"
    summary_text += f"  Mean Prediction Range: {hdg_results['Prediction_Range'].mean():.2f}\n"
    summary_text += f"  Mean Entropy: {hdg_results['Prediction_Entropy'].mean():.3f}\n"
    summary_text += f"  Full Agreement: {(hdg_results['Vote_Agreement'] == 100).sum()} samples\n"
    summary_text += f"  High Uncertainty (>0.5): {(hdg_results['HDG_Uncertainty'] > 0.5).sum()} samples\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', family='monospace')
    
    plt.suptitle(f'HDG Uncertainty Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hdg_summary_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def save_hdg_results(hdg_results, model_performances, output_dir):
    """Save HDG results and detailed reports"""
    
    print("\n" + "="*60)
    print("SAVING HDG RESULTS")
    print("="*60)
    
    # Save HDG results DataFrame
    hdg_results.to_csv(os.path.join(output_dir, 'hdg_results.csv'), index=False)
    print("✓ Saved HDG results")
    
    # Save model performance details
    perf_df = pd.DataFrame(model_performances)
    perf_df.to_csv(os.path.join(output_dir, 'hdg_model_performances.csv'), index=False)
    print("✓ Saved model performances")
    
    # Create detailed report
    report_file = os.path.join(output_dir, 'hdg_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("HDG UNCERTAINTY QUANTIFICATION REPORT\n")
        f.write("Based on Halotel–Demyanov–Gardiner Approach\n")
        f.write("="*60 + "\n\n")
        
        f.write("TRAIN/VALIDATION SPLITS:\n")
        f.write("-"*40 + "\n")
        for perf in model_performances:
            f.write(f"Split {perf['split_id']:2d}: Train={perf['train_wells']}, Val={perf['val_wells']}\n")
            f.write(f"         Val F1: {perf['val_score']:.4f}, Test Acc: {perf['test_accuracy']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("HDG ENSEMBLE STATISTICS:\n")
        f.write("-"*40 + "\n")
        
        ensemble_acc = accuracy_score(hdg_results['True_Facies'], 
                                     hdg_results['HDG_Prediction'])
        
        f.write(f"Number of Train/Val Splits: {len(model_performances)}\n")
        f.write(f"Mean Test Accuracy: {np.mean([m['test_accuracy'] for m in model_performances]):.4f}\n")
        f.write(f"HDG Ensemble Accuracy: {ensemble_acc:.4f}\n")
        
        f.write("\nUNCERTAINTY QUANTIFICATION:\n")
        f.write(f"  Mean HDG Uncertainty: {hdg_results['HDG_Uncertainty'].mean():.4f}\n")
        f.write(f"  Std HDG Uncertainty: {hdg_results['HDG_Uncertainty'].std():.4f}\n")
        f.write(f"  Mean Prediction Range: {hdg_results['Prediction_Range'].mean():.2f} facies\n")
        f.write(f"  Max Prediction Range: {hdg_results['Prediction_Range'].max()} facies\n")
    
    print("✓ Saved HDG report")
    print(f"✓ All results saved to: {output_dir}/")

def main():
    """Main execution function for HDG approach"""
    
    # Load data
    df, features = load_data()
    if df is None:
        return
    
    # Configure HDG splits
    result = select_hdg_configuration(df)
    if result[0] is None:
        return
    
    test_wells, train_val_splits, remaining_wells = result
    
    # Visualize splits
    split_fig = visualize_hdg_splits(train_val_splits, test_wells, remaining_wells)
    
    # Ask about hyperparameter tuning
    print("\n" + "-"*60)
    print("Enable hyperparameter tuning? (slower but potentially better)")
    print("Enter 'y' for yes (recommended) or 'n' for no (faster):")
    use_tuning = input("Use tuning? (y/n): ").strip().lower() == 'y'
    
    # Train HDG ensemble
    (ensemble_predictions, ensemble_probabilities, 
     model_performances, test_data, y_test) = train_hdg_ensemble(
        df, train_val_splits, test_wells, features, use_tuning
    )
    
    # Calculate HDG uncertainty metrics
    hdg_results, all_facies = calculate_hdg_uncertainty(
        ensemble_predictions, test_data
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"RF_HDG_Uncertainty_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    # First: Individual 3-panel facies proportion logs (like V3)
    create_facies_proportion_logs(hdg_results, all_facies, test_wells, output_dir)
    
    # Second: Comprehensive 6-panel HDG analysis
    create_hdg_depth_proportions(hdg_results, all_facies, test_wells, output_dir)
    
    # Third: Summary visualizations
    create_hdg_summary_visualization(model_performances, hdg_results, all_facies, 
                                    output_dir, split_fig)
    
    # Save results
    save_hdg_results(hdg_results, model_performances, output_dir)
    
    print("\n" + "="*60)
    print("HDG ANALYSIS COMPLETE")
    print("="*60)
    print(f"✓ Trained {len(model_performances)} models with train/val/test splits")
    print(f"✓ Created HDG uncertainty quantification")
    print(f"✓ Generated individual facies proportion logs (3-panel format)")
    print(f"✓ Generated comprehensive HDG analysis (6-panel format)")
    print(f"✓ Created summary visualizations")
    print(f"✓ All results saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
