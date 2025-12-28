"""
Random Forest Classification for Depositional Facies Prediction
Flexible version that works with datasets containing any subset of features
Includes support for lithology features (one-hot encoded)
Version 2 with Lithology and Engineered Features MAX (EF-MAX): Removed train/test split option - only LOWO and manual well selection
Supports all engineered features: GR_Slope, GR_Serration, RelPos, OppositionIndex, NTG_Slope, AI_Slope, BaseSharpness
Compatible with ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_AllEngineered_facies_litho.csv

FIXED VERSION: Removed double normalization since ML_ready datasets are already normalized
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from datetime import datetime

# Define facies mapping based on Equinor interpretation
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

# Define all possible features and their categories
# IMPORTANT: Keep this exact order to match the original script
ALL_POSSIBLE_FEATURES = ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH', 
                        'GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                        'NTG_Slope', 'AI_Slope', 'BaseSharpness']
RAW_LOGS = ['GR', 'RT', 'RD', 'NPHI', 'RHOB', 'DT']
DERIVED_FEATURES = ['VSH', 'PHIF', 'SW', 'KLOGH']
ENGINEERED_FEATURES = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                      'NTG_Slope', 'AI_Slope', 'BaseSharpness']

# Define lithology columns (one-hot encoded)
LITHOLOGY_COLUMNS = ['Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4']

def detect_available_features(df):
    """Detect which features are available in the dataset, maintaining order"""
    available_features = []
    available_raw_logs = []
    available_derived = []
    available_engineered = []
    available_lithology = []
    
    # Maintain the exact order from ALL_POSSIBLE_FEATURES
    for feature in ALL_POSSIBLE_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
            if feature in RAW_LOGS:
                available_raw_logs.append(feature)
            elif feature in DERIVED_FEATURES:
                available_derived.append(feature)
            elif feature in ENGINEERED_FEATURES:
                available_engineered.append(feature)
    
    # Check for lithology columns
    for litho_col in LITHOLOGY_COLUMNS:
        if litho_col in df.columns:
            available_lithology.append(litho_col)
    
    return available_features, available_raw_logs, available_derived, available_engineered, available_lithology

def load_and_explore_data():
    """Load dataset and show basic information"""
    print("Looking for ML-ready datasets...")
    
    # Try to find any ML-ready CSV file
    import glob
    csv_files = glob.glob("ML_ready*.csv")
    
    if not csv_files:
        print("No ML-ready CSV files found in the current directory.")
        csv_file = input("Please enter the path to your dataset: ").strip()
    else:
        print(f"\nFound {len(csv_files)} ML-ready dataset(s):")
        for i, file in enumerate(csv_files):
            print(f"  {i+1}. {file}")
        
        if len(csv_files) == 1:
            csv_file = csv_files[0]
            print(f"\nUsing: {csv_file}")
        else:
            choice = input(f"\nSelect dataset (1-{len(csv_files)}): ").strip()
            csv_file = csv_files[int(choice)-1] if choice.isdigit() else csv_files[0]
    
    print(f"\nLoading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Detect facies column
    facies_col = None
    for col in ['Equinor facies', 'Facies', 'facies']:
        if col in df.columns:
            facies_col = col
            break
    
    if facies_col is None:
        print("Error: No facies column found in dataset")
        return None, None, None, None, None, None, None
    
    # Detect available features including lithology and engineered features
    available_features, available_raw_logs, available_derived, available_engineered, available_lithology = detect_available_features(df)
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Facies column: {facies_col}")
    print(f"\nDetected features ({len(available_features) + len(available_lithology)}):")
    print(f"  Raw logs: {available_raw_logs}")
    print(f"  Derived features: {available_derived}")
    if available_engineered:
        print(f"  Engineered features: {available_engineered}")
        # Show which specific engineered features are available
        feature_descriptions = {
            'GR_Slope': 'GR derivative (coarsening/fining trends)',
            'GR_Serration': 'GR variability (heterogeneity)',
            'RelPos': 'Position in GR cycle',
            'OppositionIndex': 'GR-PHIF correlation',
            'NTG_Slope': 'Net-to-gross trend',
            'AI_Slope': 'Acoustic impedance trend',
            'BaseSharpness': 'Sharpness at cycle base'
        }
        for feat in available_engineered:
            desc = feature_descriptions.get(feat, '')
            print(f"    - {feat}: {desc}")
    if available_lithology:
        print(f"  Lithology features: {available_lithology}")
    
    print(f"\nUnique wells: {df['Well'].nunique()}")
    print(f"Unique facies: {df[facies_col].nunique()}")
    
    # Show wells
    wells = sorted(df['Well'].unique())
    print(f"\nAvailable wells ({len(wells)}):")
    for i, well in enumerate(wells):
        sample_count = len(df[df['Well'] == well])
        print(f"  {i+1}. {well} ({sample_count} samples)")
    
    # Show facies distribution
    print(f"\nFacies distribution:")
    facies_dist = df[facies_col].value_counts().sort_index()
    for facies_num, count in facies_dist.items():
        facies_name = FACIES_NAMES.get(facies_num, f'Unknown ({facies_num})')
        print(f"  {facies_num}: {facies_name} - {count} samples ({count/len(df)*100:.1f}%)")
    
    # Show lithology distribution if available
    if available_lithology:
        print(f"\nLithology distribution:")
        for litho_col in available_lithology:
            count = df[litho_col].sum()
            print(f"  {litho_col}: {count} samples ({count/len(df)*100:.1f}%)")
    
    return df, wells, available_features, available_engineered, available_lithology, facies_col, csv_file

def select_features(available_features, available_engineered, available_lithology):
    """Allow user to select which features to use"""
    print("\n=== Feature Selection ===")
    
    # Combine all available features
    all_available = available_features + available_lithology
    
    print("Available features for classification:")
    print("\nLog-based features:")
    counter = 1
    for feature in available_features:
        if feature in RAW_LOGS:
            print(f"  {counter}. {feature} (raw log)")
            counter += 1
        elif feature in DERIVED_FEATURES:
            print(f"  {counter}. {feature} (derived)")
            counter += 1
    
    if available_engineered:
        print("\nEngineered features:")
        for feature in available_engineered:
            print(f"  {counter}. {feature} (engineered)")
            counter += 1
    
    if available_lithology:
        print("\nLithology features (binary):")
        for feature in available_lithology:
            print(f"  {counter}. {feature}")
            counter += 1
    
    print("\nOptions:")
    print("1. Use all available features (including lithology and engineered)")
    print("2. Use all log features only (no lithology, no engineered)")
    print("3. Use log features + engineered (no lithology)")
    print("4. Use log features + lithology (no engineered)")
    print("5. Select specific features")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '2':
        # Only original log features
        selected = [f for f in available_features if f not in ENGINEERED_FEATURES]
        print(f"\nUsing all log features only: {selected}")
        return selected
    elif choice == '3':
        # Log features + engineered (no lithology)
        print(f"\nUsing log and engineered features: {available_features}")
        return available_features
    elif choice == '4':
        # Log features + lithology (no engineered)
        selected_features = [f for f in available_features if f not in ENGINEERED_FEATURES]
        selected = selected_features + available_lithology
        print(f"\nUsing log and lithology features: {selected}")
        return selected
    elif choice == '5':
        print("\nSelect features to use (comma-separated numbers):")
        indices = input("Enter feature numbers: ").strip().split(',')
        selected_features = []
        for idx in indices:
            if idx.strip().isdigit():
                idx_num = int(idx) - 1
                if idx_num < len(all_available):
                    selected_features.append(all_available[idx_num])
        
        # Separate log features and lithology features while maintaining order
        selected_log_features = [f for f in ALL_POSSIBLE_FEATURES if f in selected_features]
        selected_lithology = [f for f in LITHOLOGY_COLUMNS if f in selected_features]
        
        final_features = selected_log_features + selected_lithology
        print(f"\nSelected features: {final_features}")
        return final_features
    else:
        # Default to all features
        print(f"\nUsing all available features: {all_available}")
        return all_available

def select_wells_for_training_testing(wells):
    """Allow user to select wells for training and testing"""
    print("\n=== Well Selection ===")
    print("Choose training/testing configuration:")
    print("1. Leave-One-Well-Out (LOWO) cross-validation")
    print("2. Manual well selection")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '2':
        print("\nSelect wells for TRAINING (comma-separated numbers):")
        train_indices = input("Enter well numbers: ").strip().split(',')
        train_wells = [wells[int(idx)-1] for idx in train_indices if idx.strip().isdigit()]
        
        print("\nSelect wells for TESTING (comma-separated numbers):")
        test_indices = input("Enter well numbers: ").strip().split(',')
        test_wells = [wells[int(idx)-1] for idx in test_indices if idx.strip().isdigit()]
        
        return 'manual', train_wells, test_wells
    
    else:
        # Default to LOWO if invalid choice
        return 'LOWO', None, None

def train_and_evaluate_rf(df, mode, train_wells, test_wells, feature_cols, target_col):
    """Train Random Forest and evaluate based on selected mode"""
    
    if mode == 'LOWO':
        return lowo_cross_validation(df, feature_cols, target_col)
    
    else:  # manual mode
        # Manual well selection
        train_data = df[df['Well'].isin(train_wells)]
        test_data = df[df['Well'].isin(test_wells)]
        
        # FIXED: Use data directly without additional normalization (already normalized in ML_ready files)
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        print(f"\nTraining on wells: {train_wells}")
        print(f"Testing on wells: {test_wells}")
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Calculate accuracies by well
        well_accuracies = {}
        for well in test_wells:
            well_mask = test_data['Well'] == well
            if well_mask.sum() > 0:
                well_acc = accuracy_score(y_test[well_mask], y_pred[well_mask])
                well_accuracies[well] = well_acc
        
        return y_test, y_pred, well_accuracies, [rf.feature_importances_]

def lowo_cross_validation(df, feature_cols, target_col):
    """Perform Leave-One-Well-Out cross-validation"""
    wells = df['Well'].unique()
    
    all_true_labels = []
    all_predictions = []
    well_accuracies = {}
    feature_importances = []
    
    print("\n=== Starting Leave-One-Well-Out Cross-Validation ===")
    for i, test_well in enumerate(wells):
        print(f"\nFold {i+1}/{len(wells)}: Testing on well {test_well}")
        
        # Split data
        train_data = df[df['Well'] != test_well]
        test_data = df[df['Well'] == test_well]
        
        # FIXED: Use data directly without additional normalization (already normalized in ML_ready files)
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Calculate accuracy for this well
        accuracy = accuracy_score(y_test, y_pred)
        well_accuracies[test_well] = accuracy
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Store results
        all_true_labels.extend(y_test)
        all_predictions.extend(y_pred)
        feature_importances.append(rf.feature_importances_)
    
    return np.array(all_true_labels), np.array(all_predictions), well_accuracies, feature_importances

def create_plots(y_true, y_pred, well_accuracies, feature_importances, feature_cols, timestamp, dataset_name):
    """Create all visualization plots"""
    
    # Create output directory
    output_dir = f"RF_Classification_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate overall accuracy first
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # Get unique facies for labeling - include all possible facies from both true and predicted
    facies_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    facies_names_list = [f"{i}: {FACIES_NAMES.get(i, 'Unknown')}" for i in facies_labels]
    
    # 1. Confusion Matrix (with row-normalized colors)
    cm = confusion_matrix(y_true, y_pred, labels=facies_labels)
    
    # Calculate row-wise normalization for colors only
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_colors = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_colors = np.nan_to_num(cm_colors)  # Replace NaN with 0
    
    plt.figure(figsize=(12, 10))
    # Use normalized values for colors but display actual counts
    sns.heatmap(cm_colors, annot=cm, fmt='d', cmap='Blues',
                xticklabels=facies_names_list, yticklabels=facies_names_list,
                vmin=0, vmax=1)
    plt.title(f'Confusion Matrix - Random Forest Classification\n(Colors normalized by true facies totals)\nDataset: {dataset_name}')
    plt.xlabel('Predicted Facies')
    plt.ylabel('True Facies')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Normalized Confusion Matrix (showing proportions)
    # Handle division by zero for classes with no samples
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=facies_names_list, yticklabels=facies_names_list,
                vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix - Random Forest Classification\n(Proportions within each true facies)\nDataset: {dataset_name}')
    plt.xlabel('Predicted Facies')
    plt.ylabel('True Facies')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label('Proportion of True Facies', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Well Accuracy Plot (if multiple wells)
    if len(well_accuracies) > 1:
        plt.figure(figsize=(12, 6))
        wells_list = list(well_accuracies.keys())
        accuracies_list = list(well_accuracies.values())
        bars = plt.bar(range(len(wells_list)), accuracies_list)
        
        # Add overall accuracy line
        plt.axhline(y=overall_accuracy, color='r', linestyle='--', 
                   label=f'Overall Accuracy: {overall_accuracy:.3f}')
        
        plt.xlabel('Well')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Well\nDataset: {dataset_name}')
        plt.xticks(range(len(wells_list)), wells_list, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'well_accuracies.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Feature Importance Plot
    avg_importances = np.mean(feature_importances, axis=0)
    
    plt.figure(figsize=(14, 8))
    positions = np.arange(len(feature_cols))
    bars = plt.bar(positions, avg_importances)
    
    # Color code bars by feature type
    for i, bar in enumerate(bars):
        if feature_cols[i] in RAW_LOGS:
            bar.set_color('blue')  # Raw logs
        elif feature_cols[i] in DERIVED_FEATURES:
            bar.set_color('orange')  # Derived features
        elif feature_cols[i] in ENGINEERED_FEATURES:
            bar.set_color('red')  # Engineered features
        elif feature_cols[i] in LITHOLOGY_COLUMNS:
            bar.set_color('green')  # Lithology features
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Average Feature Importances\nDataset: {dataset_name}')
    plt.xticks(positions, feature_cols, rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Raw Logs'),
                      Patch(facecolor='orange', label='Derived Features'),
                      Patch(facecolor='red', label='Engineered Features'),
                      Patch(facecolor='green', label='Lithology Features')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Per-Class Precision/Recall/F1 Heatmap (from v17)
    target_names = [f"{i}: {FACIES_NAMES.get(i, 'Unknown')}" for i in facies_labels]
    report_dict = classification_report(
        y_true, y_pred,
        labels=facies_labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    report_df = pd.DataFrame(report_dict).T
    if 'accuracy' in report_df.index:
        report_df = report_df.drop(index='accuracy')
    
    columns_order = ['precision', 'recall', 'f1-score', 'support']
    report_df = report_df[columns_order]
    
    # Get only the class rows (not summary rows)
    class_df = report_df.loc[target_names].copy()
    class_df.index = [f"{idx} (n={int(class_df.loc[idx, 'support'])})" for idx in class_df.index]
    
    metrics_df = class_df[['precision', 'recall', 'f1-score']].astype(float)
    
    fig_height = max(6, 0.4 * len(metrics_df) + 2)
    plt.figure(figsize=(10, fig_height))
    ax = sns.heatmap(
        metrics_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={'label': 'Score'}
    )
    plt.title(f'Per-Class Precision / Recall / F1\nDataset: {dataset_name}')
    plt.xlabel('Metric')
    plt.ylabel('Class (support)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Full Classification Report Table (from v17)
    full_df = pd.DataFrame(report_dict).T
    full_df = full_df[columns_order]
    
    # Organize rows: classes first, then summary rows
    summary_rows = [name for name in ['accuracy', 'macro avg', 'weighted avg'] if name in full_df.index]
    class_rows = target_names
    full_df = pd.concat([full_df.loc[class_rows], full_df.loc[summary_rows]], axis=0)
    
    metrics_cols = ['precision', 'recall', 'f1-score']
    
    # Create mask for summary rows (they won't be colored)
    mask = pd.DataFrame(False, index=full_df.index, columns=metrics_cols + ['support'])
    if summary_rows:
        mask.loc[summary_rows, metrics_cols] = True
    mask.loc[:, 'support'] = True
    
    fig_height_full = max(6, 0.45 * len(full_df) + 2)
    plt.figure(figsize=(11.5, fig_height_full))
    ax = sns.heatmap(
        full_df[metrics_cols + ['support']].astype(float),
        annot=False,
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='white',
        mask=mask.values,
        cbar_kws={'label': 'Score'}
    )
    
    # Add text annotations
    for i, idx in enumerate(full_df.index):
        for j, col in enumerate(metrics_cols):
            val = full_df.loc[idx, col]
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha='center', va='center', color='black')
        sup_val = int(full_df.loc[idx, 'support']) if not pd.isna(full_df.loc[idx, 'support']) else 0
        ax.text(len(metrics_cols) + 0.5, i + 0.5, f"{sup_val}", ha='center', va='center', color='black')
    
    ax.set_title(f'Full Classification Report\nDataset: {dataset_name}')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Class / Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_full_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir, overall_accuracy

def save_reports(y_true, y_pred, well_accuracies, feature_importances, feature_cols, output_dir, overall_accuracy, dataset_name):
    """Save classification report and analysis results"""
    
    # Get all unique facies from both true and predicted (same as in create_plots)
    facies_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [f"{i}: {FACIES_NAMES.get(i, 'Unknown')}" for i in facies_labels]
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=facies_labels, target_names=target_names, zero_division=0)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Random Forest Classification Report (FIXED - No Double Normalization)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("NOTE: This version uses ML_ready data AS-IS (already normalized) without additional scaling\n\n")
        f.write(f"Features used: {feature_cols}\n")
        
        # Separate feature types for display
        log_features = [f for f in feature_cols if f not in LITHOLOGY_COLUMNS and f not in ENGINEERED_FEATURES]
        engineered_features = [f for f in feature_cols if f in ENGINEERED_FEATURES]
        litho_features = [f for f in feature_cols if f in LITHOLOGY_COLUMNS]
        
        if log_features:
            f.write(f"  Log-based features ({len(log_features)}): {log_features}\n")
        if engineered_features:
            f.write(f"  Engineered features ({len(engineered_features)}): {engineered_features}\n")
            # Add descriptions
            feature_descriptions = {
                'GR_Slope': 'GR derivative (coarsening/fining)',
                'GR_Serration': 'GR variability (heterogeneity)',
                'RelPos': 'Position in GR cycle',
                'OppositionIndex': 'GR-PHIF correlation',
                'NTG_Slope': 'Net-to-gross trend',
                'AI_Slope': 'Acoustic impedance trend',
                'BaseSharpness': 'Sharpness at cycle base'
            }
            for feat in engineered_features:
                desc = feature_descriptions.get(feat, '')
                if desc:
                    f.write(f"    - {feat}: {desc}\n")
        if litho_features:
            f.write(f"  Lithology features ({len(litho_features)}): {litho_features}\n")
        
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
        
        if len(well_accuracies) > 1:
            f.write("Well-by-Well Accuracies:\n")
            for well, acc in well_accuracies.items():
                f.write(f"  {well}: {acc:.4f}\n")
            f.write("\n")
        
        f.write("Facies Mapping:\n")
        for facies_num in facies_labels:
            f.write(f"  {facies_num}: {FACIES_NAMES.get(facies_num, 'Unknown')}\n")
        f.write("\n" + report)
        
        # Feature importances
        avg_importances = np.mean(feature_importances, axis=0)
        f.write("\n\nFeature Importances:\n")
        
        # Sort by importance but group by type
        feature_importance_pairs = list(zip(feature_cols, avg_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Display by type
        f.write("\nTop Features by Importance:\n")
        for feature, importance in feature_importance_pairs[:10]:  # Top 10
            if feature in ENGINEERED_FEATURES:
                feature_type = "Engineered"
            elif feature in LITHOLOGY_COLUMNS:
                feature_type = "Lithology"
            elif feature in DERIVED_FEATURES:
                feature_type = "Derived"
            else:
                feature_type = "Raw Log"
            f.write(f"  {feature} ({feature_type}): {importance:.4f}\n")
        
        if len(feature_importance_pairs) > 10:
            f.write("\nRemaining Features:\n")
            for feature, importance in feature_importance_pairs[10:]:
                if feature in ENGINEERED_FEATURES:
                    feature_type = "Engineered"
                elif feature in LITHOLOGY_COLUMNS:
                    feature_type = "Lithology"
                elif feature in DERIVED_FEATURES:
                    feature_type = "Derived"
                else:
                    feature_type = "Raw Log"
                f.write(f"  {feature} ({feature_type}): {importance:.4f}\n")
        
        # Per-facies performance
        f.write("\n\nPer-Facies Performance:\n")
        for facies in facies_labels:
            mask = y_true == facies
            if mask.sum() > 0:
                facies_pred = y_pred[mask]
                facies_true = y_true[mask]
                accuracy = accuracy_score(facies_true, facies_pred)
                precision = (facies_pred == facies).sum() / max((y_pred == facies).sum(), 1)
                recall = (facies_pred == facies).sum() / mask.sum()
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                
                f.write(f"\n{FACIES_NAMES.get(facies, f'Facies {facies}')}:\n")
                f.write(f"  Accuracy: {accuracy:.3f}\n")
                f.write(f"  Precision: {precision:.3f}\n")
                f.write(f"  Recall: {recall:.3f}\n")
                f.write(f"  F1-Score: {f1:.3f}\n")
                f.write(f"  Support: {mask.sum()}\n")

def main():
    """Main execution function"""
    
    # Generate timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    result = load_and_explore_data()
    if result[0] is None:
        return
    
    df, wells, available_features, available_engineered, available_lithology, facies_col, dataset_name = result
    
    # Select features
    feature_cols = select_features(available_features, available_engineered, available_lithology)
    
    if not feature_cols:
        print("No features selected. Exiting.")
        return
    
    # Get well selection
    mode, train_wells, test_wells = select_wells_for_training_testing(wells)
    
    # Train and evaluate
    y_true, y_pred, well_accuracies, feature_importances = train_and_evaluate_rf(
        df, mode, train_wells, test_wells, feature_cols, facies_col
    )
    
    # Create plots
    output_dir, overall_accuracy = create_plots(
        y_true, y_pred, well_accuracies, feature_importances, feature_cols, timestamp, os.path.basename(dataset_name)
    )
    
    # Save reports
    save_reports(y_true, y_pred, well_accuracies, feature_importances, 
                feature_cols, output_dir, overall_accuracy, os.path.basename(dataset_name))
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}/")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print("\nOutput files created:")
    print(f"  - {output_dir}/confusion_matrix.png")
    print(f"  - {output_dir}/confusion_matrix_normalized.png")
    if len(well_accuracies) > 1:
        print(f"  - {output_dir}/well_accuracies.png")
    print(f"  - {output_dir}/feature_importances.png")
    print(f"  - {output_dir}/classification_report_heatmap.png")
    print(f"  - {output_dir}/classification_report_full_table.png")
    print(f"  - {output_dir}/classification_report.txt")

if __name__ == "__main__":
    main()
