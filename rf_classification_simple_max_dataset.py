"""
Simple Random Forest Classification for Depositional Facies Prediction
Designed specifically for ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_AllEngineered_facies_litho.csv
Uses ALL available features from the max dataset
Only manual well selection (no LOWO)
Includes all visualization charts
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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

# Define all features from the max dataset in exact order
ALL_FEATURES = ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH', 
                'GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                'NTG_Slope', 'AI_Slope', 'BaseSharpness']

# Feature categories
RAW_LOGS = ['GR', 'RT', 'RD', 'NPHI', 'RHOB', 'DT']
DERIVED_FEATURES = ['VSH', 'PHIF', 'SW', 'KLOGH']
ENGINEERED_FEATURES = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                      'NTG_Slope', 'AI_Slope', 'BaseSharpness']

# Lithology columns (one-hot encoded)
LITHOLOGY_COLUMNS = ['Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4']

# Expected dataset filename
EXPECTED_DATASET = 'ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_AllEngineered_facies_litho.csv'

def load_and_explore_data():
    """Load the max dataset and show basic information"""
    print(f"Loading dataset: {EXPECTED_DATASET}")
    
    if not os.path.exists(EXPECTED_DATASET):
        print(f"Error: Expected dataset '{EXPECTED_DATASET}' not found!")
        print("Please make sure you have run 'prepare_ml_ready_dataset_gr_vsh_rt_rd_nphi_rhob_dt_phif_sw_klogh_litho_max.py' first.")
        return None, None, None, None
    
    df = pd.read_csv(EXPECTED_DATASET)
    
    # Verify that the expected columns are present
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    missing_lithology = [f for f in LITHOLOGY_COLUMNS if f not in df.columns]
    
    if missing_features:
        print(f"Error: Missing expected features: {missing_features}")
        return None, None, None, None
    
    if missing_lithology:
        print(f"Error: Missing expected lithology columns: {missing_lithology}")
        return None, None, None, None
    
    if 'Equinor facies' not in df.columns:
        print("Error: 'Equinor facies' column not found in dataset")
        return None, None, None, None
    
    # All expected features plus lithology columns
    all_feature_cols = ALL_FEATURES + LITHOLOGY_COLUMNS
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Using ALL features from max dataset ({len(all_feature_cols)} features):")
    print(f"  Raw logs: {RAW_LOGS}")
    print(f"  Derived features: {DERIVED_FEATURES}")
    print(f"  Engineered features: {ENGINEERED_FEATURES}")
    print(f"  Lithology features: {LITHOLOGY_COLUMNS}")
    
    # Show feature descriptions
    print("\nEngineered features descriptions:")
    feature_descriptions = {
        'GR_Slope': 'GR derivative (coarsening/fining trends)',
        'GR_Serration': 'GR variability (heterogeneity)',
        'RelPos': 'Position in GR cycle',
        'OppositionIndex': 'GR-PHIF correlation',
        'NTG_Slope': 'Net-to-gross trend',
        'AI_Slope': 'Acoustic impedance trend',
        'BaseSharpness': 'Sharpness at cycle base'
    }
    for feat in ENGINEERED_FEATURES:
        desc = feature_descriptions.get(feat, '')
        print(f"  - {feat}: {desc}")
    
    print(f"\nUnique wells: {df['Well'].nunique()}")
    print(f"Unique facies: {df['Equinor facies'].nunique()}")
    
    # Show wells
    wells = sorted(df['Well'].unique())
    print(f"\nAvailable wells ({len(wells)}):")
    for i, well in enumerate(wells):
        sample_count = len(df[df['Well'] == well])
        print(f"  {i+1}. {well} ({sample_count} samples)")
    
    # Show facies distribution
    print(f"\nFacies distribution:")
    facies_dist = df['Equinor facies'].value_counts().sort_index()
    for facies_num, count in facies_dist.items():
        facies_name = FACIES_NAMES.get(facies_num, f'Unknown ({facies_num})')
        print(f"  {facies_num}: {facies_name} - {count} samples ({count/len(df)*100:.1f}%)")
    
    # Show lithology distribution
    print(f"\nLithology distribution:")
    for litho_col in LITHOLOGY_COLUMNS:
        count = df[litho_col].sum()
        print(f"  {litho_col}: {count} samples ({count/len(df)*100:.1f}%)")
    
    return df, wells, all_feature_cols, 'Equinor facies'

def select_wells_for_training_testing(wells):
    """Allow user to select wells for training and testing"""
    print("\n=== Manual Well Selection ===")
    
    print("\nSelect wells for TRAINING (comma-separated numbers):")
    train_indices = input("Enter well numbers: ").strip().split(',')
    train_wells = [wells[int(idx)-1] for idx in train_indices if idx.strip().isdigit()]
    
    print("\nSelect wells for TESTING (comma-separated numbers):")
    test_indices = input("Enter well numbers: ").strip().split(',')
    test_wells = [wells[int(idx)-1] for idx in test_indices if idx.strip().isdigit()]
    
    return train_wells, test_wells

def train_and_evaluate_rf(df, train_wells, test_wells, feature_cols, target_col):
    """Train Random Forest and evaluate with manual well selection"""
    
    # Manual well selection
    train_data = df[df['Well'].isin(train_wells)]
    test_data = df[df['Well'].isin(test_wells)]
    
    # Use data directly (already normalized in ML_ready files)
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    print(f"\nTraining on wells: {train_wells}")
    print(f"Testing on wells: {test_wells}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Train Random Forest with optimized parameters
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
    
    return y_test, y_pred, well_accuracies, rf.feature_importances_

def create_plots(y_true, y_pred, well_accuracies, feature_importances, feature_cols, timestamp, train_wells, test_wells):
    """Create all visualization plots"""
    
    # Create output directory
    output_dir = f"RF_Classification_Simple_{timestamp}"
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
    plt.title(f'Confusion Matrix - Random Forest Classification\n(Colors normalized by true facies totals)\nTrain: {train_wells}, Test: {test_wells}')
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
    plt.title(f'Normalized Confusion Matrix - Random Forest Classification\n(Proportions within each true facies)\nTrain: {train_wells}, Test: {test_wells}')
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
        plt.title(f'Accuracy by Test Well\nAll Features from Max Dataset')
        plt.xticks(range(len(wells_list)), wells_list, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'well_accuracies.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Feature Importance Plot
    plt.figure(figsize=(16, 8))
    positions = np.arange(len(feature_cols))
    bars = plt.bar(positions, feature_importances)
    
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
    plt.title(f'Feature Importances - All Features from Max Dataset\nTrain: {train_wells}, Test: {test_wells}')
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
    
    # 5. Per-Class Precision/Recall/F1 Heatmap
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
    plt.title(f'Per-Class Precision / Recall / F1\nAll Features from Max Dataset')
    plt.xlabel('Metric')
    plt.ylabel('Class (support)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Full Classification Report Table
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
    
    ax.set_title(f'Full Classification Report\nAll Features from Max Dataset')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Class / Summary')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_full_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir, overall_accuracy

def save_reports(y_true, y_pred, well_accuracies, feature_importances, feature_cols, output_dir, overall_accuracy, train_wells, test_wells):
    """Save classification report and analysis results"""
    
    # Get all unique facies from both true and predicted
    facies_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = [f"{i}: {FACIES_NAMES.get(i, 'Unknown')}" for i in facies_labels]
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=facies_labels, target_names=target_names, zero_division=0)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Random Forest Classification Report - Simple Max Dataset Version\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {EXPECTED_DATASET}\n")
        f.write(f"Training wells: {train_wells}\n")
        f.write(f"Testing wells: {test_wells}\n")
        f.write("Uses ALL features from max dataset (no feature selection)\n\n")
        f.write(f"Total features used: {len(feature_cols)}\n")
        
        # Separate feature types for display
        log_features = [f for f in feature_cols if f in RAW_LOGS + DERIVED_FEATURES]
        engineered_features = [f for f in feature_cols if f in ENGINEERED_FEATURES]
        litho_features = [f for f in feature_cols if f in LITHOLOGY_COLUMNS]
        
        f.write(f"  Log-based features ({len(log_features)}): {log_features}\n")
        f.write(f"  Engineered features ({len(engineered_features)}): {engineered_features}\n")
        
        # Add descriptions for engineered features
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
        
        f.write(f"  Lithology features ({len(litho_features)}): {litho_features}\n")
        
        f.write(f"\nOverall Accuracy: {overall_accuracy:.4f}\n\n")
        
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
        f.write("\n\nFeature Importances:\n")
        
        # Sort by importance but group by type
        feature_importance_pairs = list(zip(feature_cols, feature_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Display top features
        f.write("\nTop 15 Features by Importance:\n")
        for i, (feature, importance) in enumerate(feature_importance_pairs[:15]):
            if feature in ENGINEERED_FEATURES:
                feature_type = "Engineered"
            elif feature in LITHOLOGY_COLUMNS:
                feature_type = "Lithology"
            elif feature in DERIVED_FEATURES:
                feature_type = "Derived"
            else:
                feature_type = "Raw Log"
            f.write(f"  {i+1:2d}. {feature} ({feature_type}): {importance:.4f}\n")
        
        if len(feature_importance_pairs) > 15:
            f.write("\nRemaining Features:\n")
            for i, (feature, importance) in enumerate(feature_importance_pairs[15:], 16):
                if feature in ENGINEERED_FEATURES:
                    feature_type = "Engineered"
                elif feature in LITHOLOGY_COLUMNS:
                    feature_type = "Lithology"
                elif feature in DERIVED_FEATURES:
                    feature_type = "Derived"
                else:
                    feature_type = "Raw Log"
                f.write(f"  {i:2d}. {feature} ({feature_type}): {importance:.4f}\n")

def main():
    """Main execution function"""
    
    # Generate timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("SIMPLE RANDOM FOREST CLASSIFICATION")
    print("For ML_ready Max Dataset (All Features)")
    print("="*60)
    
    # Load data
    result = load_and_explore_data()
    if result[0] is None:
        return
    
    df, wells, feature_cols, facies_col = result
    
    print(f"\nUsing ALL {len(feature_cols)} features (no selection required)")
    
    # Get well selection
    train_wells, test_wells = select_wells_for_training_testing(wells)
    
    if not train_wells or not test_wells:
        print("Error: Must select both training and testing wells. Exiting.")
        return
    
    # Train and evaluate
    y_true, y_pred, well_accuracies, feature_importances = train_and_evaluate_rf(
        df, train_wells, test_wells, feature_cols, facies_col
    )
    
    # Create plots
    output_dir, overall_accuracy = create_plots(
        y_true, y_pred, well_accuracies, feature_importances, feature_cols, timestamp, train_wells, test_wells
    )
    
    # Save reports
    save_reports(y_true, y_pred, well_accuracies, feature_importances, 
                feature_cols, output_dir, overall_accuracy, train_wells, test_wells)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}/")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Training wells: {train_wells}")
    print(f"Testing wells: {test_wells}")
    print(f"Features used: {len(feature_cols)} (all available)")
    
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
