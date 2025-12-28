"""
Random Forest Classification for Facies Prediction
Streamlined version for ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv
Uses all features automatically (logs, derived, engineered, lithology)
Only selection: training/testing wells
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

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

# Define all features to use (excluding confidence columns which are not features)
ALL_FEATURES = ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH',
                'GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                'NTG_Slope', 'AI_Slope', 'BaseSharpness',
                'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4']

# Feature categories for visualization
RAW_LOGS = ['GR', 'RT', 'RD', 'NPHI', 'RHOB', 'DT']
DERIVED_FEATURES = ['VSH', 'PHIF', 'SW', 'KLOGH']
ENGINEERED_FEATURES = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                       'NTG_Slope', 'AI_Slope', 'BaseSharpness']

def load_data_and_select_wells():
    """Load dataset and allow well selection"""
    print("="*80)
    print("RANDOM FOREST CLASSIFICATION - F14 DATASET")
    print("="*80)
    
    # Load the specific dataset
    dataset_path = 'ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv'
    print(f"\nLoading dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{dataset_path}' not found!")
        print("Please ensure the file exists in the current directory.")
        return None, None, None
    
    print(f"Dataset shape: {df.shape}")
    
    # Verify all required features are present
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Remove missing features from the list
        features_to_use = [f for f in ALL_FEATURES if f in df.columns]
    else:
        features_to_use = ALL_FEATURES
    
    print(f"\nUsing {len(features_to_use)} features:")
    print(f"  - {len(RAW_LOGS)} raw logs")
    print(f"  - {len(DERIVED_FEATURES)} derived features")  
    print(f"  - {len(ENGINEERED_FEATURES)} engineered features")
    print(f"  - 5 lithology features")
    
    # Filter to only samples with facies labels
    df_with_facies = df[df['Equinor facies'].notna()].copy()
    print(f"\nTotal samples with facies labels: {len(df_with_facies)} / {len(df)} ({len(df_with_facies)/len(df)*100:.1f}%)")
    
    # Show available wells
    wells = sorted(df_with_facies['Well'].unique())
    print(f"\nAvailable wells with facies labels ({len(wells)}):")
    for i, well in enumerate(wells):
        sample_count = len(df_with_facies[df_with_facies['Well'] == well])
        well_name = well.replace('15_9-', '')
        
        # Highlight F-14 if present
        if 'F-14' in well:
            print(f"  {i+1}. {well_name} ({sample_count} samples) <- F-14 with predicted PHIF/SW")
        else:
            print(f"  {i+1}. {well_name} ({sample_count} samples)")
    
    # Show facies distribution
    print(f"\nFacies distribution in dataset:")
    facies_dist = df_with_facies['Equinor facies'].value_counts().sort_index()
    for facies_num, count in facies_dist.items():
        facies_name = FACIES_NAMES.get(facies_num, f'Unknown ({facies_num})')
        print(f"  {int(facies_num)}: {facies_name} - {count} samples ({count/len(df_with_facies)*100:.1f}%)")
    
    # Well selection
    print("\n" + "="*60)
    print("WELL SELECTION")
    print("="*60)
    
    print("\nSelect wells for TRAINING (comma-separated numbers):")
    train_indices = input("Enter well numbers: ").strip().split(',')
    train_wells = []
    for idx in train_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                train_wells.append(wells[well_idx])
    
    print("\nSelect wells for TESTING (comma-separated numbers):")
    test_indices = input("Enter well numbers: ").strip().split(',')
    test_wells = []
    for idx in test_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                test_wells.append(wells[well_idx])
    
    return df_with_facies, train_wells, test_wells, features_to_use

def train_and_evaluate(df, train_wells, test_wells, feature_cols):
    """Train Random Forest and evaluate performance"""
    
    # Prepare training and testing data
    train_data = df[df['Well'].isin(train_wells)].copy()
    test_data = df[df['Well'].isin(test_wells)].copy()
    
    # Handle any remaining NaN values
    for col in feature_cols:
        if col in train_data.columns:
            median_val = train_data[col].median()
            if pd.isna(median_val):
                median_val = 0
            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)
    
    X_train = train_data[feature_cols].values
    y_train = train_data['Equinor facies'].astype(int).values
    X_test = test_data[feature_cols].values
    y_test = test_data['Equinor facies'].astype(int).values
    
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    print(f"Training wells: {[w.replace('15_9-', '') for w in train_wells]}")
    print(f"Testing wells: {[w.replace('15_9-', '') for w in test_wells]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Note if F-14 is in test set
    f14_in_test = any('F-14' in w for w in test_wells)
    if f14_in_test:
        f14_samples = len(test_data[test_data['Well'].str.contains('F-14')])
        print(f"\nNote: F-14 is in test set with {f14_samples} samples (using predicted PHIF/SW)")
    
    # Train Random Forest
    print("\nTraining model...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    max_proba = np.max(y_proba, axis=1)
    
    # Calculate metrics
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    # Calculate per-well accuracy
    well_results = {}
    for well in test_wells:
        well_mask = test_data['Well'] == well
        if well_mask.sum() > 0:
            well_acc = accuracy_score(y_test[well_mask], y_pred[well_mask])
            well_results[well] = {
                'accuracy': well_acc,
                'samples': well_mask.sum()
            }
            well_name = well.replace('15_9-', '')
            print(f"  {well_name}: {well_acc:.4f} ({well_mask.sum()} samples)")
    
    return rf, y_test, y_pred, max_proba, well_results, test_data, feature_cols

def create_visualizations(rf, y_test, y_pred, max_proba, well_results, test_data, 
                         feature_cols, train_wells, test_wells):
    """Create comprehensive visualization"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"RF_Classification_Results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Get facies labels and names
    facies_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    facies_names_list = [FACIES_NAMES.get(i, f'Unknown {i}') for i in facies_labels]
    
    # 1. Confusion Matrix (top left)
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred, labels=facies_labels)
    
    # Normalize for colors
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=facies_names_list, yticklabels=facies_names_list,
                vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax1.set_title(f'Confusion Matrix\nOverall Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Feature Importance (top middle)
    ax2 = plt.subplot(2, 3, 2)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True).tail(20)
    
    # Color by feature type
    colors = []
    for feat in importance_df['feature']:
        if feat in RAW_LOGS:
            colors.append('steelblue')
        elif feat in DERIVED_FEATURES:
            colors.append('darkgreen')
        elif feat in ENGINEERED_FEATURES:
            colors.append('darkorange')
        elif 'Lithology' in feat:
            colors.append('darkred')
        else:
            colors.append('gray')
    
    ax2.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
    ax2.set_yticks(range(len(importance_df)))
    ax2.set_yticklabels(importance_df['feature'].values, fontsize=8)
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 20 Feature Importances')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Raw Logs'),
        Patch(facecolor='darkgreen', label='Derived'),
        Patch(facecolor='darkorange', label='Engineered'),
        Patch(facecolor='darkred', label='Lithology')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # 3. Well-by-well accuracy (top right)
    ax3 = plt.subplot(2, 3, 3)
    wells = list(well_results.keys())
    well_names = [w.replace('15_9-', '') for w in wells]
    accuracies = [well_results[w]['accuracy'] for w in wells]
    samples = [well_results[w]['samples'] for w in wells]
    
    # Color F-14 differently
    colors = ['orange' if 'F-14' in w else 'steelblue' for w in wells]
    
    bars = ax3.bar(range(len(wells)), accuracies, color=colors)
    ax3.axhline(y=accuracy_score(y_test, y_pred), color='red', linestyle='--', 
                label=f'Overall: {accuracy_score(y_test, y_pred):.3f}')
    ax3.set_xticks(range(len(wells)))
    ax3.set_xticklabels(well_names, rotation=45, ha='right')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy by Well')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add sample counts
    for i, (acc, n) in enumerate(zip(accuracies, samples)):
        ax3.text(i, acc + 0.02, f'n={n}', ha='center', fontsize=8)
    
    # 4. Per-facies accuracy (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    facies_acc = []
    facies_counts = []
    
    for facies in facies_labels:
        mask = y_test == facies
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            facies_acc.append(acc)
            facies_counts.append(mask.sum())
        else:
            facies_acc.append(0)
            facies_counts.append(0)
    
    bars = ax4.bar(range(len(facies_labels)), facies_acc, color='steelblue')
    ax4.set_xticks(range(len(facies_labels)))
    ax4.set_xticklabels(facies_names_list, rotation=45, ha='right')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy by Facies Type')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    # Add counts
    for i, (acc, n) in enumerate(zip(facies_acc, facies_counts)):
        ax4.text(i, acc + 0.02, f'n={n}', ha='center', fontsize=8)
    
    # 5. Prediction confidence distribution (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(max_proba, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(max_proba.mean(), color='red', linestyle='--', 
                label=f'Mean: {max_proba.mean():.3f}')
    ax5.set_xlabel('Prediction Confidence (max probability)')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Prediction Confidence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Generate classification report
    report = classification_report(y_test, y_pred, 
                                  target_names=[FACIES_NAMES.get(i, str(i)) for i in facies_labels],
                                  output_dict=True)
    
    # Create summary text
    summary_text = "CLASSIFICATION SUMMARY\n" + "="*30 + "\n\n"
    summary_text += f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
    summary_text += f"Total Test Samples: {len(y_test)}\n\n"
    
    summary_text += "Training Wells:\n"
    for w in train_wells:
        summary_text += f"  • {w.replace('15_9-', '')}\n"
    
    summary_text += "\nTesting Wells:\n"
    for w in test_wells:
        w_name = w.replace('15_9-', '')
        if 'F-14' in w:
            summary_text += f"  • {w_name} (predicted PHIF/SW)\n"
        else:
            summary_text += f"  • {w_name}\n"
    
    summary_text += f"\nWeighted Metrics:\n"
    summary_text += f"  Precision: {report['weighted avg']['precision']:.3f}\n"
    summary_text += f"  Recall: {report['weighted avg']['recall']:.3f}\n"
    summary_text += f"  F1-Score: {report['weighted avg']['f1-score']:.3f}\n"
    
    # Add F-14 specific info if in test set
    f14_wells = [w for w in test_wells if 'F-14' in w]
    if f14_wells:
        f14_acc = well_results[f14_wells[0]]['accuracy']
        f14_samples = well_results[f14_wells[0]]['samples']
        summary_text += f"\nF-14 Performance:\n"
        summary_text += f"  Accuracy: {f14_acc:.4f}\n"
        summary_text += f"  Samples: {f14_samples}\n"
        summary_text += f"  Note: Using predicted PHIF/SW\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', family='monospace')
    
    # Overall title
    train_names = [w.replace('15_9-', '') for w in train_wells]
    test_names = [w.replace('15_9-', '') for w in test_wells]
    plt.suptitle(f'Random Forest Classification Results\nTrain: {", ".join(train_names)} | Test: {", ".join(test_names)}',
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("RANDOM FOREST CLASSIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv\n")
        f.write(f"Features used: {len(feature_cols)} (all available)\n\n")
        f.write(f"Training Wells: {train_wells}\n")
        f.write(f"Testing Wells: {test_wells}\n\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        
        if f14_wells:
            f.write("F-14 SPECIFIC RESULTS\n")
            f.write("-"*30 + "\n")
            f.write(f"F-14 uses predicted PHIF and SW values\n")
            f.write(f"F-14 Accuracy: {well_results[f14_wells[0]]['accuracy']:.4f}\n")
            f.write(f"F-14 Samples: {well_results[f14_wells[0]]['samples']}\n\n")
        
        f.write("WELL-BY-WELL ACCURACY\n")
        f.write("-"*30 + "\n")
        for well in test_wells:
            w_name = well.replace('15_9-', '')
            acc = well_results[well]['accuracy']
            n = well_results[well]['samples']
            f.write(f"{w_name}: {acc:.4f} ({n} samples)\n")
        
        f.write("\n\nDETAILED CLASSIFICATION REPORT\n")
        f.write("-"*30 + "\n")
        f.write(classification_report(y_test, y_pred, 
                                      target_names=[FACIES_NAMES.get(i, str(i)) for i in facies_labels]))
        
        f.write("\n\nTOP 20 FEATURE IMPORTANCES\n")
        f.write("-"*30 + "\n")
        importance_df_full = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        for _, row in importance_df_full.iterrows():
            feature_type = ""
            if row['feature'] in RAW_LOGS:
                feature_type = "[Raw]"
            elif row['feature'] in DERIVED_FEATURES:
                feature_type = "[Derived]"
            elif row['feature'] in ENGINEERED_FEATURES:
                feature_type = "[Engineered]"
            elif 'Lithology' in row['feature']:
                feature_type = "[Lithology]"
            f.write(f"{row['feature']:25s} {feature_type:12s}: {row['importance']:.4f}\n")
    
    print(f"\nResults saved to: {output_dir}/")
    return output_dir

def main():
    """Main execution"""
    # Load data and select wells
    result = load_data_and_select_wells()
    if result[0] is None:
        return
    
    df, train_wells, test_wells, feature_cols = result
    
    if not train_wells or not test_wells:
        print("\nError: Invalid well selection. Please restart and select valid wells.")
        return
    
    # Train and evaluate
    rf, y_test, y_pred, max_proba, well_results, test_data, feature_cols = train_and_evaluate(
        df, train_wells, test_wells, feature_cols
    )
    
    # Create visualizations
    output_dir = create_visualizations(
        rf, y_test, y_pred, max_proba, well_results, test_data,
        feature_cols, train_wells, test_wells
    )
    
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    print(f"✓ Classification completed successfully")
    print(f"✓ Overall accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✓ Results saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
