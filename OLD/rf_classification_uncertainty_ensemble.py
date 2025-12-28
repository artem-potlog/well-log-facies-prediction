"""
Random Forest Classification with Uncertainty Quantification
Uses ensemble of models trained on different well combinations
Creates facies proportion logs showing prediction uncertainty
Based on rf_classification_f14_dataset_simple.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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

def load_data():
    """Load dataset and prepare for ensemble training"""
    print("="*80)
    print("RANDOM FOREST ENSEMBLE - UNCERTAINTY QUANTIFICATION")
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

def select_wells_for_ensemble(df):
    """Select wells for ensemble training and testing"""
    
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
    print("ENSEMBLE CONFIGURATION")
    print("="*60)
    
    print("\nSelect wells available for TRAINING (will create multiple combinations):")
    print("Enter comma-separated numbers (e.g., 1,2,3,4,5,6):")
    train_indices = input("Training wells pool: ").strip().split(',')
    available_train_wells = []
    for idx in train_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                available_train_wells.append(wells[well_idx])
    
    print("\nSelect TEST well(s) (all models will predict on these):")
    print("Enter comma-separated numbers:")
    test_indices = input("Test wells: ").strip().split(',')
    test_wells = []
    for idx in test_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                test_wells.append(wells[well_idx])
    
    print("\nHow many wells to use in each training combination?")
    print(f"(Available: {len(available_train_wells)}, recommended: {len(available_train_wells)-1})")
    n_wells_per_model = int(input("Wells per model: ").strip())
    
    # Generate all possible combinations
    train_combinations = list(combinations(available_train_wells, n_wells_per_model))
    
    print(f"\n✓ Will train {len(train_combinations)} models")
    print(f"✓ Each using {n_wells_per_model} wells for training")
    print(f"✓ Testing on: {[w.replace('15_9-', '') for w in test_wells]}")
    
    # Show first few combinations as example
    print("\nExample training combinations (first 5):")
    for i, combo in enumerate(train_combinations[:5]):
        combo_names = [w.replace('15_9-', '') for w in combo]
        print(f"  Model {i+1}: {combo_names}")
    if len(train_combinations) > 5:
        print(f"  ... and {len(train_combinations)-5} more combinations")
    
    return available_train_wells, test_wells, train_combinations

def train_ensemble_models(df, train_combinations, test_wells, features):
    """Train multiple RF models on different well combinations"""
    
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE")
    print("="*60)
    
    # Prepare test data once
    test_data = df[df['Well'].isin(test_wells)].copy()
    X_test = test_data[features].values
    y_test = test_data['Equinor facies'].astype(int).values
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Store all models and their predictions
    ensemble_predictions = []
    ensemble_probabilities = []
    model_accuracies = []
    model_info = []
    
    print(f"\nTraining {len(train_combinations)} models...")
    
    for i, train_wells in enumerate(train_combinations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Training model {i+1}/{len(train_combinations)}...")
        
        # Prepare training data for this combination
        train_data = df[df['Well'].isin(train_wells)].copy()
        X_train = train_data[features].values
        y_train = train_data['Equinor facies'].astype(int).values
        
        # Train model
        rf = RandomForestClassifier(
            n_estimators=100,  # Fewer trees per model since we have many models
            max_depth=20,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42 + i,  # Different seed for each model
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)
        
        # Store results
        ensemble_predictions.append(y_pred)
        ensemble_probabilities.append(y_proba)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        model_accuracies.append(acc)
        
        # Store model info
        model_info.append({
            'model_id': i,
            'train_wells': [w.replace('15_9-', '') for w in train_wells],
            'accuracy': acc
        })
    
    print(f"\n✓ All {len(train_combinations)} models trained")
    print(f"✓ Mean accuracy across models: {np.mean(model_accuracies):.4f} (±{np.std(model_accuracies):.4f})")
    print(f"✓ Best model accuracy: {np.max(model_accuracies):.4f}")
    print(f"✓ Worst model accuracy: {np.min(model_accuracies):.4f}")
    
    # Convert to numpy arrays for easier manipulation
    ensemble_predictions = np.array(ensemble_predictions)  # Shape: (n_models, n_samples)
    
    return ensemble_predictions, ensemble_probabilities, model_accuracies, model_info, test_data, y_test

def calculate_facies_proportions(ensemble_predictions, test_data):
    """Calculate facies proportion logs from ensemble predictions"""
    
    print("\n" + "="*60)
    print("CALCULATING FACIES PROPORTIONS")
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
    proportions_df = pd.DataFrame(facies_proportions, columns=proportion_columns)
    
    # Add metadata
    proportions_df['Well'] = test_data['Well'].values
    proportions_df['Depth'] = test_data['Depth'].values
    proportions_df['True_Facies'] = test_data['Equinor facies'].astype(int).values
    
    # Add most likely facies (ensemble vote)
    proportions_df['Ensemble_Prediction'] = all_facies[np.argmax(facies_proportions, axis=1)]
    
    # Add uncertainty metric (entropy)
    entropy = -np.sum(facies_proportions * np.log(facies_proportions + 1e-10), axis=1)
    proportions_df['Prediction_Entropy'] = entropy
    proportions_df['Prediction_Uncertainty'] = 1 - np.max(facies_proportions, axis=1)
    
    # Calculate ensemble accuracy
    ensemble_acc = accuracy_score(proportions_df['True_Facies'], proportions_df['Ensemble_Prediction'])
    print(f"\n✓ Ensemble voting accuracy: {ensemble_acc:.4f}")
    
    return proportions_df, all_facies

def create_uncertainty_visualizations(proportions_df, all_facies, model_accuracies, model_info, test_wells):
    """Create comprehensive uncertainty visualizations"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"RF_Uncertainty_Ensemble_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model accuracy distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(model_accuracies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(model_accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(model_accuracies):.3f}')
    ax1.set_xlabel('Model Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {len(model_accuracies)} Model Accuracies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(proportions_df['Prediction_Uncertainty'], bins=30, 
             color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(proportions_df['Prediction_Uncertainty'].mean(), color='red', 
                linestyle='--', label=f"Mean: {proportions_df['Prediction_Uncertainty'].mean():.3f}")
    ax2.set_xlabel('Prediction Uncertainty (1 - max proportion)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Prediction Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ensemble vs Individual Model Accuracy
    ax3 = plt.subplot(3, 3, 3)
    ensemble_acc = accuracy_score(proportions_df['True_Facies'], proportions_df['Ensemble_Prediction'])
    
    # Create comparison
    accuracies = model_accuracies + [ensemble_acc]
    labels = ['Individual\nModels'] * len(model_accuracies) + ['Ensemble\nVoting']
    colors = ['steelblue'] * len(model_accuracies) + ['darkgreen']
    
    positions = list(range(len(model_accuracies))) + [len(model_accuracies) + 1]
    ax3.scatter(positions[:-1], accuracies[:-1], color='steelblue', alpha=0.5, s=30)
    ax3.scatter(positions[-1], accuracies[-1], color='darkgreen', s=200, marker='*', 
               label=f'Ensemble: {ensemble_acc:.3f}')
    ax3.axhline(np.mean(model_accuracies), color='red', linestyle='--', alpha=0.5,
               label=f'Mean Individual: {np.mean(model_accuracies):.3f}')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Individual Models vs Ensemble Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Facies proportion heatmap (for first well if multiple)
    ax4 = plt.subplot(3, 3, 4)
    first_well = test_wells[0]
    well_data = proportions_df[proportions_df['Well'] == first_well].copy()
    
    if len(well_data) > 0:
        # Sample if too many points
        if len(well_data) > 100:
            sample_idx = np.linspace(0, len(well_data)-1, 100, dtype=int)
            well_data = well_data.iloc[sample_idx]
        
        # Create proportion matrix for heatmap
        proportion_cols = [col for col in proportions_df.columns if col.startswith('Proportion_Facies_')]
        prop_matrix = well_data[proportion_cols].values.T
        
        im = ax4.imshow(prop_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax4.set_yticks(range(len(proportion_cols)))
        ax4.set_yticklabels([f"Facies {int(f)}" for f in all_facies], fontsize=8)
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Facies')
        ax4.set_title(f'Facies Proportions Along {first_well.replace("15_9-", "")}')
        plt.colorbar(im, ax=ax4, label='Proportion')
    
    # 5. Uncertainty vs Depth profile
    ax5 = plt.subplot(3, 3, 5)
    for well in test_wells:
        well_data = proportions_df[proportions_df['Well'] == well]
        if len(well_data) > 0:
            well_name = well.replace('15_9-', '')
            ax5.scatter(well_data['Prediction_Uncertainty'], well_data['Depth'], 
                       alpha=0.5, s=1, label=well_name)
    ax5.set_xlabel('Prediction Uncertainty')
    ax5.set_ylabel('Depth (m)')
    ax5.set_title('Uncertainty vs Depth')
    ax5.invert_yaxis()
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Confusion matrix for ensemble predictions
    ax6 = plt.subplot(3, 3, 6)
    from sklearn.metrics import confusion_matrix
    
    true_facies = proportions_df['True_Facies'].values
    ensemble_pred = proportions_df['Ensemble_Prediction'].values
    
    # Get unique facies in predictions
    unique_facies = sorted(np.unique(np.concatenate([true_facies, ensemble_pred])))
    facies_names = [FACIES_NAMES.get(f, f'Unknown {f}') for f in unique_facies]
    
    cm = confusion_matrix(true_facies, ensemble_pred, labels=unique_facies)
    
    # Normalize for colors
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=facies_names, yticklabels=facies_names,
                vmin=0, vmax=1)
    ax6.set_title(f'Ensemble Confusion Matrix\nAccuracy: {ensemble_acc:.3f}')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('True')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 7. High vs Low confidence accuracy
    ax7 = plt.subplot(3, 3, 7)
    
    # Split into high and low confidence bins
    uncertainty_threshold = proportions_df['Prediction_Uncertainty'].median()
    high_conf_mask = proportions_df['Prediction_Uncertainty'] < uncertainty_threshold
    low_conf_mask = ~high_conf_mask
    
    high_conf_acc = accuracy_score(
        proportions_df[high_conf_mask]['True_Facies'],
        proportions_df[high_conf_mask]['Ensemble_Prediction']
    )
    low_conf_acc = accuracy_score(
        proportions_df[low_conf_mask]['True_Facies'],
        proportions_df[low_conf_mask]['Ensemble_Prediction']
    )
    
    categories = ['High Confidence\n(Low Uncertainty)', 'Low Confidence\n(High Uncertainty)']
    accuracies = [high_conf_acc, low_conf_acc]
    counts = [high_conf_mask.sum(), low_conf_mask.sum()]
    
    bars = ax7.bar(categories, accuracies, color=['green', 'orange'])
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Accuracy vs Prediction Confidence')
    ax7.set_ylim([0, 1])
    
    # Add counts and values
    for i, (acc, n) in enumerate(zip(accuracies, counts)):
        ax7.text(i, acc + 0.02, f'{acc:.3f}\n(n={n})', ha='center')
    
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Per-facies uncertainty
    ax8 = plt.subplot(3, 3, 8)
    
    facies_uncertainties = []
    facies_labels = []
    
    for facies in unique_facies:
        mask = proportions_df['True_Facies'] == facies
        if mask.sum() > 0:
            mean_unc = proportions_df[mask]['Prediction_Uncertainty'].mean()
            facies_uncertainties.append(mean_unc)
            facies_labels.append(FACIES_NAMES.get(facies, f'F{facies}'))
    
    bars = ax8.bar(range(len(facies_labels)), facies_uncertainties, color='steelblue')
    ax8.set_xticks(range(len(facies_labels)))
    ax8.set_xticklabels(facies_labels, rotation=45, ha='right')
    ax8.set_ylabel('Mean Uncertainty')
    ax8.set_title('Average Prediction Uncertainty by Facies')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "ENSEMBLE UNCERTAINTY QUANTIFICATION\n" + "="*35 + "\n\n"
    summary_text += f"Number of Models: {len(model_accuracies)}\n"
    summary_text += f"Test Wells: {[w.replace('15_9-', '') for w in test_wells]}\n"
    summary_text += f"Test Samples: {len(proportions_df)}\n\n"
    
    summary_text += "ACCURACY METRICS:\n"
    summary_text += f"  Mean Individual: {np.mean(model_accuracies):.4f}\n"
    summary_text += f"  Std Individual: {np.std(model_accuracies):.4f}\n"
    summary_text += f"  Best Individual: {np.max(model_accuracies):.4f}\n"
    summary_text += f"  Worst Individual: {np.min(model_accuracies):.4f}\n"
    summary_text += f"  Ensemble Voting: {ensemble_acc:.4f}\n\n"
    
    summary_text += "UNCERTAINTY METRICS:\n"
    summary_text += f"  Mean Uncertainty: {proportions_df['Prediction_Uncertainty'].mean():.4f}\n"
    summary_text += f"  Std Uncertainty: {proportions_df['Prediction_Uncertainty'].std():.4f}\n"
    summary_text += f"  Max Uncertainty: {proportions_df['Prediction_Uncertainty'].max():.4f}\n\n"
    
    summary_text += "CONFIDENCE ANALYSIS:\n"
    summary_text += f"  High Conf. Accuracy: {high_conf_acc:.4f}\n"
    summary_text += f"  Low Conf. Accuracy: {low_conf_acc:.4f}\n"
    summary_text += f"  Difference: {high_conf_acc - low_conf_acc:.4f}\n"
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', family='monospace')
    
    plt.suptitle('Ensemble Uncertainty Quantification Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_dir

def create_facies_proportion_logs(proportions_df, all_facies, output_dir):
    """Create detailed facies proportion logs for each test well"""
    
    print("\n" + "="*60)
    print("CREATING FACIES PROPORTION LOGS")
    print("="*60)
    
    # Save the full proportion DataFrame
    proportions_file = os.path.join(output_dir, 'facies_proportions.csv')
    proportions_df.to_csv(proportions_file, index=False)
    print(f"✓ Saved full proportions to: {proportions_file}")
    
    # Create a log plot for each test well
    test_wells = proportions_df['Well'].unique()
    
    for well in test_wells:
        well_name = well.replace('15_9-', '')
        well_data = proportions_df[proportions_df['Well'] == well].sort_values('Depth')
        
        if len(well_data) == 0:
            continue
        
        # Create figure for this well
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        
        # Get proportion columns
        prop_cols = [col for col in proportions_df.columns if col.startswith('Proportion_Facies_')]
        
        # 1. Stacked area plot of facies proportions
        ax = axes[0]
        depths = well_data['Depth'].values
        
        # Create stacked data
        bottom = np.zeros(len(depths))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_facies)))
        
        for i, (col, facies) in enumerate(zip(prop_cols, all_facies)):
            props = well_data[col].values
            facies_name = FACIES_NAMES.get(facies, f'F{facies}')
            ax.fill_betweenx(depths, bottom, bottom + props, 
                            alpha=0.7, label=facies_name, color=colors[i])
            bottom += props
        
        ax.set_xlabel('Cumulative Proportion')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Facies Proportions')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty profile
        ax = axes[1]
        ax.plot(well_data['Prediction_Uncertainty'].values, depths, 'r-', linewidth=1)
        ax.fill_betweenx(depths, 0, well_data['Prediction_Uncertainty'].values, 
                         alpha=0.3, color='red')
        ax.set_xlabel('Prediction Uncertainty')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - Uncertainty Profile')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_unc = well_data['Prediction_Uncertainty'].mean()
        ax.axvline(mean_unc, color='blue', linestyle='--', 
                  label=f'Mean: {mean_unc:.3f}')
        ax.legend()
        
        # 3. True vs Ensemble Predicted
        ax = axes[2]
        
        # Plot true facies
        true_facies = well_data['True_Facies'].values
        ensemble_pred = well_data['Ensemble_Prediction'].values
        
        # Create color mapping
        facies_colors = {f: colors[i] for i, f in enumerate(all_facies)}
        
        # Plot as colored strips
        for i in range(len(depths)-1):
            # True facies (left half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0, 0.45,
                           color=facies_colors.get(true_facies[i], 'gray'), alpha=0.7)
            # Predicted facies (right half)
            ax.fill_betweenx([depths[i], depths[i+1]], 0.55, 1,
                           color=facies_colors.get(ensemble_pred[i], 'gray'), alpha=0.7)
        
        ax.set_xlim([0, 1])
        ax.set_xticks([0.225, 0.775])
        ax.set_xticklabels(['True', 'Predicted'])
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'{well_name} - True vs Ensemble Prediction')
        ax.invert_yaxis()
        
        # Calculate accuracy for this well
        well_acc = accuracy_score(true_facies, ensemble_pred)
        ax.text(0.5, 0.02, f'Accuracy: {well_acc:.3f}', 
               transform=ax.transAxes, ha='center', fontweight='bold')
        
        plt.suptitle(f'Facies Proportion Log - Well {well_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        fig_name = f'facies_proportion_log_{well_name}.png'
        plt.savefig(os.path.join(output_dir, fig_name), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created proportion log for {well_name}")
    
    # Create summary report
    report_file = os.path.join(output_dir, 'uncertainty_report.txt')
    with open(report_file, 'w') as f:
        f.write("ENSEMBLE UNCERTAINTY QUANTIFICATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("ENSEMBLE CONFIGURATION\n")
        f.write("-"*30 + "\n")
        f.write(f"Number of models: {len(proportions_df.columns[proportions_df.columns.str.startswith('Proportion')])}\n")
        f.write(f"Test wells: {list(test_wells)}\n")
        f.write(f"Total test samples: {len(proportions_df)}\n\n")
        
        f.write("PER-WELL RESULTS\n")
        f.write("-"*30 + "\n")
        for well in test_wells:
            well_name = well.replace('15_9-', '')
            well_data = proportions_df[proportions_df['Well'] == well]
            if len(well_data) > 0:
                well_acc = accuracy_score(well_data['True_Facies'], well_data['Ensemble_Prediction'])
                mean_unc = well_data['Prediction_Uncertainty'].mean()
                f.write(f"\n{well_name}:\n")
                f.write(f"  Samples: {len(well_data)}\n")
                f.write(f"  Ensemble Accuracy: {well_acc:.4f}\n")
                f.write(f"  Mean Uncertainty: {mean_unc:.4f}\n")
                f.write(f"  Min Uncertainty: {well_data['Prediction_Uncertainty'].min():.4f}\n")
                f.write(f"  Max Uncertainty: {well_data['Prediction_Uncertainty'].max():.4f}\n")
        
        f.write("\n\nFACIES-WISE UNCERTAINTY\n")
        f.write("-"*30 + "\n")
        for facies in all_facies:
            mask = proportions_df['True_Facies'] == facies
            if mask.sum() > 0:
                facies_name = FACIES_NAMES.get(facies, f'Unknown {facies}')
                mean_unc = proportions_df[mask]['Prediction_Uncertainty'].mean()
                facies_acc = accuracy_score(
                    proportions_df[mask]['True_Facies'],
                    proportions_df[mask]['Ensemble_Prediction']
                )
                f.write(f"\n{facies_name} (Facies {int(facies)}):\n")
                f.write(f"  Samples: {mask.sum()}\n")
                f.write(f"  Accuracy: {facies_acc:.4f}\n")
                f.write(f"  Mean Uncertainty: {mean_unc:.4f}\n")
    
    print(f"✓ Saved uncertainty report to: {report_file}")
    
def main():
    """Main execution function"""
    
    # Load data
    df, features = load_data()
    if df is None:
        return
    
    # Select wells for ensemble
    available_train_wells, test_wells, train_combinations = select_wells_for_ensemble(df)
    
    if not train_combinations or not test_wells:
        print("Error: Invalid well selection")
        return
    
    # Train ensemble of models
    (ensemble_predictions, ensemble_probabilities, 
     model_accuracies, model_info, test_data, y_test) = train_ensemble_models(
        df, train_combinations, test_wells, features
    )
    
    # Calculate facies proportions
    proportions_df, all_facies = calculate_facies_proportions(
        ensemble_predictions, test_data
    )
    
    # Create visualizations
    output_dir = create_uncertainty_visualizations(
        proportions_df, all_facies, model_accuracies, model_info, test_wells
    )
    
    # Create detailed facies proportion logs
    create_facies_proportion_logs(proportions_df, all_facies, output_dir)
    
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    print(f"✓ Trained {len(model_accuracies)} models")
    print(f"✓ Created facies proportion logs")
    print(f"✓ Quantified prediction uncertainty")
    print(f"✓ Results saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
