"""
Random Forest Classification with Uncertainty Quantification - Version 2
Enhanced to clearly show model combinations and ensemble structure
Uses ensemble of models trained on different well combinations
Creates facies proportion logs showing prediction uncertainty
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
    print("RANDOM FOREST ENSEMBLE - UNCERTAINTY QUANTIFICATION V2")
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

def visualize_model_combinations(train_combinations, test_wells, available_train_wells):
    """Create a visualization showing which wells are used in each model"""
    
    print("\n" + "="*60)
    print("MODEL COMBINATIONS VISUALIZATION")
    print("="*60)
    
    # Create a binary matrix showing which wells are in each model
    n_models = len(train_combinations)
    n_wells = len(available_train_wells)
    
    combination_matrix = np.zeros((n_models, n_wells))
    
    for i, combo in enumerate(train_combinations):
        for j, well in enumerate(available_train_wells):
            if well in combo:
                combination_matrix[i, j] = 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n_models*0.3)))
    
    # Plot 1: Combination matrix
    well_names = [w.replace('15_9-', '') for w in available_train_wells]
    
    im = ax1.imshow(combination_matrix, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    ax1.set_yticks(range(n_models))
    ax1.set_yticklabels([f'Model {i+1}' for i in range(n_models)])
    ax1.set_xticks(range(n_wells))
    ax1.set_xticklabels(well_names, rotation=45, ha='right')
    ax1.set_xlabel('Training Wells')
    ax1.set_ylabel('Models')
    ax1.set_title(f'{n_models} Model Combinations\n(Blue = Included, White = Excluded)')
    
    # Add grid
    for i in range(n_models + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(n_wells + 1):
        ax1.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # Plot 2: Well usage frequency
    well_usage = combination_matrix.sum(axis=0)
    bars = ax2.bar(range(n_wells), well_usage, color='steelblue')
    ax2.set_xticks(range(n_wells))
    ax2.set_xticklabels(well_names, rotation=45, ha='right')
    ax2.set_ylabel('Number of Models')
    ax2.set_title('Well Usage Frequency Across Models')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, well_usage)):
        ax2.text(i, count + 0.1, f'{int(count)}', ha='center')
    
    # Add test wells information
    test_names = [w.replace('15_9-', '') for w in test_wells]
    fig.text(0.5, 0.02, f'Test Wells: {", ".join(test_names)}', 
             ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig

def select_wells_for_ensemble(df):
    """Select wells for ensemble training and testing - Enhanced version"""
    
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
    print("Enter comma-separated numbers (e.g., 1,2,3,4):")
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
    
    print(f"\n✓ Selected {len(available_train_wells)} wells for training pool")
    print(f"✓ Selected {len(test_wells)} wells for testing")
    
    # Calculate possible combinations
    print("\nPossible model configurations:")
    for k in range(1, len(available_train_wells) + 1):
        n_combinations = len(list(combinations(range(len(available_train_wells)), k)))
        print(f"  {k} wells per model -> {n_combinations} models")
    
    print(f"\nHow many wells to use in each training combination?")
    print(f"(Enter a number between 1 and {len(available_train_wells)})")
    n_wells_per_model = int(input("Wells per model: ").strip())
    
    # Generate all possible combinations
    train_combinations = list(combinations(available_train_wells, n_wells_per_model))
    
    print("\n" + "="*60)
    print(f"ENSEMBLE STRUCTURE: {len(train_combinations)} MODELS")
    print("="*60)
    
    print(f"\n✓ Will train {len(train_combinations)} models")
    print(f"✓ Each using {n_wells_per_model} wells for training")
    print(f"✓ Testing on: {[w.replace('15_9-', '') for w in test_wells]}")
    
    # Show ALL combinations
    print("\n" + "-"*60)
    print("COMPLETE LIST OF MODEL COMBINATIONS:")
    print("-"*60)
    for i, combo in enumerate(train_combinations):
        combo_names = [w.replace('15_9-', '') for w in combo]
        print(f"  Model {i+1:3d}: [{', '.join(combo_names)}]")
    
    # Create visualization
    combo_fig = visualize_model_combinations(train_combinations, test_wells, available_train_wells)
    
    # Ask user to confirm
    print("\n" + "-"*60)
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled. Please restart.")
        return None, None, None, None
    
    return available_train_wells, test_wells, train_combinations, combo_fig

def train_ensemble_models(df, train_combinations, test_wells, features):
    """Train multiple RF models on different well combinations - Enhanced reporting"""
    
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
    print("-"*40)
    
    for i, train_wells in enumerate(train_combinations):
        # Detailed progress for each model
        well_names = [w.replace('15_9-', '') for w in train_wells]
        print(f"Model {i+1:3d}/{len(train_combinations)}: Training on [{', '.join(well_names)}]", end=" ... ")
        
        # Prepare training data for this combination
        train_data = df[df['Well'].isin(train_wells)].copy()
        X_train = train_data[features].values
        y_train = train_data['Equinor facies'].astype(int).values
        
        # Train model
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
            'model_id': i + 1,
            'train_wells': well_names,
            'n_training_samples': len(X_train),
            'accuracy': acc
        })
        
        print(f"Accuracy: {acc:.3f}")
    
    print("-"*40)
    print(f"\n✓ All {len(train_combinations)} models trained successfully")
    
    # Summary statistics
    print("\nMODEL ACCURACY SUMMARY:")
    print(f"  Mean:   {np.mean(model_accuracies):.4f} (±{np.std(model_accuracies):.4f})")
    print(f"  Median: {np.median(model_accuracies):.4f}")
    print(f"  Best:   {np.max(model_accuracies):.4f} (Model {np.argmax(model_accuracies)+1})")
    print(f"  Worst:  {np.min(model_accuracies):.4f} (Model {np.argmin(model_accuracies)+1})")
    
    # Show best and worst models
    best_idx = np.argmax(model_accuracies)
    worst_idx = np.argmin(model_accuracies)
    print(f"\nBest Model  (#{best_idx+1}): {model_info[best_idx]['train_wells']} -> {model_accuracies[best_idx]:.4f}")
    print(f"Worst Model (#{worst_idx+1}): {model_info[worst_idx]['train_wells']} -> {model_accuracies[worst_idx]:.4f}")
    
    # Convert to numpy arrays
    ensemble_predictions = np.array(ensemble_predictions)
    
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
    
    # Add vote count for winning facies
    proportions_df['Max_Votes'] = np.max(facies_proportions * n_models, axis=1).astype(int)
    proportions_df['Vote_Percentage'] = np.max(facies_proportions, axis=1) * 100
    
    # Add uncertainty metrics
    entropy = -np.sum(facies_proportions * np.log(facies_proportions + 1e-10), axis=1)
    proportions_df['Prediction_Entropy'] = entropy
    proportions_df['Prediction_Uncertainty'] = 1 - np.max(facies_proportions, axis=1)
    
    # Calculate ensemble accuracy
    ensemble_acc = accuracy_score(proportions_df['True_Facies'], proportions_df['Ensemble_Prediction'])
    
    print(f"\n✓ Ensemble voting accuracy: {ensemble_acc:.4f}")
    
    # Show voting statistics
    print("\nVOTING STATISTICS:")
    print(f"  Unanimous predictions (100% agreement): {(proportions_df['Vote_Percentage'] == 100).sum()} samples")
    print(f"  High confidence (>80% agreement): {(proportions_df['Vote_Percentage'] > 80).sum()} samples")
    print(f"  Medium confidence (50-80% agreement): {((proportions_df['Vote_Percentage'] >= 50) & (proportions_df['Vote_Percentage'] <= 80)).sum()} samples")
    print(f"  Low confidence (<50% agreement): {(proportions_df['Vote_Percentage'] < 50).sum()} samples")
    
    return proportions_df, all_facies

def create_enhanced_visualizations(proportions_df, all_facies, model_accuracies, model_info, test_wells, combo_fig):
    """Create enhanced visualizations with model combination information"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"RF_Uncertainty_Ensemble_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combination figure
    combo_fig.savefig(os.path.join(output_dir, 'model_combinations.png'), dpi=150, bbox_inches='tight')
    plt.close(combo_fig)
    
    # Create main analysis figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model accuracy distribution with annotations
    ax1 = plt.subplot(3, 3, 1)
    
    # Create histogram
    n, bins, patches = ax1.hist(model_accuracies, bins=15, color='steelblue', 
                                edgecolor='black', alpha=0.7)
    
    # Color code the best and worst
    best_idx = np.argmax(model_accuracies)
    worst_idx = np.argmin(model_accuracies)
    
    # Highlight best and worst in the histogram
    ax1.axvline(model_accuracies[best_idx], color='green', linestyle='--', 
                linewidth=2, label=f'Best: {model_accuracies[best_idx]:.3f}')
    ax1.axvline(model_accuracies[worst_idx], color='red', linestyle='--', 
                linewidth=2, label=f'Worst: {model_accuracies[worst_idx]:.3f}')
    ax1.axvline(np.mean(model_accuracies), color='orange', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(model_accuracies):.3f}')
    
    ax1.set_xlabel('Model Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {len(model_accuracies)} Model Accuracies\n({len(model_info[0]["train_wells"])} wells per model)')
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
    
    # 3. Voting agreement distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(proportions_df['Vote_Percentage'], bins=20, 
             color='green', edgecolor='black', alpha=0.7)
    ax3.axvline(proportions_df['Vote_Percentage'].mean(), color='red', 
                linestyle='--', label=f"Mean: {proportions_df['Vote_Percentage'].mean():.1f}%")
    ax3.set_xlabel('Vote Agreement (%)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Model Agreement Distribution\n({len(model_accuracies)} models voting)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Model ranking
    ax4 = plt.subplot(3, 3, 4)
    sorted_indices = np.argsort(model_accuracies)[::-1]
    sorted_accs = [model_accuracies[i] for i in sorted_indices]
    
    bars = ax4.bar(range(len(sorted_accs)), sorted_accs, color='steelblue')
    bars[0].set_color('green')  # Best
    bars[-1].set_color('red')   # Worst
    
    ax4.set_xlabel('Model Rank')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Models Ranked by Accuracy')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add ensemble line
    ensemble_acc = accuracy_score(proportions_df['True_Facies'], proportions_df['Ensemble_Prediction'])
    ax4.axhline(ensemble_acc, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Ensemble: {ensemble_acc:.3f}')
    ax4.legend()
    
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
    
    # 6. Confusion matrix for ensemble
    ax6 = plt.subplot(3, 3, 6)
    from sklearn.metrics import confusion_matrix
    
    true_facies = proportions_df['True_Facies'].values
    ensemble_pred = proportions_df['Ensemble_Prediction'].values
    
    unique_facies = sorted(np.unique(np.concatenate([true_facies, ensemble_pred])))
    facies_names = [FACIES_NAMES.get(f, f'F{f}') for f in unique_facies]
    
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
    
    # 7. Confidence calibration
    ax7 = plt.subplot(3, 3, 7)
    
    # Bin predictions by confidence
    confidence_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        mask = (proportions_df['Vote_Percentage'] >= confidence_bins[i]*100) & \
               (proportions_df['Vote_Percentage'] < confidence_bins[i+1]*100)
        if mask.sum() > 0:
            acc = accuracy_score(proportions_df[mask]['True_Facies'],
                               proportions_df[mask]['Ensemble_Prediction'])
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1])/2 * 100)
            bin_accuracies.append(acc)
            bin_counts.append(mask.sum())
    
    bars = ax7.bar(bin_centers, bin_accuracies, width=8, color='steelblue', alpha=0.7)
    ax7.plot([0, 100], [0, 1], 'r--', label='Perfect calibration')
    ax7.set_xlabel('Vote Agreement (%)')
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Confidence Calibration')
    ax7.set_xlim([45, 105])
    ax7.set_ylim([0, 1])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Add sample counts
    for bar, count, center in zip(bars, bin_counts, bin_centers):
        ax7.text(center, bar.get_height() + 0.02, f'n={count}', ha='center', fontsize=8)
    
    # 8. Per-facies voting patterns
    ax8 = plt.subplot(3, 3, 8)
    
    facies_vote_means = []
    facies_labels = []
    
    for facies in unique_facies:
        mask = proportions_df['True_Facies'] == facies
        if mask.sum() > 0:
            mean_vote = proportions_df[mask]['Vote_Percentage'].mean()
            facies_vote_means.append(mean_vote)
            facies_labels.append(FACIES_NAMES.get(facies, f'F{facies}'))
    
    bars = ax8.bar(range(len(facies_labels)), facies_vote_means, color='steelblue')
    ax8.set_xticks(range(len(facies_labels)))
    ax8.set_xticklabels(facies_labels, rotation=45, ha='right')
    ax8.set_ylabel('Mean Vote Agreement (%)')
    ax8.set_title('Average Model Agreement by Facies')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Color code based on agreement level
    for bar, vote_mean in zip(bars, facies_vote_means):
        if vote_mean > 80:
            bar.set_color('green')
        elif vote_mean < 60:
            bar.set_color('orange')
    
    # 9. Detailed summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "ENSEMBLE CONFIGURATION\n" + "="*35 + "\n\n"
    summary_text += f"Total Models: {len(model_accuracies)}\n"
    summary_text += f"Wells per Model: {len(model_info[0]['train_wells'])}\n"
    summary_text += f"Test Wells: {[w.replace('15_9-', '') for w in test_wells]}\n"
    summary_text += f"Test Samples: {len(proportions_df)}\n\n"
    
    summary_text += "MODEL PERFORMANCE:\n"
    summary_text += f"  Individual Mean: {np.mean(model_accuracies):.4f}\n"
    summary_text += f"  Individual Std: {np.std(model_accuracies):.4f}\n"
    summary_text += f"  Ensemble Voting: {ensemble_acc:.4f}\n"
    summary_text += f"  Improvement: {(ensemble_acc - np.mean(model_accuracies))*100:.1f}%\n\n"
    
    summary_text += "VOTING STATISTICS:\n"
    summary_text += f"  Mean Agreement: {proportions_df['Vote_Percentage'].mean():.1f}%\n"
    summary_text += f"  Unanimous: {(proportions_df['Vote_Percentage'] == 100).sum()} samples\n"
    summary_text += f"  High Conf (>80%): {(proportions_df['Vote_Percentage'] > 80).sum()} samples\n"
    summary_text += f"  Low Conf (<50%): {(proportions_df['Vote_Percentage'] < 50).sum()} samples\n"
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', family='monospace')
    
    plt.suptitle(f'Ensemble Uncertainty Analysis ({len(model_accuracies)} Models)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed model information
    model_df = pd.DataFrame(model_info)
    model_df.to_csv(os.path.join(output_dir, 'model_details.csv'), index=False)
    
    return output_dir

def save_detailed_results(proportions_df, model_info, output_dir):
    """Save comprehensive results and reports"""
    
    print("\n" + "="*60)
    print("SAVING DETAILED RESULTS")
    print("="*60)
    
    # Save proportions
    proportions_df.to_csv(os.path.join(output_dir, 'facies_proportions.csv'), index=False)
    print(f"✓ Saved facies proportions")
    
    # Create detailed report
    report_file = os.path.join(output_dir, 'ensemble_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ENSEMBLE UNCERTAINTY QUANTIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL COMBINATIONS USED:\n")
        f.write("-"*40 + "\n")
        for info in model_info:
            f.write(f"Model {info['model_id']:3d}: [{', '.join(info['train_wells'])}] "
                   f"-> Accuracy: {info['accuracy']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ENSEMBLE STATISTICS:\n")
        f.write("-"*40 + "\n")
        
        ensemble_acc = accuracy_score(proportions_df['True_Facies'], 
                                     proportions_df['Ensemble_Prediction'])
        
        f.write(f"Number of Models: {len(model_info)}\n")
        f.write(f"Mean Individual Accuracy: {np.mean([m['accuracy'] for m in model_info]):.4f}\n")
        f.write(f"Ensemble Voting Accuracy: {ensemble_acc:.4f}\n")
        f.write(f"Improvement over Mean: {(ensemble_acc - np.mean([m['accuracy'] for m in model_info]))*100:.2f}%\n")
        
        f.write("\nVOTING AGREEMENT:\n")
        f.write(f"  Mean: {proportions_df['Vote_Percentage'].mean():.1f}%\n")
        f.write(f"  Std: {proportions_df['Vote_Percentage'].std():.1f}%\n")
        f.write(f"  Min: {proportions_df['Vote_Percentage'].min():.1f}%\n")
        f.write(f"  Max: {proportions_df['Vote_Percentage'].max():.1f}%\n")
    
    print(f"✓ Saved detailed report")
    print(f"✓ All results saved to: {output_dir}/")

def main():
    """Main execution function"""
    
    # Load data
    df, features = load_data()
    if df is None:
        return
    
    # Select wells and show combinations
    result = select_wells_for_ensemble(df)
    if result[0] is None:
        return
    
    available_train_wells, test_wells, train_combinations, combo_fig = result
    
    # Train ensemble
    (ensemble_predictions, ensemble_probabilities, 
     model_accuracies, model_info, test_data, y_test) = train_ensemble_models(
        df, train_combinations, test_wells, features
    )
    
    # Calculate proportions
    proportions_df, all_facies = calculate_facies_proportions(
        ensemble_predictions, test_data
    )
    
    # Create visualizations
    output_dir = create_enhanced_visualizations(
        proportions_df, all_facies, model_accuracies, model_info, test_wells, combo_fig
    )
    
    # Save results
    save_detailed_results(proportions_df, model_info, output_dir)
    
    print("\n" + "="*60)
    print("PROCESS COMPLETE")
    print("="*60)
    print(f"✓ Trained {len(model_accuracies)} models successfully")
    print(f"✓ Created ensemble predictions with uncertainty")
    print(f"✓ All results saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
