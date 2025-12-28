"""
Neural Network for Facies Prediction with Bayesian Uncertainty Quantification
Using Monte Carlo Dropout approach for uncertainty estimation
ENHANCED VERSION: Well-based data splitting to prevent data leakage
Based on: https://www.sciencedirect.com/science/article/pii/S0920410521004770
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_recall_fscore_support)
from sklearn.utils import class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

class MCDropout(layers.Dropout):
    """Monte Carlo Dropout layer - keeps dropout active during inference"""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def load_and_explore_data():
    """Load dataset and explore well distribution"""
    print("="*80)
    print("BAYESIAN NEURAL NETWORK - WELL-BASED SPLITTING")
    print("="*80)
    
    # Load the dataset
    dataset_path = 'ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv'
    print(f"\nLoading dataset: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{dataset_path}' not found!")
        return None
    
    # Filter to only samples with facies labels
    df_with_facies = df[df['Equinor facies'].notna()].copy()
    print(f"Total samples with facies labels: {len(df_with_facies)} / {len(df)}")
    
    # Show well distribution
    wells = sorted(df_with_facies['Well'].unique())
    print(f"\nAvailable wells with facies labels ({len(wells)}):")
    for i, well in enumerate(wells):
        sample_count = len(df_with_facies[df_with_facies['Well'] == well])
        well_name = well.replace('15_9-', '')
        facies_dist = df_with_facies[df_with_facies['Well'] == well]['Equinor facies'].value_counts().sort_index()
        
        if 'F-14' in well:
            print(f"  {i+1}. {well_name} ({sample_count} samples) <- F-14 with predicted PHIF/SW")
        else:
            print(f"  {i+1}. {well_name} ({sample_count} samples)")
        
        # Show facies distribution for this well
        facies_summary = ", ".join([f"F{int(f)}:{int(c)}" for f, c in facies_dist.items()])
        print(f"      Facies: {facies_summary}")
    
    return df_with_facies, wells

def select_wells_for_training():
    """Interactive well selection for training and testing"""
    df, wells = load_and_explore_data()
    if df is None:
        return None, None, None, None
    
    print("\n" + "="*60)
    print("WELL SELECTION FOR TRAINING/TESTING")
    print("="*60)
    
    print("\nSelect wells for TRAINING (comma-separated numbers, e.g., 1,2,3,4):")
    train_indices = input("Training wells: ").strip().split(',')
    train_wells = []
    for idx in train_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                train_wells.append(wells[well_idx])
    
    print("\nSelect wells for VALIDATION (comma-separated numbers):")
    val_indices = input("Validation wells: ").strip().split(',')
    val_wells = []
    for idx in val_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                val_wells.append(wells[well_idx])
    
    print("\nSelect wells for TESTING (comma-separated numbers):")
    test_indices = input("Test wells: ").strip().split(',')
    test_wells = []
    for idx in test_indices:
        if idx.strip().isdigit():
            well_idx = int(idx) - 1
            if 0 <= well_idx < len(wells):
                test_wells.append(wells[well_idx])
    
    # Validate selection
    all_selected = set(train_wells + val_wells + test_wells)
    if len(all_selected) != len(train_wells) + len(val_wells) + len(test_wells):
        print("Error: Wells cannot be used in multiple sets!")
        return None, None, None, None
    
    print(f"\n✓ Training wells: {[w.replace('15_9-', '') for w in train_wells]}")
    print(f"✓ Validation wells: {[w.replace('15_9-', '') for w in val_wells]}")
    print(f"✓ Test wells: {[w.replace('15_9-', '') for w in test_wells]}")
    
    # Show data distribution
    train_data = df[df['Well'].isin(train_wells)]
    val_data = df[df['Well'].isin(val_wells)]
    test_data = df[df['Well'].isin(test_wells)]
    
    print(f"\nData distribution:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Testing: {len(test_data)} samples")
    
    # Show facies distribution in each set
    print(f"\nFacies distribution:")
    print("Training set:")
    train_facies = train_data['Equinor facies'].value_counts().sort_index()
    for facies, count in train_facies.items():
        facies_name = FACIES_NAMES.get(int(facies), f'F{int(facies)}')
        print(f"  {facies_name}: {count}")
    
    print("Validation set:")
    val_facies = val_data['Equinor facies'].value_counts().sort_index()
    for facies, count in val_facies.items():
        facies_name = FACIES_NAMES.get(int(facies), f'F{int(facies)}')
        print(f"  {facies_name}: {count}")
    
    print("Test set:")
    test_facies = test_data['Equinor facies'].value_counts().sort_index()
    for facies, count in test_facies.items():
        facies_name = FACIES_NAMES.get(int(facies), f'F{int(facies)}')
        print(f"  {facies_name}: {count}")
    
    # Confirm selection
    print("\n" + "-"*60)
    confirm = input("Proceed with this well configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Configuration cancelled. Please restart.")
        return None, None, None, None
    
    return df, train_wells, val_wells, test_wells

def prepare_well_based_data(df, train_wells, val_wells, test_wells):
    """Prepare training, validation, and test sets based on well selection"""
    
    print("\n" + "="*60)
    print("PREPARING WELL-BASED DATA SPLITS")
    print("="*60)
    
    # Define feature columns
    base_features = ['GR', 'RT', 'NPHI', 'RHOB', 'RD', 'DT', 'VSH', 'KLOGH', 'PHIF', 'SW']
    engineered_features = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                           'NTG_Slope', 'AI_Slope', 'BaseSharpness']
    lithology_features = ['Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4']
    
    # Combine all features
    feature_columns = base_features + engineered_features + lithology_features
    
    # Get available features
    available_features = [f for f in feature_columns if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Handle missing values
    for col in available_features:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
    
    # Split data by wells
    train_data = df[df['Well'].isin(train_wells)].copy()
    val_data = df[df['Well'].isin(val_wells)].copy()
    test_data = df[df['Well'].isin(test_wells)].copy()
    
    # Prepare features and targets
    X_train = train_data[available_features].values
    y_train = train_data['Equinor facies'].astype(int).values
    
    X_val = val_data[available_features].values
    y_val = val_data['Equinor facies'].astype(int).values
    
    X_test = test_data[available_features].values
    y_test = test_data['Equinor facies'].astype(int).values
    
    print(f"\nFinal data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Get unique classes
    all_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
    n_classes = len(all_classes)
    
    print(f"\nClasses found: {all_classes}")
    print(f"Number of classes: {n_classes}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            available_features, n_classes, train_data, val_data, test_data)

def create_bayesian_nn(input_dim, n_classes, dropout_rate=0.3, hidden_units=[128, 64, 32]):
    """
    Create a neural network with Monte Carlo dropout for uncertainty quantification
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    n_classes : int
        Number of output classes
    dropout_rate : float
        Dropout rate for MC dropout layers
    hidden_units : list
        List of hidden layer sizes
    """
    
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
    ])
    
    # Add hidden layers with MC Dropout
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            kernel_initializer='he_normal',
            name=f'hidden_{i+1}'
        ))
        model.add(layers.BatchNormalization())
        model.add(MCDropout(dropout_rate, name=f'mc_dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))
    
    return model

def mc_predict(model, X, n_samples=100):
    """
    Make predictions with uncertainty using Monte Carlo dropout
    
    Parameters:
    -----------
    model : keras.Model
        Trained model with MC dropout
    X : np.array
        Input features
    n_samples : int
        Number of forward passes for MC sampling
    
    Returns:
    --------
    mean_probs : np.array
        Mean predicted probabilities
    std_probs : np.array
        Standard deviation of predictions (uncertainty)
    all_predictions : np.array
        All sampled predictions
    """
    
    predictions = []
    
    print(f"Running {n_samples} Monte Carlo forward passes...")
    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        pred = model.predict(X, verbose=0)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate mean and std
    mean_probs = np.mean(predictions, axis=0)
    std_probs = np.std(predictions, axis=0)
    
    return mean_probs, std_probs, predictions

def calculate_uncertainty_metrics(std_probs, pred_classes):
    """
    Calculate various uncertainty metrics
    
    Parameters:
    -----------
    std_probs : np.array
        Standard deviation of predicted probabilities
    pred_classes : np.array
        Predicted class labels
    
    Returns:
    --------
    dict : Dictionary containing uncertainty metrics
    """
    
    # Predictive entropy (overall uncertainty)
    predictive_entropy = -np.sum(std_probs * np.log(std_probs + 1e-10), axis=1)
    
    # Maximum uncertainty (uncertainty of predicted class)
    max_uncertainty = np.array([std_probs[i, pred_classes[i]] for i in range(len(pred_classes))])
    
    # Mean standard deviation across all classes
    mean_std = np.mean(std_probs, axis=1)
    
    return {
        'predictive_entropy': predictive_entropy,
        'max_uncertainty': max_uncertainty,
        'mean_std': mean_std
    }

def create_monte_carlo_analysis(all_predictions, y_test, test_data, test_wells, save_path):
    """
    Create analysis of Monte Carlo dropout samples (similar to RF ensemble analysis)
    """
    n_samples, n_test, n_classes = all_predictions.shape
    
    # Calculate voting-like statistics from MC samples
    mc_predictions = np.argmax(all_predictions, axis=2)  # Shape: (n_samples, n_test)
    
    # Create figure for MC analysis
    fig = plt.figure(figsize=(20, 12))
    
    # 1. MC Sample Agreement Distribution (like voting agreement)
    ax1 = plt.subplot(2, 3, 1)
    
    # For each test sample, calculate agreement across MC samples
    agreement_percentages = []
    for i in range(n_test):
        predictions_at_i = mc_predictions[:, i]
        mode_pred = np.argmax(np.bincount(predictions_at_i))
        agreement = np.sum(predictions_at_i == mode_pred) / n_samples * 100
        agreement_percentages.append(agreement)
    
    ax1.hist(agreement_percentages, bins=20, color='green', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(agreement_percentages), color='red', linestyle='--', 
                label=f"Mean: {np.mean(agreement_percentages):.1f}%")
    ax1.set_xlabel('MC Sample Agreement (%)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Monte Carlo Agreement Distribution\n({n_samples} forward passes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Variance Across MC Samples
    ax2 = plt.subplot(2, 3, 2)
    
    # Calculate variance in probability predictions across MC samples
    variance_per_sample = np.var(all_predictions, axis=0).mean(axis=1)
    
    ax2.hist(variance_per_sample, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(variance_per_sample.mean(), color='red', linestyle='--',
                label=f"Mean: {variance_per_sample.mean():.4f}")
    ax2.set_xlabel('Variance in Probability Predictions')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Variance Distribution Across MC Samples')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MC Sample "Accuracy" Distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate accuracy for each MC sample
    mc_accuracies = []
    for i in range(n_samples):
        acc = accuracy_score(y_test, mc_predictions[i])
        mc_accuracies.append(acc)
    
    ax3.hist(mc_accuracies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(mc_accuracies), color='orange', linestyle='--',
                label=f"Mean: {np.mean(mc_accuracies):.3f}")
    ax3.axvline(np.max(mc_accuracies), color='green', linestyle='--',
                label=f"Best: {np.max(mc_accuracies):.3f}")
    ax3.axvline(np.min(mc_accuracies), color='red', linestyle='--',
                label=f"Worst: {np.min(mc_accuracies):.3f}")
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Distribution of {n_samples} MC Sample Accuracies')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Depth Profile with Uncertainty Bands
    ax4 = plt.subplot(2, 3, 4)
    
    # Get mean predictions and confidence intervals
    mean_probs = np.mean(all_predictions, axis=0)
    std_probs = np.std(all_predictions, axis=0)
    predicted_facies = np.argmax(mean_probs, axis=1)
    
    colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for well in test_wells[:1]:  # Show first test well
        well_mask = test_data['Well'] == well
        if well_mask.sum() > 0:
            well_indices = np.where(well_mask)[0]
            depths = test_data.loc[well_mask, 'Depth'].values
            
            # Sort by depth
            sort_idx = np.argsort(depths)
            depths_sorted = depths[sort_idx]
            
            # Plot true facies
            true_facies_well = y_test[well_indices][sort_idx]
            pred_facies_well = predicted_facies[well_indices][sort_idx]
            
            ax4.scatter(true_facies_well, depths_sorted, c='blue', alpha=0.6, 
                       s=10, label='True')
            ax4.scatter(pred_facies_well + 0.2, depths_sorted, c='red', alpha=0.6,
                       s=10, label='Predicted')
            
            well_name = well.replace('15_9-', '')
            ax4.set_title(f'Facies Profile - Well {well_name}')
            break
    
    ax4.set_xlabel('Facies')
    ax4.set_ylabel('Depth (m)')
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Confidence Calibration
    ax5 = plt.subplot(2, 3, 5)
    
    # Bin predictions by confidence (max probability)
    max_probs = np.max(mean_probs, axis=1)
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], predicted_facies[mask])
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1])/2)
            bin_accuracies.append(acc)
            bin_counts.append(mask.sum())
    
    if bin_centers:
        bars = ax5.bar(bin_centers, bin_accuracies, width=0.08, color='steelblue', alpha=0.7)
        ax5.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        
        # Add sample counts
        for bar, count, center in zip(bars, bin_counts, bin_centers):
            if bar.get_height() > 0:
                ax5.text(center, bar.get_height() + 0.02, f'n={count}', 
                        ha='center', fontsize=8)
    
    ax5.set_xlabel('Confidence (Max Probability)')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Confidence Calibration')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1.1])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    ensemble_accuracy = accuracy_score(y_test, predicted_facies)
    
    summary_text = "MONTE CARLO DROPOUT ANALYSIS\n" + "="*35 + "\n\n"
    summary_text += f"MC Forward Passes: {n_samples}\n"
    summary_text += f"Test Samples: {n_test}\n"
    summary_text += f"Number of Classes: {n_classes}\n\n"
    
    summary_text += "MC SAMPLE STATISTICS:\n"
    summary_text += f"  Mean Accuracy: {np.mean(mc_accuracies):.4f}\n"
    summary_text += f"  Std Accuracy: {np.std(mc_accuracies):.4f}\n"
    summary_text += f"  Best Sample: {np.max(mc_accuracies):.4f}\n"
    summary_text += f"  Worst Sample: {np.min(mc_accuracies):.4f}\n\n"
    
    summary_text += "ENSEMBLE RESULTS:\n"
    summary_text += f"  Final Accuracy: {ensemble_accuracy:.4f}\n"
    summary_text += f"  Mean Agreement: {np.mean(agreement_percentages):.1f}%\n"
    summary_text += f"  Mean Variance: {variance_per_sample.mean():.4f}\n\n"
    
    high_agreement = np.array(agreement_percentages) > 80
    summary_text += f"HIGH CONFIDENCE:\n"
    summary_text += f"  Samples >80% agreement: {high_agreement.sum()}\n"
    summary_text += f"  Percentage: {100*high_agreement.sum()/len(agreement_percentages):.1f}%"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', family='monospace')
    
    plt.suptitle('Monte Carlo Dropout Analysis (Similar to Ensemble Analysis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'monte_carlo_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return agreement_percentages

def plot_well_based_analysis(train_data, val_data, test_data, y_test, y_pred, uncertainties, 
                           train_wells, val_wells, test_wells, save_path):
    """
    Create comprehensive analysis plots for well-based splitting
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # 1. Well distribution
    ax = axes[0, 0]
    well_counts = []
    well_labels = []
    colors = []
    
    for wells, label, color in [(train_wells, 'Train', 'blue'), 
                               (val_wells, 'Val', 'orange'), 
                               (test_wells, 'Test', 'red')]:
        for well in wells:
            if label == 'Train':
                count = len(train_data[train_data['Well'] == well])
            elif label == 'Val':
                count = len(val_data[val_data['Well'] == well])
            else:
                count = len(test_data[test_data['Well'] == well])
            
            well_counts.append(count)
            well_labels.append(f"{well.replace('15_9-', '')}\n({label})")
            colors.append(color)
    
    bars = ax.bar(range(len(well_counts)), well_counts, color=colors, alpha=0.7)
    ax.set_xticks(range(len(well_labels)))
    ax.set_xticklabels(well_labels, rotation=45, ha='right')
    ax.set_ylabel('Sample Count')
    ax.set_title('Sample Distribution by Well and Split')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (Test Wells)')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('True')
    
    # 3. Uncertainty distribution
    axes[0, 2].hist(uncertainties['mean_std'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Mean Uncertainty (Std Dev)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Uncertainty Distribution')
    axes[0, 2].axvline(np.mean(uncertainties['mean_std']), color='red', 
                       linestyle='--', label=f"Mean: {np.mean(uncertainties['mean_std']):.3f}")
    axes[0, 2].legend()
    
    # 4. Uncertainty vs Correctness
    correct = (y_test == y_pred).astype(int)
    colors_correct = ['red' if c == 0 else 'green' for c in correct]
    axes[1, 0].scatter(range(len(correct)), uncertainties['mean_std'], 
                       c=colors_correct, alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Uncertainty')
    axes[1, 0].set_title('Uncertainty vs Prediction Correctness\n(Green=Correct, Red=Wrong)')
    
    # 5. Box plot of uncertainty by correctness
    correct_unc = uncertainties['mean_std'][correct == 1]
    incorrect_unc = uncertainties['mean_std'][correct == 0]
    axes[1, 1].boxplot([correct_unc, incorrect_unc], labels=['Correct', 'Incorrect'])
    axes[1, 1].set_ylabel('Uncertainty')
    axes[1, 1].set_title('Uncertainty Distribution by Correctness')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Per-well uncertainty analysis
    ax = axes[1, 2]
    well_uncertainties = []
    well_names = []
    
    for well in test_wells:
        well_mask = test_data['Well'] == well
        if well_mask.sum() > 0:
            well_test_indices = test_data[well_mask].index
            # Map back to test set indices
            test_indices_for_well = []
            for idx in well_test_indices:
                test_pos = list(test_data.index).index(idx)
                test_indices_for_well.append(test_pos)
            
            if test_indices_for_well:
                well_unc = uncertainties['mean_std'][test_indices_for_well]
                well_uncertainties.append(well_unc)
                well_names.append(well.replace('15_9-', ''))
    
    if well_uncertainties:
        ax.boxplot(well_uncertainties, labels=well_names)
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Distribution by Test Well')
        ax.grid(True, alpha=0.3)
    
    # 7. Depth vs Uncertainty for test wells
    ax = axes[2, 0]
    for well in test_wells:
        well_data = test_data[test_data['Well'] == well]
        if len(well_data) > 0:
            well_name = well.replace('15_9-', '')
            # Get uncertainties for this well
            well_test_indices = []
            for idx in well_data.index:
                test_pos = list(test_data.index).index(idx)
                well_test_indices.append(test_pos)
            
            if well_test_indices:
                well_unc = uncertainties['mean_std'][well_test_indices]
                ax.scatter(well_unc, well_data['Depth'], alpha=0.6, s=2, label=well_name)
    
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Uncertainty vs Depth (Test Wells)')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Facies-wise performance
    ax = axes[2, 1]
    unique_facies = np.unique(y_test)
    facies_acc = []
    facies_names = []
    
    for facies in unique_facies:
        mask = y_test == facies
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            facies_acc.append(acc)
            facies_names.append(FACIES_NAMES.get(int(facies), f'F{int(facies)}'))
    
    bars = ax.bar(range(len(facies_acc)), facies_acc, color='steelblue')
    ax.set_xticks(range(len(facies_names)))
    ax.set_xticklabels(facies_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Facies Accuracy (Test Wells)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color code based on performance
    for bar, acc in zip(bars, facies_acc):
        if acc > 0.8:
            bar.set_color('green')
        elif acc < 0.5:
            bar.set_color('red')
    
    # 9. Summary statistics
    ax = axes[2, 2]
    ax.axis('off')
    
    summary_text = "WELL-BASED SPLITTING RESULTS\n" + "="*35 + "\n\n"
    summary_text += f"Training Wells: {len(train_wells)}\n"
    summary_text += f"Validation Wells: {len(val_wells)}\n"
    summary_text += f"Test Wells: {len(test_wells)}\n\n"
    
    summary_text += f"Training Samples: {len(train_data)}\n"
    summary_text += f"Validation Samples: {len(val_data)}\n"
    summary_text += f"Test Samples: {len(test_data)}\n\n"
    
    accuracy = accuracy_score(y_test, y_pred)
    summary_text += f"Test Accuracy: {accuracy:.4f}\n"
    summary_text += f"Mean Uncertainty: {np.mean(uncertainties['mean_std']):.4f}\n"
    summary_text += f"Std Uncertainty: {np.std(uncertainties['mean_std']):.4f}\n\n"
    
    # Uncertainty vs correctness analysis
    correct_predictions = (y_test == y_pred)
    mean_unc_correct = np.mean(uncertainties['mean_std'][correct_predictions])
    mean_unc_incorrect = np.mean(uncertainties['mean_std'][~correct_predictions])
    
    summary_text += f"Uncertainty (Correct): {mean_unc_correct:.4f}\n"
    summary_text += f"Uncertainty (Wrong): {mean_unc_incorrect:.4f}\n"
    summary_text += f"Uncertainty Ratio: {mean_unc_incorrect/mean_unc_correct:.2f}\n"
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace')
    
    plt.suptitle('Bayesian Neural Network - Well-Based Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'well_based_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_facies_proportions_and_profiles(all_predictions, y_test, test_data, test_wells, save_path):
    """
    Create facies proportions and depth profiles similar to RF ensemble
    """
    from sklearn.metrics import accuracy_score
    n_samples, n_test, n_classes = all_predictions.shape
    
    # Calculate facies proportions from MC samples (like ensemble voting)
    mc_predictions = np.argmax(all_predictions, axis=2)  # Shape: (n_samples, n_test)
    
    # Get all possible facies values
    all_facies = np.unique(np.concatenate([y_test, mc_predictions.flatten()]))
    n_facies = len(all_facies)
    
    # Create proportion matrix
    facies_proportions = np.zeros((n_test, n_facies))
    
    # For each test sample, count predictions from all MC samples
    for i in range(n_test):
        predictions_at_i = mc_predictions[:, i]
        for j, facies in enumerate(all_facies):
            count = np.sum(predictions_at_i == facies)
            facies_proportions[i, j] = count / n_samples
    
    # Create DataFrame with proportions
    proportion_columns = [f'Proportion_Facies_{int(f)}' for f in all_facies]
    proportions_df = pd.DataFrame(facies_proportions, columns=proportion_columns, index=test_data.index)
    
    # Add metadata
    proportions_df['Well'] = test_data['Well'].values
    proportions_df['Depth'] = test_data['Depth'].values
    proportions_df['True_Facies'] = y_test
    
    # Add most likely facies (ensemble vote)
    proportions_df['MC_Prediction'] = all_facies[np.argmax(facies_proportions, axis=1)]
    
    # Add vote count for winning facies
    proportions_df['Max_Votes'] = np.max(facies_proportions * n_samples, axis=1).astype(int)
    proportions_df['Vote_Percentage'] = np.max(facies_proportions, axis=1) * 100
    
    # Add uncertainty metrics
    mean_probs = np.mean(all_predictions, axis=0)
    std_probs = np.std(all_predictions, axis=0)
    proportions_df['Prediction_Uncertainty'] = 1 - np.max(facies_proportions, axis=1)
    proportions_df['Mean_StdDev'] = np.mean(std_probs, axis=1)
    
    # Save proportions
    proportions_df.to_csv(os.path.join(save_path, 'facies_proportions_mc.csv'), index=False)
    print(f"✓ Saved facies proportions from {n_samples} MC samples")
    
    # Select first test well for detailed visualization
    test_well = test_wells[0] if test_wells else None
    
    if test_well:
        well_mask = proportions_df['Well'] == test_well
        well_data = proportions_df[well_mask].sort_values('Depth')
        
        if len(well_data) > 0:
            well_name = test_well.replace('15_9-', '')
            
            # ========== FIGURE 1: Main comparison plots (matching RF ensemble) ==========
            fig1 = plt.figure(figsize=(15, 10))
            
            # 1. Facies Proportions as stacked area chart (horizontal, like RF ensemble)
            ax1 = plt.subplot(1, 3, 1)
            
            # Prepare data for stacked area
            depths = well_data['Depth'].values
            
            # Create stacked data with cumulative proportions
            bottom = np.zeros(len(depths))
            colors = plt.cm.Set3(np.linspace(0, 1, n_facies))
            
            # Create stacked horizontal bars using fill_betweenx
            for i, (col, facies) in enumerate(zip(proportion_columns, all_facies)):
                props = well_data[col].values
                facies_name = FACIES_NAMES.get(int(facies), f'F{int(facies)}')
                ax1.fill_betweenx(depths, bottom, bottom + props,
                                alpha=0.7, label=facies_name, color=colors[i])
                bottom += props
            
            ax1.set_xlabel('Cumulative Proportion')
            ax1.set_ylabel('Depth (m)')
            ax1.set_title(f'Facies Proportions - Well {well_name}\n(from {n_samples} MC samples)')
            ax1.invert_yaxis()  # Invert y-axis so shallow depths are at top
            ax1.set_xlim([0, 1])
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2. True vs Predicted Facies (colored strips like RF ensemble)
            ax2 = plt.subplot(1, 3, 2)
            
            # Plot true and predicted facies
            true_facies = well_data['True_Facies'].values
            pred_facies = well_data['MC_Prediction'].values
            
            # Create color mapping for facies
            facies_colors = {f: colors[i] for i, f in enumerate(all_facies)}
            
            # Plot as colored strips
            for i in range(len(depths)-1):
                # True facies (left half)
                ax2.fill_betweenx([depths[i], depths[i+1]], 0, 0.45,
                               color=facies_colors.get(true_facies[i], 'gray'), alpha=0.7)
                # Predicted facies (right half)
                ax2.fill_betweenx([depths[i], depths[i+1]], 0.55, 1,
                               color=facies_colors.get(pred_facies[i], 'gray'), alpha=0.7)
            
            # Add mismatches as markers in the middle
            mismatch_mask = true_facies != pred_facies
            if mismatch_mask.sum() > 0:
                ax2.scatter([0.5] * mismatch_mask.sum(), depths[mismatch_mask], 
                          c='red', s=20, marker='x', label=f'Mismatch ({mismatch_mask.sum()})', zorder=5)
            
            ax2.set_xlim([0, 1])
            ax2.set_xticks([0.225, 0.775])
            ax2.set_xticklabels(['True', 'Predicted'], fontsize=10, fontweight='bold')
            ax2.set_ylabel('Depth (m)')
            ax2.set_title(f'True vs Predicted Facies - Well {well_name}')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Calculate and display accuracy for this well
            well_accuracy = accuracy_score(true_facies, pred_facies)
            ax2.text(0.5, 0.98, f'Accuracy: {well_accuracy:.3f}', 
                    transform=ax2.transAxes, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if mismatch_mask.sum() > 0:
                ax2.legend(loc='lower center', fontsize=8)
            
            # 3. Uncertainty Profile (matching RF ensemble format)
            ax3 = plt.subplot(1, 3, 3)
            
            uncertainty = well_data['Prediction_Uncertainty'].values
            
            # Main uncertainty plot
            ax3.plot(uncertainty, depths, 'r-', linewidth=1)
            ax3.fill_betweenx(depths, 0, uncertainty, alpha=0.3, color='red')
            
            ax3.set_xlabel('Prediction Uncertainty')
            ax3.set_ylabel('Depth (m)')
            ax3.set_title(f'Uncertainty Profile - Well {well_name}')
            ax3.invert_yaxis()
            ax3.set_xlim([0, 1])
            ax3.grid(True, alpha=0.3)
            
            # Add mean line
            mean_unc = uncertainty.mean()
            ax3.axvline(mean_unc, color='blue', linestyle='--',
                       label=f'Mean: {mean_unc:.3f}')
            ax3.legend()
            
            # Add overall title and save first figure
            plt.suptitle(f'Facies Proportions and Depth Profiles - Well {well_name}\n(MC Dropout with {n_samples} samples)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'facies_proportions_profiles.png'), dpi=300, bbox_inches='tight')
            plt.show()
            
            # ========== FIGURE 2: Additional Analysis ==========
            fig2 = plt.figure(figsize=(15, 10))
            
            # 1. Vote Agreement Profile
            ax4 = plt.subplot(2, 3, 1)
            
            vote_percentage = well_data['Vote_Percentage'].values
            
            # Create color gradient based on agreement
            colors_agreement = ['red' if v < 50 else 'orange' if v < 80 else 'green' for v in vote_percentage]
            
            ax4.scatter(vote_percentage, depths, c=colors_agreement, s=20, alpha=0.7)
            ax4.plot(vote_percentage, depths, 'k-', linewidth=0.5, alpha=0.3)
            
            # Add threshold lines
            ax4.axvline(50, color='orange', linestyle='--', alpha=0.3, label='50% threshold')
            ax4.axvline(80, color='green', linestyle='--', alpha=0.3, label='80% threshold')
            
            ax4.set_xlabel('MC Sample Agreement (%)')
            ax4.set_ylabel('Depth (m)')
            ax4.set_title(f'Prediction Agreement Profile - Well {well_name}')
            ax4.invert_yaxis()
            ax4.set_xlim([0, 100])
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 2. Facies-specific uncertainty
            ax5 = plt.subplot(2, 3, 2)
            
            # Calculate mean uncertainty for each predicted facies
            facies_uncertainties = {}
            for facies in all_facies:
                mask = pred_facies == facies
                if mask.sum() > 0:
                    facies_uncertainties[facies] = {
                        'mean': np.mean(uncertainty[mask]),
                        'std': np.std(uncertainty[mask]),
                        'count': mask.sum()
                    }
            
            if facies_uncertainties:
                facies_list = list(facies_uncertainties.keys())
                means = [facies_uncertainties[f]['mean'] for f in facies_list]
                stds = [facies_uncertainties[f]['std'] for f in facies_list]
                counts = [facies_uncertainties[f]['count'] for f in facies_list]
                
                bars = ax5.bar(range(len(facies_list)), means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
                
                # Color code by uncertainty level
                for bar, mean_val in zip(bars, means):
                    if mean_val > 0.3:
                        bar.set_color('red')
                    elif mean_val > 0.2:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                ax5.set_xticks(range(len(facies_list)))
                ax5.set_xticklabels([FACIES_NAMES.get(int(f), f'F{int(f)}') for f in facies_list], 
                                   rotation=45, ha='right')
                ax5.set_ylabel('Mean Uncertainty')
                ax5.set_title(f'Uncertainty by Predicted Facies - Well {well_name}')
                ax5.grid(True, alpha=0.3, axis='y')
                
                # Add sample counts
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax5.text(i, bar.get_height() + stds[i] + 0.01, f'n={count}', 
                            ha='center', fontsize=8)
            
            # 3. Summary Statistics and Facies Legend
            ax6 = plt.subplot(2, 3, 3)
            ax6.axis('off')
            
            well_accuracy = accuracy_score(true_facies, pred_facies)
            
            # Add facies color legend
            legend_elements = []
            for i, facies in enumerate(all_facies):
                facies_name = FACIES_NAMES.get(int(facies), f'F{int(facies)}')
                legend_elements.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7, label=facies_name))
            
            legend = ax6.legend(handles=legend_elements, loc='upper left', title='Facies Legend',
                               fontsize=8, title_fontsize=9, ncol=2)
            
            summary_text = f"WELL {well_name} SUMMARY\n" + "="*35 + "\n\n"
            summary_text += f"Total Samples: {len(well_data)}\n"
            summary_text += f"Unique Facies (True): {len(np.unique(true_facies))}\n"
            summary_text += f"Unique Facies (Predicted): {len(np.unique(pred_facies))}\n\n"
            
            summary_text += "PERFORMANCE:\n"
            summary_text += f"  Accuracy: {well_accuracy:.4f}\n"
            summary_text += f"  Mismatches: {mismatch_mask.sum()} ({100*mismatch_mask.sum()/len(well_data):.1f}%)\n\n"
            
            summary_text += "UNCERTAINTY:\n"
            summary_text += f"  Mean Voting Uncertainty: {np.mean(uncertainty):.4f}\n"
            summary_text += f"  Std Voting Uncertainty: {np.std(uncertainty):.4f}\n"
            summary_text += f"  Mean StdDev: {well_data['Mean_StdDev'].mean():.4f}\n\n"
            
            summary_text += "MC VOTING:\n"
            summary_text += f"  Mean Agreement: {np.mean(vote_percentage):.1f}%\n"
            summary_text += f"  Unanimous (100%): {(vote_percentage == 100).sum()} samples\n"
            summary_text += f"  High Conf (>80%): {(vote_percentage > 80).sum()} samples\n"
            summary_text += f"  Low Conf (<50%): {(vote_percentage < 50).sum()} samples\n\n"
            
            summary_text += "DEPTH RANGE:\n"
            summary_text += f"  Min: {depths.min():.1f} m\n"
            summary_text += f"  Max: {depths.max():.1f} m\n"
            summary_text += f"  Range: {depths.max() - depths.min():.1f} m"
            
            ax6.text(0.05, 0.55, summary_text, transform=ax6.transAxes, fontsize=9,
                    verticalalignment='top', family='monospace')
            
            # 4. Per-depth uncertainty heatmap
            ax7 = plt.subplot(2, 3, 4)
            
            # Create bins for depth
            n_depth_bins = min(50, len(depths))
            depth_bins = np.linspace(depths.min(), depths.max(), n_depth_bins)
            depth_labels = [(depth_bins[i] + depth_bins[i+1])/2 for i in range(len(depth_bins)-1)]
            
            # Bin the uncertainties
            binned_uncertainties = []
            for i in range(len(depth_bins)-1):
                mask = (depths >= depth_bins[i]) & (depths < depth_bins[i+1])
                if mask.sum() > 0:
                    binned_uncertainties.append(uncertainty[mask].mean())
                else:
                    binned_uncertainties.append(np.nan)
            
            # Plot as heatmap-style bar chart
            colors_unc = ['green' if not np.isnan(u) and u < 0.2 else 'orange' if not np.isnan(u) and u < 0.4 else 'red' if not np.isnan(u) else 'gray' for u in binned_uncertainties]
            ax7.barh(range(len(binned_uncertainties)), binned_uncertainties, color=colors_unc, alpha=0.7)
            ax7.set_yticks(range(0, len(depth_labels), max(1, len(depth_labels)//10)))
            ax7.set_yticklabels([f'{depth_labels[i]:.1f}' for i in range(0, len(depth_labels), max(1, len(depth_labels)//10))])
            ax7.set_xlabel('Mean Uncertainty')
            ax7.set_ylabel('Depth (m)')
            ax7.set_title('Depth-binned Uncertainty')
            ax7.set_xlim([0, 1])
            ax7.grid(True, alpha=0.3)
            ax7.invert_yaxis()
            
            # 5. Confusion matrix for this well
            ax8 = plt.subplot(2, 3, 5)
            
            from sklearn.metrics import confusion_matrix
            unique_facies = np.unique(np.concatenate([true_facies, pred_facies]))
            cm = confusion_matrix(true_facies, pred_facies, labels=unique_facies)
            
            # Normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)
            
            im = ax8.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            
            # Add text annotations
            for i in range(len(unique_facies)):
                for j in range(len(unique_facies)):
                    text = f'{cm[i, j]}'
                    color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                    ax8.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
            
            ax8.set_xticks(range(len(unique_facies)))
            ax8.set_yticks(range(len(unique_facies)))
            ax8.set_xticklabels([f'F{int(f)}' for f in unique_facies], fontsize=8)
            ax8.set_yticklabels([f'F{int(f)}' for f in unique_facies], fontsize=8)
            ax8.set_xlabel('Predicted')
            ax8.set_ylabel('True')
            ax8.set_title(f'Confusion Matrix (Acc: {well_accuracy:.3f})')
            plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
            
            # 6. MC agreement vs depth scatter
            ax9 = plt.subplot(2, 3, 6)
            
            # Scatter plot with color coding
            scatter = ax9.scatter(vote_percentage, depths, c=vote_percentage, 
                                cmap='RdYlGn', vmin=0, vmax=100, s=10, alpha=0.6)
            ax9.set_xlabel('MC Agreement (%)')
            ax9.set_ylabel('Depth (m)')
            ax9.set_title('Agreement vs Depth Distribution')
            ax9.invert_yaxis()
            ax9.set_xlim([0, 100])
            ax9.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax9, fraction=0.046, pad=0.04)
            
            # Add trend line
            z = np.polyfit(vote_percentage, depths, 1)
            p = np.poly1d(z)
            ax9.plot(vote_percentage, p(vote_percentage), "k--", alpha=0.3, linewidth=1)
            
            # Save second figure
            plt.suptitle(f'Additional Analysis - Well {well_name} (MC Dropout)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'facies_additional_analysis.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    return proportions_df

def create_facies_uncertainty_analysis(mean_probs, std_probs, y_test, test_data, test_wells, save_path):
    """
    Create detailed per-facies uncertainty analysis
    """
    predicted_facies = np.argmax(mean_probs, axis=1)
    unique_facies = np.unique(np.concatenate([y_test, predicted_facies]))
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Per-Facies Uncertainty Distribution
    ax1 = plt.subplot(2, 3, 1)
    
    facies_uncertainties = []
    facies_labels = []
    
    for facies in unique_facies:
        mask = y_test == facies
        if mask.sum() > 0:
            # Get mean uncertainty for this true facies
            mean_std_facies = np.mean(std_probs[mask], axis=1)
            facies_uncertainties.append(mean_std_facies)
            facies_labels.append(FACIES_NAMES.get(int(facies), f'F{int(facies)}'))
    
    if facies_uncertainties:
        bp = ax1.boxplot(facies_uncertainties, labels=facies_labels)
        ax1.set_xlabel('True Facies')
        ax1.set_ylabel('Mean Uncertainty (Std Dev)')
        ax1.set_title('Uncertainty Distribution by True Facies')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix with Confidence Overlay
    ax2 = plt.subplot(2, 3, 2)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predicted_facies, labels=unique_facies)
    
    # Calculate mean confidence for each cell
    confidence_matrix = np.zeros_like(cm, dtype=float)
    for i, true_facies in enumerate(unique_facies):
        for j, pred_facies in enumerate(unique_facies):
            mask = (y_test == true_facies) & (predicted_facies == pred_facies)
            if mask.sum() > 0:
                # Mean confidence is 1 - mean uncertainty
                mean_conf = 1 - np.mean(std_probs[mask].mean(axis=1))
                confidence_matrix[i, j] = mean_conf
    
    # Normalize confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    im = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations with counts and confidence
    for i in range(len(unique_facies)):
        for j in range(len(unique_facies)):
            if cm[i, j] > 0:
                text = f'{cm[i, j]}\n({confidence_matrix[i, j]:.2f})'
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    ax2.set_xticks(range(len(unique_facies)))
    ax2.set_yticks(range(len(unique_facies)))
    ax2.set_xticklabels([FACIES_NAMES.get(int(f), f'F{f}') for f in unique_facies], rotation=45, ha='right')
    ax2.set_yticklabels([FACIES_NAMES.get(int(f), f'F{f}') for f in unique_facies])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix with Mean Confidence\n(count)\\n(confidence)')
    
    # 3. Facies Transition Analysis
    ax3 = plt.subplot(2, 3, 3)
    
    # For each test well, analyze facies transitions
    transition_uncertainties = []
    non_transition_uncertainties = []
    
    for well in test_wells:
        well_mask = test_data['Well'] == well
        if well_mask.sum() > 10:  # Need enough samples
            well_indices = np.where(well_mask)[0]
            depths = test_data.loc[well_mask, 'Depth'].values
            
            # Sort by depth
            sort_idx = np.argsort(depths)
            well_facies = y_test[well_indices][sort_idx]
            well_uncertainties = np.mean(std_probs[well_indices][sort_idx], axis=1)
            
            # Find transitions
            for i in range(1, len(well_facies)):
                if well_facies[i] != well_facies[i-1]:
                    # Transition point
                    transition_uncertainties.extend([well_uncertainties[i-1], well_uncertainties[i]])
                else:
                    # Non-transition
                    non_transition_uncertainties.append(well_uncertainties[i])
    
    if transition_uncertainties and non_transition_uncertainties:
        ax3.boxplot([non_transition_uncertainties, transition_uncertainties], 
                    labels=['Non-transition', 'Transition'])
        ax3.set_ylabel('Uncertainty')
        ax3.set_title('Uncertainty at Facies Transitions')
        ax3.grid(True, alpha=0.3)
    
    # 4. Probability Distribution for Each Facies
    ax4 = plt.subplot(2, 3, 4)
    
    # Show probability distribution for correct vs incorrect predictions
    correct_mask = y_test == predicted_facies
    
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        max_probs_correct = np.max(mean_probs[correct_mask], axis=1)
        max_probs_incorrect = np.max(mean_probs[~correct_mask], axis=1)
        
        ax4.hist(max_probs_correct, bins=30, alpha=0.5, color='green', 
                label=f'Correct (n={correct_mask.sum()})', density=True)
        ax4.hist(max_probs_incorrect, bins=30, alpha=0.5, color='red', 
                label=f'Incorrect (n={(~correct_mask).sum()})', density=True)
        ax4.set_xlabel('Maximum Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Probability Distribution: Correct vs Incorrect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Well-specific Performance
    ax5 = plt.subplot(2, 3, 5)
    
    well_names = []
    well_accuracies = []
    well_uncertainties_mean = []
    
    for well in test_wells:
        well_mask = test_data['Well'] == well
        if well_mask.sum() > 0:
            well_indices = np.where(well_mask)[0]
            well_acc = accuracy_score(y_test[well_indices], predicted_facies[well_indices])
            well_unc = np.mean(std_probs[well_indices].mean(axis=1))
            
            well_names.append(well.replace('15_9-', ''))
            well_accuracies.append(well_acc)
            well_uncertainties_mean.append(well_unc)
    
    if well_names:
        x = np.arange(len(well_names))
        width = 0.35
        
        ax5_twin = ax5.twinx()
        
        bars1 = ax5.bar(x - width/2, well_accuracies, width, label='Accuracy', color='steelblue')
        bars2 = ax5_twin.bar(x + width/2, well_uncertainties_mean, width, label='Mean Uncertainty', color='coral')
        
        ax5.set_xlabel('Test Well')
        ax5.set_ylabel('Accuracy', color='steelblue')
        ax5_twin.set_ylabel('Mean Uncertainty', color='coral')
        ax5.set_title('Per-Well Performance and Uncertainty')
        ax5.set_xticks(x)
        ax5.set_xticklabels(well_names)
        ax5.tick_params(axis='y', labelcolor='steelblue')
        ax5_twin.tick_params(axis='y', labelcolor='coral')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, well_accuracies):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', fontsize=9)
        for bar, val in zip(bars2, well_uncertainties_mean):
            ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{val:.3f}', ha='center', fontsize=9)
    
    # 6. Facies Prediction Success Rate
    ax6 = plt.subplot(2, 3, 6)
    
    facies_accuracies = []
    facies_counts = []
    facies_names_list = []
    
    for facies in unique_facies:
        mask = y_test == facies
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], predicted_facies[mask])
            facies_accuracies.append(acc)
            facies_counts.append(mask.sum())
            facies_names_list.append(FACIES_NAMES.get(int(facies), f'F{int(facies)}'))
    
    if facies_accuracies:
        # Create bubble chart
        colors = plt.cm.RdYlGn(np.array(facies_accuracies))
        scatter = ax6.scatter(range(len(facies_accuracies)), facies_accuracies, 
                            s=np.array(facies_counts)*5, c=colors, alpha=0.6, edgecolors='black')
        
        ax6.set_xticks(range(len(facies_names_list)))
        ax6.set_xticklabels(facies_names_list, rotation=45, ha='right')
        ax6.set_ylabel('Accuracy')
        ax6.set_ylim([0, 1])
        ax6.set_title('Per-Facies Accuracy\n(bubble size = sample count)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add text labels
        for i, (acc, count, name) in enumerate(zip(facies_accuracies, facies_counts, facies_names_list)):
            ax6.text(i, acc + 0.05, f'{acc:.2f}\nn={count}', ha='center', fontsize=8)
    
    plt.suptitle('Detailed Facies Uncertainty Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'facies_uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'NN_Bayesian_WellSplit_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Select wells and prepare data
    result = select_wells_for_training()
    if result[0] is None:
        return
    
    df, train_wells, val_wells, test_wells = result
    
    # Prepare well-based data splits
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     feature_columns, n_classes, train_data, val_data, test_data) = prepare_well_based_data(
        df, train_wells, val_wells, test_wells
    )
    
    print(f"\nNumber of features: {len(feature_columns)}")
    print(f"Number of classes: {n_classes}")
    
    # Features are already scaled in the ML_ready dataset
    print("\nNote: Features are already scaled and ML-ready, no additional scaling needed.")
    
    # Prepare labels for training
    y_train_cat = to_categorical(y_train, n_classes)
    y_val_cat = to_categorical(y_val, n_classes)
    
    # Calculate class weights based on training data
    class_weights_values = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights_values))
    
    # Create model
    print("\n" + "="*60)
    print("Creating Bayesian Neural Network with MC Dropout...")
    
    model = create_bayesian_nn(
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        dropout_rate=0.3,
        hidden_units=[256, 128, 64, 32]
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\n" + "="*60)
    print("Training model...")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Make predictions with uncertainty quantification
    print("\n" + "="*60)
    print("Making predictions with uncertainty quantification...")
    
    mean_probs, std_probs, all_predictions = mc_predict(model, X_test, n_samples=100)
    y_pred = np.argmax(mean_probs, axis=1)
    
    # Calculate uncertainty metrics
    uncertainties = calculate_uncertainty_metrics(std_probs, y_pred)
    
    # Evaluation metrics
    print("\n" + "="*60)
    print("Model Performance on Test Wells:")
    print("-" * 40)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Uncertainty statistics
    print("\n" + "="*60)
    print("Uncertainty Statistics:")
    print("-" * 40)
    print(f"Mean uncertainty (std): {np.mean(uncertainties['mean_std']):.4f}")
    print(f"Std of uncertainty: {np.std(uncertainties['mean_std']):.4f}")
    print(f"Min uncertainty: {np.min(uncertainties['mean_std']):.4f}")
    print(f"Max uncertainty: {np.max(uncertainties['mean_std']):.4f}")
    
    # Analyze uncertainty vs correctness
    correct_predictions = (y_test == y_pred)
    mean_unc_correct = np.mean(uncertainties['mean_std'][correct_predictions])
    mean_unc_incorrect = np.mean(uncertainties['mean_std'][~correct_predictions])
    
    print(f"\nMean uncertainty for correct predictions: {mean_unc_correct:.4f}")
    print(f"Mean uncertainty for incorrect predictions: {mean_unc_incorrect:.4f}")
    print(f"Uncertainty ratio (incorrect/correct): {mean_unc_incorrect/mean_unc_correct:.2f}")
    
    # Create well-based analysis plots
    plot_well_based_analysis(train_data, val_data, test_data, y_test, y_pred, uncertainties,
                            train_wells, val_wells, test_wells, output_dir)
    
    # Create Monte Carlo analysis (similar to RF ensemble analysis)
    print("\nCreating Monte Carlo dropout analysis...")
    agreement_percentages = create_monte_carlo_analysis(
        all_predictions, y_test, test_data, test_wells, output_dir
    )
    
    # Create facies proportions and depth profiles (similar to RF ensemble)
    print("\nCreating facies proportions and depth profiles...")
    proportions_df = create_facies_proportions_and_profiles(
        all_predictions, y_test, test_data, test_wells, output_dir
    )
    
    # Create detailed facies uncertainty analysis
    print("\nCreating detailed facies uncertainty analysis...")
    create_facies_uncertainty_analysis(
        mean_probs, std_probs, y_test, test_data, test_wells, output_dir
    )
    
    # Create prediction results dataframe with uncertainty
    print("\n" + "="*60)
    print("Creating prediction results with uncertainty...")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Well': test_data['Well'].values,
        'Depth': test_data['Depth'].values,
        'True_Facies': y_test,
        'Predicted_Facies': y_pred,
        'Prediction_Confidence': 1 - uncertainties['mean_std'],
        'Uncertainty_StdDev': uncertainties['mean_std'],
        'Predictive_Entropy': uncertainties['predictive_entropy'],
        'Correct': correct_predictions.astype(int)
    })
    
    # Add probability columns for each class
    for i in range(n_classes):
        results_df[f'Prob_Class_{i}'] = mean_probs[:, i]
        results_df[f'Std_Class_{i}'] = std_probs[:, i]
    
    # Save results
    results_file = os.path.join(output_dir, 'predictions_with_uncertainty.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    # High confidence predictions analysis
    high_conf_threshold = 0.8
    high_conf_mask = results_df['Prediction_Confidence'] > high_conf_threshold
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = accuracy_score(
            results_df.loc[high_conf_mask, 'True_Facies'],
            results_df.loc[high_conf_mask, 'Predicted_Facies']
        )
        
        print(f"\nHigh confidence predictions (>={high_conf_threshold}):")
        print(f"  Number of samples: {high_conf_mask.sum()} ({100*high_conf_mask.sum()/len(results_df):.1f}%)")
        print(f"  Accuracy: {high_conf_accuracy:.4f}")
    
    # Low confidence predictions analysis
    low_conf_threshold = 0.6
    low_conf_mask = results_df['Prediction_Confidence'] < low_conf_threshold
    if low_conf_mask.sum() > 0:
        low_conf_accuracy = accuracy_score(
            results_df.loc[low_conf_mask, 'True_Facies'],
            results_df.loc[low_conf_mask, 'Predicted_Facies']
        )
        print(f"\nLow confidence predictions (<{low_conf_threshold}):")
        print(f"  Number of samples: {low_conf_mask.sum()} ({100*low_conf_mask.sum()/len(results_df):.1f}%)")
        print(f"  Accuracy: {low_conf_accuracy:.4f}")
    
    # Save model summary to text file
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("BAYESIAN NEURAL NETWORK WITH WELL-BASED SPLITTING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input features: {len(feature_columns)}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Dropout rate: 0.3\n")
        f.write(f"MC samples for uncertainty: 100\n\n")
        
        f.write("WELL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training wells: {[w.replace('15_9-', '') for w in train_wells]}\n")
        f.write(f"Validation wells: {[w.replace('15_9-', '') for w in val_wells]}\n")
        f.write(f"Test wells: {[w.replace('15_9-', '') for w in test_wells]}\n\n")
        
        f.write("DATA DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training samples: {len(train_data)}\n")
        f.write(f"Validation samples: {len(val_data)}\n")
        f.write(f"Test samples: {len(test_data)}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n\nPERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Mean Uncertainty: {np.mean(uncertainties['mean_std']):.4f}\n")
        f.write(f"Uncertainty for Correct: {mean_unc_correct:.4f}\n")
        f.write(f"Uncertainty for Incorrect: {mean_unc_incorrect:.4f}\n")
        
        f.write("\n\nCLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_test, y_pred))
    
    print("\n" + "="*60)
    print("COMPLETE! All results saved to:", output_dir)
    print("="*60)
    print("\nKey improvements with well-based splitting:")
    print("- No data leakage between training and test sets")
    print("- Realistic evaluation for new well prediction")
    print("- Proper uncertainty quantification for unseen wells")
    print("- Geological validity maintained")

if __name__ == "__main__":
    main()
