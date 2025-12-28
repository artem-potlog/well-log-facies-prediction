"""
Random Forest Regression for PHIF and SW Prediction for Well F-14
Version 3: Includes confidence/uncertainty columns and proper numeric formatting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("RANDOM FOREST REGRESSION FOR F-14 PHIF AND SW PREDICTION - V3")
print("With Confidence/Uncertainty Columns and Proper Formatting")
print("="*80)

# Load the dataset
print("\nLoading ML-ready dataset with lithology features...")
df = pd.read_csv('ML_ready_10_base_plus_engineered_F14_special_litho.csv')
print(f"Dataset shape: {df.shape}")

# Check available columns
all_columns = df.columns.tolist()
print(f"\nTotal columns: {len(all_columns)}")

# Identify wells
wells = df['Well'].unique()
print(f"\nWells in dataset: {wells}")
print(f"Number of wells: {len(wells)}")

# Verify F-14 has missing PHIF and SW
f14_mask = df['Well'] == '15_9-F-14'
f14_data = df[f14_mask]
print(f"\nF-14 samples: {len(f14_data)}")
print(f"F-14 PHIF NaN count: {f14_data['PHIF'].isna().sum()}")
print(f"F-14 SW NaN count: {f14_data['SW'].isna().sum()}")

# Check facies and lithology distribution for F-14
print(f"\nF-14 Label Statistics:")
if 'Equinor facies' in f14_data.columns:
    facies_nan_count = f14_data['Equinor facies'].isna().sum()
    print(f"  Facies NaN count: {facies_nan_count} ({facies_nan_count/len(f14_data)*100:.1f}%)")
    if facies_nan_count < len(f14_data):
        print(f"  Facies distribution: {f14_data['Equinor facies'].value_counts().to_dict()}")

if 'Lithology' in f14_data.columns:
    litho_nan_count = f14_data['Lithology'].isna().sum()
    print(f"  Lithology NaN count: {litho_nan_count} ({litho_nan_count/len(f14_data)*100:.1f}%)")
    if litho_nan_count < len(f14_data):
        print(f"  Lithology distribution: {f14_data['Lithology'].value_counts().to_dict()}")

# Get training data (all wells except F-14, and only samples with non-NaN PHIF/SW)
train_mask = (df['Well'] != '15_9-F-14') & df['PHIF'].notna() & df['SW'].notna()
train_data = df[train_mask].copy()
print(f"\nTraining data samples (excluding F-14, with valid PHIF/SW): {len(train_data)}")

# Define features for prediction
exclude_features = ['PHIF', 'SW', 'Well', 'Depth', 'Equinor facies', 'Lithology']
predictor_features = [col for col in all_columns if col not in exclude_features]

# Filter out features that are all NaN in training data
valid_features = []
for feat in predictor_features:
    if train_data[feat].notna().sum() > 0:
        valid_features.append(feat)
    else:
        print(f"  Warning: Feature '{feat}' is all NaN in training data, excluding...")

predictor_features = valid_features

print(f"\nPredictor features ({len(predictor_features)}):")

# Organize features by category
base_features = ['GR', 'RT', 'NPHI', 'RHOB', 'RD', 'DT', 'VSH', 'KLOGH']
engineered_features = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                       'NTG_Slope', 'AI_Slope', 'BaseSharpness']
lithology_features = [f'Lithology_{i}' for i in range(5)]

print("\n  Base features:")
for feat in base_features:
    if feat in predictor_features:
        print(f"    - {feat}")

print("\n  Engineered features:")
for feat in engineered_features:
    if feat in predictor_features:
        print(f"    - {feat}")

print("\n  Lithology features:")
for feat in lithology_features:
    if feat in predictor_features:
        print(f"    - {feat}")

# Handle NaN values in lithology features for training data
print("\n" + "="*60)
print("PREPARING TRAINING DATA")
print("="*60)

# Create a copy for training
train_data_clean = train_data.copy()

# Handle NaN values in lithology features
for litho_feat in lithology_features:
    if litho_feat in train_data_clean.columns:
        nan_count = train_data_clean[litho_feat].isna().sum()
        if nan_count > 0:
            print(f"  Filling {nan_count} NaN values in {litho_feat} with 0 for training")
            train_data_clean[litho_feat] = train_data_clean[litho_feat].fillna(0)

# Prepare training data
X_train = train_data_clean[predictor_features].values
y_train_phif = train_data_clean['PHIF'].values
y_train_sw = train_data_clean['SW'].values

# Prepare F-14 data for prediction
f14_data_clean = f14_data.copy()
for litho_feat in lithology_features:
    if litho_feat in f14_data_clean.columns:
        nan_count = f14_data_clean[litho_feat].isna().sum()
        if nan_count > 0:
            print(f"  F-14: Filling {nan_count} NaN values in {litho_feat} with 0 for prediction")
            f14_data_clean[litho_feat] = f14_data_clean[litho_feat].fillna(0)

X_f14 = f14_data_clean[predictor_features].values

print(f"\nTraining set shape: {X_train.shape}")
print(f"F-14 prediction set shape: {X_f14.shape}")

print("\n" + "="*60)
print("TRAINING RANDOM FOREST MODELS")
print("="*60)

# Function to train and evaluate model
def train_rf_model(X, y, target_name, n_estimators=200, max_depth=20, random_state=42):
    """Train Random Forest model with cross-validation"""
    
    print(f"\n--- Training model for {target_name} ---")
    
    # Initialize model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_r2 = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
    cv_scores_mse = -cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_scores_mae = -cross_val_score(rf_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    print(f"Cross-validation results:")
    print(f"  R² Score: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
    print(f"  RMSE: {np.sqrt(cv_scores_mse.mean()):.4f} (+/- {np.sqrt(cv_scores_mse).std() * 2:.4f})")
    print(f"  MAE: {cv_scores_mae.mean():.4f} (+/- {cv_scores_mae.std() * 2:.4f})")
    
    # Train final model on all training data
    rf_model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': predictor_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 important features for {target_name}:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    return rf_model, feature_importance, cv_scores_r2.mean()

# Train PHIF model
print("\n" + "="*40)
print("PHIF MODEL")
print("="*40)
rf_phif, importance_phif, r2_phif = train_rf_model(X_train, y_train_phif, "PHIF", 
                                                   n_estimators=300, max_depth=25)

# Train SW model
print("\n" + "="*40)
print("SW MODEL")
print("="*40)
rf_sw, importance_sw, r2_sw = train_rf_model(X_train, y_train_sw, "SW", 
                                             n_estimators=300, max_depth=25)

# Make predictions for F-14
print("\n" + "="*60)
print("PREDICTING F-14 VALUES WITH CONFIDENCE")
print("="*60)

# Get predictions
phif_pred = rf_phif.predict(X_f14)
sw_pred = rf_sw.predict(X_f14)

# Calculate prediction uncertainties using all trees
print("\nCalculating prediction uncertainties...")
trees_phif = np.array([tree.predict(X_f14) for tree in rf_phif.estimators_])
trees_sw = np.array([tree.predict(X_f14) for tree in rf_sw.estimators_])

# Calculate standard deviation (uncertainty) for each prediction
phif_std = trees_phif.std(axis=0)
sw_std = trees_sw.std(axis=0)

# Convert uncertainty to confidence (probability-like score)
# Using inverse of coefficient of variation, normalized to 0-1
# Lower std relative to prediction = higher confidence
phif_confidence = 1 / (1 + phif_std / (np.abs(phif_pred) + 1e-6))
sw_confidence = 1 / (1 + sw_std / (np.abs(sw_pred) + 1e-6))

# Normalize confidence to 0-1 range more intuitively
phif_confidence = (phif_confidence - phif_confidence.min()) / (phif_confidence.max() - phif_confidence.min() + 1e-6)
sw_confidence = (sw_confidence - sw_confidence.min()) / (sw_confidence.max() - sw_confidence.min() + 1e-6)

# Scale to percentage
phif_confidence_pct = phif_confidence * 100
sw_confidence_pct = sw_confidence * 100

print(f"\nPHIF predictions for F-14:")
print(f"  Range: [{phif_pred.min():.4f}, {phif_pred.max():.4f}]")
print(f"  Mean: {phif_pred.mean():.4f}")
print(f"  Mean confidence: {phif_confidence_pct.mean():.1f}%")

print(f"\nSW predictions for F-14:")
print(f"  Range: [{sw_pred.min():.4f}, {sw_pred.max():.4f}]")
print(f"  Mean: {sw_pred.mean():.4f}")
print(f"  Mean confidence: {sw_confidence_pct.mean():.1f}%")

# Calculate 95% confidence intervals
phif_lower = phif_pred - 1.96 * phif_std
phif_upper = phif_pred + 1.96 * phif_std
sw_lower = sw_pred - 1.96 * sw_std
sw_upper = sw_pred + 1.96 * sw_std

print(f"\n95% Confidence Intervals:")
print(f"  PHIF: Average width = {(phif_upper - phif_lower).mean():.4f}")
print(f"  SW: Average width = {(sw_upper - sw_lower).mean():.4f}")

# Compare with training data statistics
print(f"\nComparison with training data statistics:")
print(f"PHIF - Training mean: {y_train_phif.mean():.4f}, F-14 predicted mean: {phif_pred.mean():.4f}")
print(f"SW - Training mean: {y_train_sw.mean():.4f}, F-14 predicted mean: {sw_pred.mean():.4f}")

# Create dataset with populated F-14 values and confidence scores
print("\n" + "="*60)
print("CREATING COMPLETE DATASET WITH CONFIDENCE SCORES")
print("="*60)

# Create a copy of the original dataframe
df_complete = df.copy()

# Fill F-14 PHIF and SW values
df_complete.loc[f14_mask, 'PHIF'] = phif_pred
df_complete.loc[f14_mask, 'SW'] = sw_pred

# Add confidence columns (initialize all with NaN first)
df_complete['PHIF_Confidence'] = np.nan
df_complete['SW_Confidence'] = np.nan
df_complete['PHIF_Std'] = np.nan
df_complete['SW_Std'] = np.nan
df_complete['PHIF_Lower_95CI'] = np.nan
df_complete['PHIF_Upper_95CI'] = np.nan
df_complete['SW_Lower_95CI'] = np.nan
df_complete['SW_Upper_95CI'] = np.nan

# Fill confidence values only for F-14
df_complete.loc[f14_mask, 'PHIF_Confidence'] = phif_confidence_pct
df_complete.loc[f14_mask, 'SW_Confidence'] = sw_confidence_pct
df_complete.loc[f14_mask, 'PHIF_Std'] = phif_std
df_complete.loc[f14_mask, 'SW_Std'] = sw_std
df_complete.loc[f14_mask, 'PHIF_Lower_95CI'] = phif_lower
df_complete.loc[f14_mask, 'PHIF_Upper_95CI'] = phif_upper
df_complete.loc[f14_mask, 'SW_Lower_95CI'] = sw_lower
df_complete.loc[f14_mask, 'SW_Upper_95CI'] = sw_upper

# Ensure all numeric columns are properly typed as float64
numeric_columns = df_complete.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df_complete[col] = df_complete[col].astype(np.float64)

# Verify no more NaNs in PHIF and SW
print(f"\nVerification:")
print(f"  Total PHIF NaN count: {df_complete['PHIF'].isna().sum()}")
print(f"  Total SW NaN count: {df_complete['SW'].isna().sum()}")

# Verify facies and lithology are preserved
if 'Equinor facies' in df_complete.columns:
    print(f"  Facies column preserved: Yes")
    facies_counts = df_complete['Equinor facies'].value_counts(dropna=False)
    print(f"  Total facies labels: {facies_counts.sum()}")

if 'Lithology' in df_complete.columns:
    print(f"  Lithology column preserved: Yes")
    litho_counts = df_complete['Lithology'].value_counts(dropna=False)
    print(f"  Total lithology labels: {litho_counts.sum()}")

print(f"\nNew confidence columns added:")
print(f"  - PHIF_Confidence: Confidence score (0-100%)")
print(f"  - SW_Confidence: Confidence score (0-100%)")
print(f"  - PHIF_Std: Standard deviation of predictions")
print(f"  - SW_Std: Standard deviation of predictions")
print(f"  - PHIF_Lower_95CI: Lower 95% confidence interval")
print(f"  - PHIF_Upper_95CI: Upper 95% confidence interval")
print(f"  - SW_Lower_95CI: Lower 95% confidence interval")
print(f"  - SW_Upper_95CI: Upper 95% confidence interval")

# Save the complete dataset with proper formatting
output_filename = 'ML_ready_F14_PHIF_SW_predicted_with_confidence.csv'

# Save with explicit float formatting to prevent apostrophe issues
print(f"\nSaving dataset with proper numeric formatting...")
df_complete.to_csv(output_filename, index=False, float_format='%.8f')

print(f"\nComplete dataset saved as: {output_filename}")
print(f"  Shape: {df_complete.shape}")
print(f"  Columns: {len(df_complete.columns)}")

# Also save as Excel for guaranteed proper formatting
excel_filename = 'ML_ready_F14_PHIF_SW_predicted_with_confidence.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_complete.to_excel(writer, sheet_name='Predictions', index=False)
    
    # Create a summary sheet
    summary_data = {
        'Metric': ['F-14 Samples', 'Mean PHIF', 'Mean SW', 'Mean PHIF Confidence (%)', 
                   'Mean SW Confidence (%)', 'PHIF Model R²', 'SW Model R²'],
        'Value': [len(f14_data), phif_pred.mean(), sw_pred.mean(), 
                  phif_confidence_pct.mean(), sw_confidence_pct.mean(), r2_phif, r2_sw]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"  Excel version saved as: {excel_filename}")

# Create comprehensive visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Figure 1: Feature importance and confidence visualization
fig1, axes = plt.subplots(3, 3, figsize=(18, 14))

# 1. Feature importance for PHIF
ax = axes[0, 0]
top_features_phif = importance_phif.head(10)
colors = []
for feat in top_features_phif['feature']:
    if feat in base_features:
        colors.append('steelblue')
    elif feat in engineered_features:
        colors.append('darkgreen')
    elif feat.startswith('Lithology_'):
        colors.append('darkred')
    else:
        colors.append('gray')

ax.barh(range(len(top_features_phif)), top_features_phif['importance'].values, color=colors)
ax.set_yticks(range(len(top_features_phif)))
ax.set_yticklabels(top_features_phif['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title(f'Top 10 Features for PHIF (R²={r2_phif:.3f})')
ax.invert_yaxis()

# 2. Feature importance for SW
ax = axes[0, 1]
top_features_sw = importance_sw.head(10)
colors = []
for feat in top_features_sw['feature']:
    if feat in base_features:
        colors.append('steelblue')
    elif feat in engineered_features:
        colors.append('darkgreen')
    elif feat.startswith('Lithology_'):
        colors.append('darkred')
    else:
        colors.append('gray')

ax.barh(range(len(top_features_sw)), top_features_sw['importance'].values, color=colors)
ax.set_yticks(range(len(top_features_sw)))
ax.set_yticklabels(top_features_sw['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title(f'Top 10 Features for SW (R²={r2_sw:.3f})')
ax.invert_yaxis()

# 3. Confidence distribution
ax = axes[0, 2]
ax.hist(phif_confidence_pct, bins=30, alpha=0.5, label='PHIF', color='blue', density=True)
ax.hist(sw_confidence_pct, bins=30, alpha=0.5, label='SW', color='red', density=True)
ax.set_xlabel('Confidence (%)')
ax.set_ylabel('Density')
ax.set_title('Prediction Confidence Distribution')
ax.legend()

# 4. PHIF predictions with confidence
ax = axes[1, 0]
f14_depths = f14_data['Depth'].values
scatter = ax.scatter(phif_pred, f14_depths, c=phif_confidence_pct, cmap='RdYlGn', s=1, alpha=0.6)
ax.set_xlabel('PHIF Predicted')
ax.set_ylabel('Depth (m)')
ax.set_title('F-14 PHIF Predictions (colored by confidence)')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='Confidence (%)')

# 5. SW predictions with confidence
ax = axes[1, 1]
scatter = ax.scatter(sw_pred, f14_depths, c=sw_confidence_pct, cmap='RdYlGn', s=1, alpha=0.6)
ax.set_xlabel('SW Predicted')
ax.set_ylabel('Depth (m)')
ax.set_title('F-14 SW Predictions (colored by confidence)')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='Confidence (%)')

# 6. Confidence vs Uncertainty
ax = axes[1, 2]
ax.scatter(phif_std, phif_confidence_pct, alpha=0.3, s=1, label='PHIF', c='blue')
ax.scatter(sw_std, sw_confidence_pct, alpha=0.3, s=1, label='SW', c='red')
ax.set_xlabel('Prediction Std Dev')
ax.set_ylabel('Confidence (%)')
ax.set_title('Confidence vs Uncertainty')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. PHIF distribution comparison
ax = axes[2, 0]
ax.hist(y_train_phif, bins=50, alpha=0.5, label='Training wells', density=True, color='blue')
ax.hist(phif_pred, bins=50, alpha=0.5, label='F-14 predicted', density=True, color='orange')
ax.set_xlabel('PHIF')
ax.set_ylabel('Density')
ax.set_title('PHIF Distribution Comparison')
ax.legend()

# 8. SW distribution comparison
ax = axes[2, 1]
ax.hist(y_train_sw, bins=50, alpha=0.5, label='Training wells', density=True, color='blue')
ax.hist(sw_pred, bins=50, alpha=0.5, label='F-14 predicted', density=True, color='orange')
ax.set_xlabel('SW')
ax.set_ylabel('Density')
ax.set_title('SW Distribution Comparison')
ax.legend()

# 9. Confidence by depth zones
ax = axes[2, 2]
depth_zones = [(f14_depths.min(), 2500), (2500, 3000), (3000, 3500), (3500, f14_depths.max())]
zone_labels = []
phif_conf_means = []
sw_conf_means = []

for zone_start, zone_end in depth_zones:
    zone_mask = (f14_depths >= zone_start) & (f14_depths <= zone_end)
    if zone_mask.sum() > 0:
        zone_labels.append(f'{zone_start:.0f}-{zone_end:.0f}m')
        phif_conf_means.append(phif_confidence_pct[zone_mask].mean())
        sw_conf_means.append(sw_confidence_pct[zone_mask].mean())

x_pos = np.arange(len(zone_labels))
width = 0.35
ax.bar(x_pos - width/2, phif_conf_means, width, label='PHIF', color='blue')
ax.bar(x_pos + width/2, sw_conf_means, width, label='SW', color='red')
ax.set_xlabel('Depth Zone')
ax.set_ylabel('Mean Confidence (%)')
ax.set_title('Confidence by Depth Zones')
ax.set_xticks(x_pos)
ax.set_xticklabels(zone_labels, rotation=45)
ax.legend()

plt.suptitle('F-14 Prediction Results with Confidence Analysis', fontsize=14)
plt.tight_layout()
plt.savefig('f14_predictions_with_confidence.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Confidence interval visualization
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

# PHIF with confidence intervals
ax = axes[0, 0]
ax.plot(phif_pred, f14_depths, 'b-', linewidth=0.5, label='Prediction')
ax.fill_betweenx(f14_depths, phif_lower, phif_upper, alpha=0.2, color='blue', label='95% CI')
ax.set_xlabel('PHIF')
ax.set_ylabel('Depth (m)')
ax.set_title('PHIF Predictions with 95% Confidence Interval')
ax.invert_yaxis()
ax.legend()
ax.grid(True, alpha=0.3)

# SW with confidence intervals
ax = axes[0, 1]
ax.plot(sw_pred, f14_depths, 'r-', linewidth=0.5, label='Prediction')
ax.fill_betweenx(f14_depths, sw_lower, sw_upper, alpha=0.2, color='red', label='95% CI')
ax.set_xlabel('SW')
ax.set_ylabel('Depth (m)')
ax.set_title('SW Predictions with 95% Confidence Interval')
ax.invert_yaxis()
ax.legend()
ax.grid(True, alpha=0.3)

# Uncertainty vs depth for PHIF
ax = axes[1, 0]
scatter = ax.scatter(phif_std, f14_depths, c=f14_depths, cmap='viridis', s=1, alpha=0.6)
ax.set_xlabel('PHIF Prediction Std Dev')
ax.set_ylabel('Depth (m)')
ax.set_title('PHIF Uncertainty vs Depth')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='Depth (m)')
ax.grid(True, alpha=0.3)

# Uncertainty vs depth for SW
ax = axes[1, 1]
scatter = ax.scatter(sw_std, f14_depths, c=f14_depths, cmap='plasma', s=1, alpha=0.6)
ax.set_xlabel('SW Prediction Std Dev')
ax.set_ylabel('Depth (m)')
ax.set_title('SW Uncertainty vs Depth')
ax.invert_yaxis()
plt.colorbar(scatter, ax=ax, label='Depth (m)')
ax.grid(True, alpha=0.3)

plt.suptitle('Confidence Intervals and Uncertainty Analysis', fontsize=14)
plt.tight_layout()
plt.savefig('f14_confidence_intervals.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\n" + "="*60)
print("DETAILED CONFIDENCE STATISTICS")
print("="*60)

print("\nPHIF Confidence Statistics:")
print(f"  Mean: {phif_confidence_pct.mean():.2f}%")
print(f"  Median: {np.median(phif_confidence_pct):.2f}%")
print(f"  Std Dev: {phif_confidence_pct.std():.2f}%")
print(f"  Min: {phif_confidence_pct.min():.2f}%")
print(f"  Max: {phif_confidence_pct.max():.2f}%")

print("\nSW Confidence Statistics:")
print(f"  Mean: {sw_confidence_pct.mean():.2f}%")
print(f"  Median: {np.median(sw_confidence_pct):.2f}%")
print(f"  Std Dev: {sw_confidence_pct.std():.2f}%")
print(f"  Min: {sw_confidence_pct.min():.2f}%")
print(f"  Max: {sw_confidence_pct.max():.2f}%")

# Samples with low confidence
low_conf_threshold = 30  # %
phif_low_conf = phif_confidence_pct < low_conf_threshold
sw_low_conf = sw_confidence_pct < low_conf_threshold

print(f"\nSamples with confidence < {low_conf_threshold}%:")
print(f"  PHIF: {phif_low_conf.sum()} samples ({phif_low_conf.sum()/len(f14_data)*100:.1f}%)")
print(f"  SW: {sw_low_conf.sum()} samples ({sw_low_conf.sum()/len(f14_data)*100:.1f}%)")

# High confidence samples
high_conf_threshold = 70  # %
phif_high_conf = phif_confidence_pct > high_conf_threshold
sw_high_conf = sw_confidence_pct > high_conf_threshold

print(f"\nSamples with confidence > {high_conf_threshold}%:")
print(f"  PHIF: {phif_high_conf.sum()} samples ({phif_high_conf.sum()/len(f14_data)*100:.1f}%)")
print(f"  SW: {sw_high_conf.sum()} samples ({sw_high_conf.sum()/len(f14_data)*100:.1f}%)")

print("\n" + "="*60)
print("PROCESS COMPLETE")
print("="*60)
print(f"✓ Models trained successfully with lithology features")
print(f"✓ F-14 PHIF and SW values predicted with confidence scores")
print(f"✓ Dataset saved with proper numeric formatting")
print(f"✓ Files created:")
print(f"  - {output_filename} (CSV with confidence columns)")
print(f"  - {excel_filename} (Excel format - no apostrophe issues)")
print(f"✓ Visualizations saved as PNG files")
print(f"✓ Cross-validation R²: PHIF={r2_phif:.4f}, SW={r2_sw:.4f}")
print("="*60)
