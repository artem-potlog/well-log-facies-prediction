import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Default window size for AI Slope calculation (in meters)
DEFAULT_AI_SLOPE_WINDOW = 0.5  # meters

# Parse command line arguments
parser = argparse.ArgumentParser(description='Prepare ML-ready dataset with AI Slope engineered feature')
parser.add_argument('--ai-slope-window', type=float, default=DEFAULT_AI_SLOPE_WINDOW,
                    help=f'Window size for AI Slope calculation in meters (default: {DEFAULT_AI_SLOPE_WINDOW})')
parser.add_argument('--input', type=str, default='Dataset_logs_core_v4_cleaned.csv',
                    help='Input dataset path (default: Dataset_logs_core_v4_cleaned.csv)')
parser.add_argument('--output', type=str, default=None,
                    help='Output filename (default: auto-generated based on window size)')
args = parser.parse_args()

# Set configuration from arguments
window_size_ai_slope = args.ai_slope_window
input_file = args.input

# Generate output filename based on window size if not specified
if args.output is None:
    if window_size_ai_slope == DEFAULT_AI_SLOPE_WINDOW:
        output_filename = 'ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_AISlope_facies_litho.csv'
    else:
        # Include window size in filename if not default
        window_str = str(window_size_ai_slope).replace('.', 'p')
        output_filename = f'ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_AISlope{window_str}m_facies_litho.csv'
else:
    output_filename = args.output

# Display configuration
print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Input dataset: {input_file}")
print(f"Output dataset: {output_filename}")
print(f"AI Slope window size: {window_size_ai_slope} meters")
print("="*60 + "\n")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(input_file)
print(f"Initial dataset shape: {df.shape}")

# Define the actual column names in the dataset
actual_columns = {
    'Well': 'Well Name',
    'Depth': 'MD, m',
    'GR': 'GR_Cropped _Combined_Final, API',
    'VSH': 'VSH_Cropped _Combined_Final, v/v',
    'RT': 'RT_Cropped _Combined_Final, ohm.m',
    'RD': 'RD_Cropped _Combined_Final, ohm.m',
    'NPHI': 'NPHI_Cropped _Combined_Final, v/v',
    'RHOB': 'RHOB_Cropped _Combined_Final, g/cm3',
    'DT': 'DT_Cropped _Combined_Final, us/ft',
    'PHIF': 'PHIF_Cropped _Combined_Final',
    'SW': 'SW_Cropped _Combined_Final',
    'KLOGH': 'KLOGH_Cropped _Combined_Final',
    'Equinor facies': 'Equinor_Facies_Cropped',
    'Lithology': 'Lithology_HWU_Cropped'
}

# Create a dataframe with standardized column names
df_selected = pd.DataFrame()
for standard_name, actual_name in actual_columns.items():
    if actual_name in df.columns:
        df_selected[standard_name] = df[actual_name]
    else:
        print(f"Warning: Column '{actual_name}' not found in dataset")

# Define the feature logs (labels excluded from this list)
# Note: AI_Slope will be added after calculation
feature_logs = ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH']

# Verify all required columns are present
required_columns = list(actual_columns.keys())
missing_cols = [col for col in required_columns if col not in df_selected.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

initial_rows = len(df_selected)
print(f"Initial rows after column selection: {initial_rows}")

# Initialize statistics dictionary
stats = {
    'initial_rows': initial_rows,
    'after_empty_removal': 0,
    'after_zero_removal': 0,
    'after_unrealistic_removal': 0,
    'after_imputation': 0,
    'after_drop_unlabeled': 0,
    'after_drop_unlithology': 0,
    'final_rows': 0
}

# Step 1: Drop rows with empty values in selected feature logs (labels may be NaN at this step)
print("\nStep 1: Removing rows with empty values in selected feature logs...")
mask_not_empty = df_selected[feature_logs].notna().all(axis=1)
df_selected = df_selected[mask_not_empty].copy()
stats['after_empty_removal'] = len(df_selected)
print(f"Rows after removing empty values: {stats['after_empty_removal']} (dropped: {initial_rows - stats['after_empty_removal']})")

# Step 2: Replace zeros with NaN and drop those rows (features only)
print("\nStep 2: Replacing zeros with NaN and dropping those rows (features only)...")
for col in feature_logs:
    df_selected.loc[df_selected[col] == 0, col] = np.nan
mask_not_zero = df_selected[feature_logs].notna().all(axis=1)
df_selected = df_selected[mask_not_zero].copy()
stats['after_zero_removal'] = len(df_selected)
print(f"Rows after removing zeros: {stats['after_zero_removal']} (dropped: {stats['after_empty_removal'] - stats['after_zero_removal']})")

# Step 3: Realistic value checks for logs (KLOGH excluded)
print("\nStep 3: Checking for realistic log values...")
realistic_ranges = {
    'GR': (0, 300),         # API units
    'VSH': (0, 1),          # fraction
    'RT': (0.1, 10000),     # ohm.m
    'RD': (0.1, 10000),     # ohm.m
    'NPHI': (-0.15, 0.6),   # fraction
    'RHOB': (1.5, 3.2),     # g/cc
    'DT': (40, 200),        # us/ft
    'PHIF': (0, 0.4),       # fraction
    'SW': (0, 1)            # fraction
    # KLOGH excluded - assumed realistic
}

for col in feature_logs:
    if col not in realistic_ranges:
        print(f"  {col}: Skipping realistic value check")
        continue
    min_val, max_val = realistic_ranges[col]
    mask = (df_selected[col] < min_val) | (df_selected[col] > max_val)
    n_unrealistic = mask.sum()
    if n_unrealistic > 0:
        print(f"  {col}: Found {n_unrealistic} unrealistic values outside range [{min_val}, {max_val}]")
        df_selected.loc[mask, col] = np.nan

# Drop rows with NaN values after unrealistic value replacement
mask_realistic = df_selected[feature_logs].notna().all(axis=1)
df_selected = df_selected[mask_realistic].copy()
stats['after_unrealistic_removal'] = len(df_selected)
print(f"Rows after removing unrealistic values: {stats['after_unrealistic_removal']} (dropped: {stats['after_zero_removal'] - stats['after_unrealistic_removal']})")

# Step 4: Per-well KNN imputation with depth awareness (features only)
print("\nStep 4: Performing per-well KNN imputation (if needed)...")
df_imputed = df_selected.copy()
wells = df_selected['Well'].unique()
print(f"Number of wells: {len(wells)}")

for well in wells:
    well_mask = df_selected['Well'] == well
    well_data = df_selected[well_mask].copy()

    if len(well_data) < 5:
        print(f"  Skipping {well}: too few samples ({len(well_data)})")
        continue

    if well_data[feature_logs].isna().sum().sum() > 0:
        print(f"  Imputing {well}...")
        depth_scaler = StandardScaler()
        well_data['Depth_scaled'] = depth_scaler.fit_transform(well_data[['Depth']])
        X_impute = well_data[['Depth_scaled'] + feature_logs].values
        imputer = KNNImputer(n_neighbors=min(5, len(well_data) - 1))
        X_imputed = imputer.fit_transform(X_impute)
        df_imputed.loc[well_mask, feature_logs] = X_imputed[:, 1:]

# If any NaNs remain, drop those rows
nan_count = df_imputed[feature_logs].isna().sum().sum()
if nan_count > 0:
    print(f"\nWarning: {nan_count} NaN values remain after imputation. Dropping these rows...")
    mask_no_nan = df_imputed[feature_logs].notna().all(axis=1)
    df_imputed = df_imputed[mask_no_nan].copy()

stats['after_imputation'] = len(df_imputed)
print(f"Rows after imputation: {stats['after_imputation']}")

# Step 5: Remove rows without facies labels (ensure supervised-ready)
print("\nStep 5: Dropping rows without Equinor facies labels...")
before_labels = len(df_imputed)
df_imputed = df_imputed[df_imputed['Equinor facies'].notna()].copy()
stats['after_drop_unlabeled'] = len(df_imputed)
print(f"Rows after dropping unlabeled facies: {stats['after_drop_unlabeled']} (dropped: {before_labels - stats['after_drop_unlabeled']})")

# Step 6: Remove rows without lithology labels
print("\nStep 6: Dropping rows without lithology labels...")
before_litho = len(df_imputed)
df_imputed = df_imputed[df_imputed['Lithology'].notna()].copy()
stats['after_drop_unlithology'] = len(df_imputed)
print(f"Rows after dropping unlabeled lithology: {stats['after_drop_unlithology']} (dropped: {before_litho - stats['after_drop_unlithology']})")

# Step 7: Create one-hot encoded lithology columns
print("\nStep 7: Creating one-hot encoded lithology columns...")
lithology_values = [0, 1, 2, 3, 4]
for litho_val in lithology_values:
    df_imputed[f'Lithology_{litho_val}'] = (df_imputed['Lithology'] == litho_val).astype(int)

# Verify one-hot encoding
print("Lithology one-hot encoding distribution:")
for litho_val in lithology_values:
    count = df_imputed[f'Lithology_{litho_val}'].sum()
    print(f"  Lithology_{litho_val}: {count} samples")

# Step 8: Apply log10 transformation to RT and RD before normalization
print("\nStep 8: Applying log10 transformation to RT and RD...")
df_imputed['RT'] = np.log10(df_imputed['RT'])
df_imputed['RD'] = np.log10(df_imputed['RD'])

# NEW STEP: Calculate AI Slope (Acoustic Impedance slope) using configured window size
print(f"\nStep 8a: Calculating AI Slope (Acoustic Impedance slope) in {window_size_ai_slope} m window...")
df_imputed['AI_Slope'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 2:
        continue
    
    # Sort by depth to ensure proper ordering
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    # Calculate AI as RHOB * DT (normalized version)
    ai_values = well_data['RHOB'].values * well_data['DT'].values
    ai_slopes = np.zeros(len(depths))
    
    for i in range(len(depths)):
        # Find points within the window (±window_size/2)
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_ai_slope / 2
        
        # Need at least 2 points to calculate slope
        if np.sum(window_mask) >= 2:
            window_depths = depths[window_mask]
            window_ai = ai_values[window_mask]
            
            # Calculate slope using linear regression
            if len(window_depths) > 1:
                # Use numpy polyfit for robust slope calculation
                coeffs = np.polyfit(window_depths, window_ai, 1)
                ai_slopes[i] = coeffs[0]  # Slope is the first coefficient
            else:
                ai_slopes[i] = 0
        else:
            ai_slopes[i] = 0
    
    # Assign slopes back to the dataframe
    df_imputed.loc[well_indices, 'AI_Slope'] = ai_slopes

# Add AI_Slope to feature logs
feature_logs.append('AI_Slope')

# Check for any NaN values in AI_Slope and handle them
ai_slope_nans = df_imputed['AI_Slope'].isna().sum()
if ai_slope_nans > 0:
    print(f"Warning: {ai_slope_nans} NaN values in AI_Slope. Filling with 0...")
    df_imputed['AI_Slope'].fillna(0, inplace=True)

print(f"AI_Slope calculated (window={window_size_ai_slope}m). Range: [{df_imputed['AI_Slope'].min():.4f}, {df_imputed['AI_Slope'].max():.4f}]")
print(f"AI_Slope mean: {df_imputed['AI_Slope'].mean():.4f}, std: {df_imputed['AI_Slope'].std():.4f}")

# Step 9: Global normalization using StandardScaler (features only, not labels)
print("\nStep 9: Applying global normalization to selected features...")
scaler = StandardScaler()
df_normalized = df_imputed.copy()
df_normalized[feature_logs] = scaler.fit_transform(df_imputed[feature_logs])

# Final dataset
df_final = df_normalized.copy()
stats['final_rows'] = len(df_final)

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Initial rows: {stats['initial_rows']}")
print(f"After removing empty values: {stats['after_empty_removal']} (dropped: {stats['initial_rows'] - stats['after_empty_removal']})")
print(f"After removing zeros: {stats['after_zero_removal']} (dropped: {stats['after_empty_removal'] - stats['after_zero_removal']})")
print(f"After removing unrealistic values: {stats['after_unrealistic_removal']} (dropped: {stats['after_zero_removal'] - stats['after_unrealistic_removal']})")
print(f"After imputation: {stats['after_imputation']} (dropped: {stats['after_unrealistic_removal'] - stats['after_imputation']})")
print(f"After dropping unlabeled facies: {stats['after_drop_unlabeled']} (dropped: {before_labels - stats['after_drop_unlabeled']})")
print(f"After dropping unlabeled lithology: {stats['after_drop_unlithology']} (dropped: {before_litho - stats['after_drop_unlithology']})")
print(f"Final rows: {stats['final_rows']}")
print(f"Total rows dropped: {stats['initial_rows'] - stats['final_rows']} ({(stats['initial_rows'] - stats['final_rows'])/stats['initial_rows']*100:.2f}%)")
print(f"Total rows retained: {stats['final_rows']} ({stats['final_rows']/stats['initial_rows']*100:.2f}%)")

# Save the ML-ready dataset (includes facies label, one-hot encoded lithology, and AI_Slope)
df_final.to_csv(output_filename, index=False)
print(f"\nML-ready dataset saved as: {output_filename}")

# Print sample of the final dataset
print("\nSample of final dataset:")
print(df_final.head())

# Print normalized value ranges
print("\nNormalized value ranges (features only):")
for col in feature_logs:
    print(f"{col}: min={df_final[col].min():.3f}, max={df_final[col].max():.3f}, mean={df_final[col].mean():.3f}, std={df_final[col].std():.3f}")

# Facies distribution
print("\nFacies distribution in final dataset:")
facies_counts = df_final['Equinor facies'].value_counts().sort_index()
for facies, count in facies_counts.items():
    print(f"  {int(facies)}: {count} samples ({count/len(df_final)*100:.1f}%)")

# Lithology distribution
print("\nLithology distribution in final dataset:")
litho_counts = df_final['Lithology'].value_counts().sort_index()
for litho, count in litho_counts.items():
    print(f"  {int(litho)}: {count} samples ({count/len(df_final)*100:.1f}%)")

# Wells and their sample counts
print("\nWells in final dataset:")
well_counts = df_final['Well'].value_counts().sort_index()
for well, count in well_counts.items():
    print(f"  {well}: {count} samples")
