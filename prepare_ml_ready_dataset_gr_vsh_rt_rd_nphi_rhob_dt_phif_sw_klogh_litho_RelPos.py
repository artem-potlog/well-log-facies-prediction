import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter, find_peaks
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Parameters for Relative Position (RelPos) calculation
# These control how GR cycles are detected and characterized

# Smoothing window for Savitzky-Golay filter (in number of samples)
# - Smaller windows (11-15): Less smoothing, captures smaller cycles
# - Medium windows (17-25): Balanced smoothing (default: 21)
# - Larger windows (27-35): More smoothing, captures larger cycles
RELPOS_SMOOTH_WINDOW = 12  # samples (default: 21)

# Minimum distance between peaks/valleys in cycle detection (in samples)
# - Smaller distance (5-8): Detects more frequent cycles
# - Medium distance (9-15): Balanced detection (default: 10)
# - Larger distance (16-25): Detects only major cycles
RELPOS_PEAK_DISTANCE = 6  # samples (default: 10)

# Note: 1 sample typically = 0.1-0.15 meters depending on sampling rate

# Input and output file paths
INPUT_FILE = 'Dataset_logs_core_v4_cleaned.csv'
OUTPUT_FILE = 'ML_ready_GR_VSH_RT_RD_NPHI_RHOB_DT_PHIF_SW_KLOGH_RelPos_facies_litho.csv'
# ========================================================

# Load the dataset
print("="*60)
print("RELATIVE POSITION (RelPos) DATA PREPARATION")
print("="*60)
print(f"Configuration:")
print(f"  - Input file: {INPUT_FILE}")
print(f"  - Output file: {OUTPUT_FILE}")
print(f"  - Smoothing window: {RELPOS_SMOOTH_WINDOW} samples")
print(f"  - Peak detection distance: {RELPOS_PEAK_DISTANCE} samples")
print("="*60 + "\n")

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
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
# Note: RelPos will be added after calculation
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

# NEW STEP: Calculate Relative Position (RelPos) in GR cycles
print(f"\nStep 8a: Calculating Relative Position (RelPos) in GR cycles...")
print(f"  Using smoothing window: {RELPOS_SMOOTH_WINDOW} samples")
print(f"  Using peak detection distance: {RELPOS_PEAK_DISTANCE} samples")
df_imputed['RelPos'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 10:  # Need enough points for smoothing and cycle detection
        print(f"  Skipping {well}: too few samples for cycle detection ({len(well_data)})")
        df_imputed.loc[well_mask, 'RelPos'] = 0.5  # Default to middle position
        continue
    
    # Sort by depth to ensure proper ordering
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    
    # Smooth GR curve to reduce noise
    # Window length should be odd and less than data length
    window_length = min(RELPOS_SMOOTH_WINDOW, len(gr_values))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(5, window_length)  # At least 5 points
    
    try:
        gr_smooth = savgol_filter(gr_values, window_length, 3)
    except:
        # If smoothing fails, use original values
        gr_smooth = gr_values
    
    # Find peaks (maxima - muddy) and valleys (minima - clean)
    peaks, _ = find_peaks(gr_smooth, distance=RELPOS_PEAK_DISTANCE)  # Minimum samples between peaks
    valleys, _ = find_peaks(-gr_smooth, distance=RELPOS_PEAK_DISTANCE)  # Invert to find minima
    
    # Combine and sort turning points
    turning_points = np.sort(np.concatenate([peaks, valleys]))
    
    # Add boundaries if needed
    if len(turning_points) == 0 or turning_points[0] > 0:
        turning_points = np.concatenate([[0], turning_points])
    if turning_points[-1] < len(gr_values) - 1:
        turning_points = np.concatenate([turning_points, [len(gr_values) - 1]])
    
    # Calculate RelPos for each point
    rel_positions = np.zeros(len(gr_values))
    
    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx = turning_points[i + 1]
        
        # Get GR values at cycle boundaries
        gr_start = gr_smooth[start_idx]
        gr_end = gr_smooth[end_idx]
        
        # Determine cycle type
        if gr_start > gr_end:
            # Coarsening-up cycle (GR decreasing)
            # RelPos goes from 0 (muddy) to 1 (clean)
            for j in range(start_idx, min(end_idx + 1, len(gr_values))):
                if gr_start != gr_end:
                    # Normalize position based on GR value
                    rel_positions[j] = (gr_start - gr_smooth[j]) / (gr_start - gr_end)
                else:
                    rel_positions[j] = 0.5
        else:
            # Fining-up cycle (GR increasing)
            # RelPos goes from 1 (clean) to 0 (muddy)
            for j in range(start_idx, min(end_idx + 1, len(gr_values))):
                if gr_end != gr_start:
                    # Normalize position based on GR value
                    rel_positions[j] = 1 - (gr_smooth[j] - gr_start) / (gr_end - gr_start)
                else:
                    rel_positions[j] = 0.5
    
    # Clip values to [0, 1] range
    rel_positions = np.clip(rel_positions, 0, 1)
    
    # Assign back to dataframe
    df_imputed.loc[well_indices, 'RelPos'] = rel_positions
    
    # Print summary for this well
    n_cycles = len(turning_points) - 1
    print(f"  {well}: {n_cycles} cycles detected, RelPos range: [{rel_positions.min():.3f}, {rel_positions.max():.3f}]")

# Add RelPos to feature logs
feature_logs.append('RelPos')

# Check for any NaN values in RelPos and handle them
relpos_nans = df_imputed['RelPos'].isna().sum()
if relpos_nans > 0:
    print(f"Warning: {relpos_nans} NaN values in RelPos. Filling with 0.5...")
    df_imputed['RelPos'].fillna(0.5, inplace=True)

print(f"RelPos calculated. Range: [{df_imputed['RelPos'].min():.4f}, {df_imputed['RelPos'].max():.4f}]")
print(f"RelPos mean: {df_imputed['RelPos'].mean():.4f}, std: {df_imputed['RelPos'].std():.4f}")

# Print RelPos statistics by facies
print("\nRelPos statistics by facies:")
for facies in sorted(df_imputed['Equinor facies'].unique()):
    facies_data = df_imputed[df_imputed['Equinor facies'] == facies]['RelPos']
    print(f"  Facies {int(facies)}: mean={facies_data.mean():.4f}, std={facies_data.std():.4f}, median={facies_data.median():.4f}")

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

# Save the ML-ready dataset (includes facies label, one-hot encoded lithology, and RelPos)
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nML-ready dataset saved as: {OUTPUT_FILE}")
if RELPOS_SMOOTH_WINDOW != 21 or RELPOS_PEAK_DISTANCE != 10:
    print(f"Note: Using non-default parameters:")
    if RELPOS_SMOOTH_WINDOW != 21:
        print(f"  - Smoothing window: {RELPOS_SMOOTH_WINDOW} samples (default: 21)")
    if RELPOS_PEAK_DISTANCE != 10:
        print(f"  - Peak distance: {RELPOS_PEAK_DISTANCE} samples (default: 10)")

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
