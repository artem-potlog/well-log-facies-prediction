import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Dataset_logs_core_v4_cleaned.csv')
print(f"Initial dataset shape: {df.shape}")

# Define the actual column names in the dataset - ONLY 10 base features
actual_columns = {
    'Well': 'Well Name',
    'Depth': 'MD, m',
    'GR': 'GR_Cropped _Combined_Final, API',
    'RT': 'RT_Cropped _Combined_Final, ohm.m',
    'NPHI': 'NPHI_Cropped _Combined_Final, v/v',
    'RHOB': 'RHOB_Cropped _Combined_Final, g/cm3',
    'RD': 'RD_Cropped _Combined_Final, ohm.m',
    'DT': 'DT_Cropped _Combined_Final, us/ft',
    'VSH': 'VSH_Cropped _Combined_Final, v/v',
    'KLOGH': 'KLOGH_Cropped _Combined_Final',
    'PHIF': 'PHIF_Cropped _Combined_Final',
    'SW': 'SW_Cropped _Combined_Final',
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

# Define the 10 base feature logs
# Core features that must be valid (no zeros/NaNs allowed)
core_features = ['GR', 'RT', 'NPHI', 'RHOB', 'RD', 'DT', 'VSH', 'KLOGH']
# Special features that can have zeros/NaNs (will be empty for F-14)
special_features = ['PHIF', 'SW']
# All base features combined
base_feature_logs = core_features + special_features

print(f"\nCore features (must be valid): {core_features}")
print(f"Special features (zeros/NaNs allowed, empty for F-14): {special_features}")
print(f"Total base features: {len(base_feature_logs)}")

# Special handling for well F-14: Set PHIF and SW to NaN
print(f"\nSpecial handling: Setting PHIF and SW to NaN for well F-14...")
f14_mask = df_selected['Well'].str.contains('F-14', na=False)
f14_count = f14_mask.sum()
print(f"Found {f14_count} samples for well F-14")

if f14_count > 0:
    df_selected.loc[f14_mask, 'PHIF'] = np.nan
    df_selected.loc[f14_mask, 'SW'] = np.nan
    print("PHIF and SW set to NaN for all F-14 samples")

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
    'final_rows': 0
}

# Step 1: Drop rows with empty values ONLY in core features (PHIF/SW can be NaN)
print("\nStep 1: Removing rows with empty values in CORE features only...")
print("  (PHIF and SW are allowed to have NaN values)")
mask_not_empty = df_selected[core_features].notna().all(axis=1)
df_selected = df_selected[mask_not_empty].copy()
stats['after_empty_removal'] = len(df_selected)
print(f"Rows after removing empty core values: {stats['after_empty_removal']} (dropped: {initial_rows - stats['after_empty_removal']})")

# Step 2: Replace zeros with NaN and drop those rows (ONLY for core features, NOT for PHIF/SW)
print("\nStep 2: Replacing zeros with NaN and dropping those rows (CORE features only)...")
print("  (PHIF and SW are allowed to have zero values)")
for col in core_features:  # Only apply to core features
    df_selected.loc[df_selected[col] == 0, col] = np.nan

mask_not_zero = df_selected[core_features].notna().all(axis=1)  # Only check core features
df_selected = df_selected[mask_not_zero].copy()
stats['after_zero_removal'] = len(df_selected)
print(f"Rows after removing zeros in core features: {stats['after_zero_removal']} (dropped: {stats['after_empty_removal'] - stats['after_zero_removal']})")

# Step 3: Realistic value checks for logs (KLOGH excluded, PHIF/SW have special handling)
print("\nStep 3: Checking for realistic log values...")
realistic_ranges = {
    'GR': (0, 300),         # API units
    'RT': (0.1, 10000),     # ohm.m
    'NPHI': (-0.15, 0.6),   # fraction
    'RHOB': (1.5, 3.2),     # g/cc
    'RD': (0.1, 10000),     # ohm.m
    'DT': (40, 200),        # us/ft
    'VSH': (0, 1),          # fraction
    # KLOGH excluded - assumed realistic
    # PHIF: (0, 0.4) - allowing zeros/NaNs, but check realistic range when not zero/NaN
    # SW: (0, 1) - allowing zeros/NaNs, but check realistic range when not zero/NaN
}

# Handle core features with standard realistic checks
for col in core_features:
    if col not in realistic_ranges:
        print(f"  {col}: Skipping realistic value check")
        continue
    min_val, max_val = realistic_ranges[col]
    mask = (df_selected[col] < min_val) | (df_selected[col] > max_val)
    n_unrealistic = mask.sum()
    if n_unrealistic > 0:
        print(f"  {col}: Found {n_unrealistic} unrealistic values outside range [{min_val}, {max_val}]")
        df_selected.loc[mask, col] = np.nan

# Handle special features (PHIF/SW) - only check realistic ranges when values are not NaN or zero
for col in special_features:
    if col == 'PHIF':
        # PHIF realistic range: (0, 0.4) but allow zeros and NaNs
        mask = df_selected[col].notna() & (df_selected[col] != 0) & ((df_selected[col] < 0) | (df_selected[col] > 0.4))
        n_unrealistic = mask.sum()
        if n_unrealistic > 0:
            print(f"  {col}: Found {n_unrealistic} unrealistic non-zero values outside range [0, 0.4]")
            df_selected.loc[mask, col] = np.nan
        else:
            print(f"  {col}: All non-zero values are realistic (zeros and NaNs preserved)")
    elif col == 'SW':
        # SW realistic range: (0, 1) but allow zeros and NaNs
        mask = df_selected[col].notna() & (df_selected[col] != 0) & ((df_selected[col] < 0) | (df_selected[col] > 1))
        n_unrealistic = mask.sum()
        if n_unrealistic > 0:
            print(f"  {col}: Found {n_unrealistic} unrealistic non-zero values outside range [0, 1]")
            df_selected.loc[mask, col] = np.nan
        else:
            print(f"  {col}: All non-zero values are realistic (zeros and NaNs preserved)")

# Drop rows with NaN values after unrealistic value replacement (ONLY for core features)
mask_realistic = df_selected[core_features].notna().all(axis=1)
df_selected = df_selected[mask_realistic].copy()
stats['after_unrealistic_removal'] = len(df_selected)
print(f"Rows after removing unrealistic core values: {stats['after_unrealistic_removal']} (dropped: {stats['after_zero_removal'] - stats['after_unrealistic_removal']})")

# Step 4: Per-well KNN imputation with depth awareness
print("\nStep 4: Performing per-well KNN imputation (if needed)...")
print("  Core features: Standard imputation")
print("  PHIF/SW: Will be imputed if NaN (except F-14 which stays NaN), zeros are preserved")

df_imputed = df_selected.copy()
wells = df_selected['Well'].unique()
print(f"Number of wells: {len(wells)}")

for well in wells:
    well_mask = df_selected['Well'] == well
    well_data = df_selected[well_mask].copy()

    if len(well_data) < 5:
        print(f"  Skipping {well}: too few samples ({len(well_data)})")
        continue

    # For F-14, don't impute PHIF/SW (keep them as NaN)
    if 'F-14' in well:
        # Only impute core features for F-14
        impute_features = core_features
        print(f"  Imputing {well} (F-14 - only core features, PHIF/SW stay NaN)...")
    else:
        # For other wells, impute all features
        impute_features = base_feature_logs
        print(f"  Imputing {well} (all features)...")

    # Check if imputation is needed for the relevant features
    needs_imputation = well_data[impute_features].isna().sum().sum() > 0
    
    if needs_imputation:
        # Create a copy to avoid issues with indexing
        well_data_copy = well_data.copy()
        depth_scaler = StandardScaler()
        well_data_copy['Depth_scaled'] = depth_scaler.fit_transform(well_data_copy[['Depth']])
        
        # Build the feature matrix for imputation
        impute_columns = ['Depth_scaled'] + impute_features
        X_impute = well_data_copy[impute_columns].values
        
        # Perform imputation
        imputer = KNNImputer(n_neighbors=min(5, len(well_data_copy) - 1))
        X_imputed = imputer.fit_transform(X_impute)
        
        # Assign back to the main dataframe, excluding the depth column
        for i, feature in enumerate(impute_features):
            df_imputed.loc[well_mask, feature] = X_imputed[:, i + 1]  # Skip depth column

# Check remaining NaNs only for core features 
core_nan_count = df_imputed[core_features].isna().sum().sum()
if core_nan_count > 0:
    print(f"\nWarning: {core_nan_count} NaN values remain in core features after imputation. Dropping these rows...")
    mask_no_core_nan = df_imputed[core_features].notna().all(axis=1)
    df_imputed = df_imputed[mask_no_core_nan].copy()

# Handle remaining NaNs in PHIF/SW by filling with zeros (except F-14)
phif_nan_count = df_imputed['PHIF'].isna().sum()
sw_nan_count = df_imputed['SW'].isna().sum()

# Fill NaNs with zeros, but preserve F-14 NaNs
f14_mask = df_imputed['Well'].str.contains('F-14', na=False)
non_f14_mask = ~f14_mask

if phif_nan_count > 0:
    f14_phif_nans = df_imputed.loc[f14_mask, 'PHIF'].isna().sum()
    other_phif_nans = df_imputed.loc[non_f14_mask, 'PHIF'].isna().sum()
    print(f"  PHIF NaNs: {f14_phif_nans} in F-14 (preserved), {other_phif_nans} in others (filled with 0)")
    df_imputed.loc[non_f14_mask & df_imputed['PHIF'].isna(), 'PHIF'] = 0

if sw_nan_count > 0:
    f14_sw_nans = df_imputed.loc[f14_mask, 'SW'].isna().sum()
    other_sw_nans = df_imputed.loc[non_f14_mask, 'SW'].isna().sum()
    print(f"  SW NaNs: {f14_sw_nans} in F-14 (preserved), {other_sw_nans} in others (filled with 0)")
    df_imputed.loc[non_f14_mask & df_imputed['SW'].isna(), 'SW'] = 0

stats['after_imputation'] = len(df_imputed)
print(f"Rows after imputation: {stats['after_imputation']}")

# Step 5: Skip dropping facies labels (keep all rows)
print("\nStep 5: SKIPPED - Keeping all rows (not dropping unlabeled facies)")

# Step 6: Skip dropping lithology labels (keep all rows) 
print("\nStep 6: SKIPPED - Keeping all rows (not dropping unlabeled lithology)")

# Step 7: Create one-hot encoded lithology columns (only for non-NaN lithology)
print("\nStep 7: Creating one-hot encoded lithology columns...")
print("  Note: Keeping NaN for samples with missing lithology labels")

# Check how many samples have NaN lithology
nan_litho_count = df_imputed['Lithology'].isna().sum()
print(f"  Samples with NaN lithology: {nan_litho_count} ({nan_litho_count/len(df_imputed)*100:.1f}%)")

lithology_values = [0, 1, 2, 3, 4]
for litho_val in lithology_values:
    # Initialize column with NaN
    df_imputed[f'Lithology_{litho_val}'] = np.nan
    
    # For non-NaN lithology, set 1 if matches, 0 otherwise
    non_nan_mask = df_imputed['Lithology'].notna()
    df_imputed.loc[non_nan_mask, f'Lithology_{litho_val}'] = (
        df_imputed.loc[non_nan_mask, 'Lithology'] == litho_val
    ).astype(int)
    
# Show the distribution (excluding NaN)
print("Lithology one-hot encoding distribution (non-NaN samples only):")
for litho_val in lithology_values:
    count = df_imputed[f'Lithology_{litho_val}'].sum()
    nan_count = df_imputed[f'Lithology_{litho_val}'].isna().sum()
    print(f"  Lithology_{litho_val}: {int(count)} samples (+ {nan_count} NaN)")

# Step 8: Apply log10 transformation to RT and RD before normalization
print("\nStep 8: Applying log10 transformation to RT and RD...")
df_imputed['RT'] = np.log10(df_imputed['RT'])
df_imputed['RD'] = np.log10(df_imputed['RD'])

# Update wells list after filtering
wells = df_imputed['Well'].unique()

# ENGINEERED FEATURES SECTION
print("\n=== Calculating Engineered Features ===")
print("Note: OppositionIndex will use available data (GR-PHIF correlation, with special handling for F-14)")

# 1. GR Slope (d(GR)/dz) in 0.3 m window
print("\nCalculating GR Slope (d(GR)/dz) in 0.3 m window...")
window_size_slope = 0.3  # meters
df_imputed['GR_Slope'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 2:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    gr_slopes = np.zeros(len(depths))
    
    for i in range(len(depths)):
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_slope / 2
        
        if np.sum(window_mask) >= 2:
            window_depths = depths[window_mask]
            window_gr = gr_values[window_mask]
            
            if len(window_depths) > 1:
                coeffs = np.polyfit(window_depths, window_gr, 1)
                gr_slopes[i] = coeffs[0]
            else:
                gr_slopes[i] = 0
        else:
            if i == 0 and len(depths) > 1:
                gr_slopes[i] = (gr_values[1] - gr_values[0]) / (depths[1] - depths[0])
            elif i == len(depths) - 1 and len(depths) > 1:
                gr_slopes[i] = (gr_values[-1] - gr_values[-2]) / (depths[-1] - depths[-2])
            else:
                gr_slopes[i] = 0
    
    df_imputed.loc[well_indices, 'GR_Slope'] = gr_slopes

# 2. GR Serration (rolling std of GR) in 3m window
print("\nCalculating GR Serration (rolling std) in 3m window...")
window_size_serr = 3.0  # meters
df_imputed['GR_Serration'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 2:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    gr_serrations = np.zeros(len(depths))
    
    for i in range(len(depths)):
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_serr / 2
        
        if np.sum(window_mask) >= 2:
            window_gr = gr_values[window_mask]
            gr_serrations[i] = np.std(window_gr, ddof=1)
        else:
            gr_serrations[i] = 0
    
    df_imputed.loc[well_indices, 'GR_Serration'] = gr_serrations

# 3. Relative Position (RelPos) in GR cycles
print("\nCalculating Relative Position (RelPos) in GR cycles...")
df_imputed['RelPos'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 10:
        df_imputed.loc[well_mask, 'RelPos'] = 0.5
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    
    window_length = min(21, len(gr_values))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(5, window_length)
    
    try:
        gr_smooth = savgol_filter(gr_values, window_length, 3)
    except:
        gr_smooth = gr_values
    
    peaks, _ = find_peaks(gr_smooth, distance=10)
    valleys, _ = find_peaks(-gr_smooth, distance=10)
    
    turning_points = np.sort(np.concatenate([peaks, valleys]))
    
    if len(turning_points) == 0 or turning_points[0] > 0:
        turning_points = np.concatenate([[0], turning_points])
    if turning_points[-1] < len(gr_values) - 1:
        turning_points = np.concatenate([turning_points, [len(gr_values) - 1]])
    
    rel_positions = np.zeros(len(gr_values))
    
    for i in range(len(turning_points) - 1):
        start_idx = turning_points[i]
        end_idx = turning_points[i + 1]
        
        gr_start = gr_smooth[start_idx]
        gr_end = gr_smooth[end_idx]
        
        if gr_start > gr_end:
            for j in range(start_idx, min(end_idx + 1, len(gr_values))):
                if gr_start != gr_end:
                    rel_positions[j] = (gr_start - gr_smooth[j]) / (gr_start - gr_end)
                else:
                    rel_positions[j] = 0.5
        else:
            for j in range(start_idx, min(end_idx + 1, len(gr_values))):
                if gr_end != gr_start:
                    rel_positions[j] = 1 - (gr_smooth[j] - gr_start) / (gr_end - gr_start)
                else:
                    rel_positions[j] = 0.5
    
    rel_positions = np.clip(rel_positions, 0, 1)
    df_imputed.loc[well_indices, 'RelPos'] = rel_positions

# 4. Opposition Index (correlation between GR and PHIF in window, with special handling)
print("\nCalculating Opposition Index (GR-PHIF correlation in window)...")
print("  Note: For F-14 where PHIF is NaN, using GR-NPHI correlation instead")
window_size_corr = 1.0  # meters
df_imputed['OppositionIndex'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 3:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    
    # For F-14, use NPHI since PHIF is NaN; for others use PHIF
    if 'F-14' in well:
        porosity_values = well_data['NPHI'].values
        print(f"    Using GR-NPHI correlation for {well} (F-14)")
    else:
        porosity_values = well_data['PHIF'].values
        print(f"    Using GR-PHIF correlation for {well}")
    
    opposition_indices = np.zeros(len(depths))
    
    for i in range(len(depths)):
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_corr / 2
        
        if np.sum(window_mask) >= 3:  # Need at least 3 points for meaningful correlation
            window_gr = gr_values[window_mask]
            window_porosity = porosity_values[window_mask]
            
            if np.std(window_gr) > 0 and np.std(window_porosity) > 0:
                corr, _ = pearsonr(window_gr, window_porosity)
                opposition_indices[i] = corr
            else:
                opposition_indices[i] = 0
        else:
            opposition_indices[i] = 0
    
    df_imputed.loc[well_indices, 'OppositionIndex'] = opposition_indices

# 5. NTG Slope (Net-to-Gross slope across window halves)
print("\nCalculating NTG Slope (Net-to-Gross slope)...")
window_size_ntg = 1.0  # meters
df_imputed['NTG_Slope'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 4:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    vsh_values = well_data['VSH'].values
    ntg_slopes = np.zeros(len(depths))
    
    for i in range(len(depths)):
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_ntg / 2
        
        if np.sum(window_mask) >= 4:  # Need enough points for two halves
            window_depths = depths[window_mask]
            window_vsh = vsh_values[window_mask]
            
            # Split window into top and bottom halves
            mid_idx = len(window_depths) // 2
            
            # Calculate NTG for each half
            ntg_top = np.mean(window_vsh[:mid_idx] < 0.35)
            ntg_bottom = np.mean(window_vsh[mid_idx:] < 0.35)
            
            # Slope is top minus bottom (positive = improving upward)
            ntg_slopes[i] = ntg_top - ntg_bottom
        else:
            ntg_slopes[i] = 0
    
    df_imputed.loc[well_indices, 'NTG_Slope'] = ntg_slopes

# 6. AI Slope (Acoustic Impedance slope)
print("\nCalculating AI Slope (Acoustic Impedance slope)...")
window_size_ai = 1.0  # meters
df_imputed['AI_Slope'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 2:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    # Calculate AI as RHOB * DT (normalized version)
    ai_values = well_data['RHOB'].values * well_data['DT'].values
    ai_slopes = np.zeros(len(depths))
    
    for i in range(len(depths)):
        current_depth = depths[i]
        window_mask = np.abs(depths - current_depth) <= window_size_ai / 2
        
        if np.sum(window_mask) >= 2:
            window_depths = depths[window_mask]
            window_ai = ai_values[window_mask]
            
            if len(window_depths) > 1:
                coeffs = np.polyfit(window_depths, window_ai, 1)
                ai_slopes[i] = coeffs[0]
            else:
                ai_slopes[i] = 0
        else:
            ai_slopes[i] = 0
    
    df_imputed.loc[well_indices, 'AI_Slope'] = ai_slopes

# 7. Base Sharpness (max |GR'| at cycle base)
print("\nCalculating Base Sharpness (max |GR'| at cycle base)...")
window_size_sharp = 1.0  # meters for gradient calculation
df_imputed['BaseSharpness'] = np.nan

for well in wells:
    well_mask = df_imputed['Well'] == well
    well_data = df_imputed[well_mask].copy()
    
    if len(well_data) < 3:
        continue
    
    well_data = well_data.sort_values('Depth')
    well_indices = well_data.index
    
    depths = well_data['Depth'].values
    gr_values = well_data['GR'].values
    
    # Calculate gradient at each point
    gr_gradient = np.zeros(len(depths))
    for i in range(1, len(depths) - 1):
        gr_gradient[i] = (gr_values[i+1] - gr_values[i-1]) / (depths[i+1] - depths[i-1])
    
    # Handle boundaries
    if len(depths) > 1:
        gr_gradient[0] = (gr_values[1] - gr_values[0]) / (depths[1] - depths[0])
        gr_gradient[-1] = (gr_values[-1] - gr_values[-2]) / (depths[-1] - depths[-2])
    
    base_sharpness = np.zeros(len(depths))
    
    # For each point, find the sharpness at the nearest cycle base
    for i in range(len(depths)):
        current_depth = depths[i]
        # Look for local minima (cycle bases) in a window
        window_mask = np.abs(depths - current_depth) <= window_size_sharp
        
        if np.sum(window_mask) >= 3:
            window_gr = gr_values[window_mask]
            window_idx = np.where(window_mask)[0]
            
            # Find local minima in the window
            local_min_idx = []
            for j in range(1, len(window_gr) - 1):
                if window_gr[j] < window_gr[j-1] and window_gr[j] < window_gr[j+1]:
                    local_min_idx.append(window_idx[j])
            
            if local_min_idx:
                # Get maximum gradient magnitude at these base points
                base_gradients = [abs(gr_gradient[idx]) for idx in local_min_idx]
                base_sharpness[i] = max(base_gradients)
            else:
                # If no local minimum, use the point with lowest GR
                min_gr_idx = window_idx[np.argmin(window_gr)]
                base_sharpness[i] = abs(gr_gradient[min_gr_idx])
        else:
            base_sharpness[i] = abs(gr_gradient[i])
    
    df_imputed.loc[well_indices, 'BaseSharpness'] = base_sharpness

# Add all engineered features to feature list
engineered_features = ['GR_Slope', 'GR_Serration', 'RelPos', 'OppositionIndex', 
                      'NTG_Slope', 'AI_Slope', 'BaseSharpness']
lithology_features = [f'Lithology_{i}' for i in [0, 1, 2, 3, 4]]
all_features = base_feature_logs + engineered_features + lithology_features

# Fill any NaN values in engineered features with 0
for feature in engineered_features:
    nan_count = df_imputed[feature].isna().sum()
    if nan_count > 0:
        print(f"\nWarning: {nan_count} NaN values in {feature}. Filling with 0...")
        df_imputed[feature].fillna(0, inplace=True)

# Print engineered feature statistics
print("\n=== Engineered Feature Statistics (before normalization) ===")
for feature in engineered_features:
    values = df_imputed[feature]
    print(f"\n{feature}:")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  Mean: {values.mean():.4f}, Std: {values.std():.4f}")

# Print special handling summary for PHIF and SW
print("\n=== PHIF and SW Summary ===")
f14_mask = df_imputed['Well'].str.contains('F-14', na=False)
f14_phif_nan = df_imputed.loc[f14_mask, 'PHIF'].isna().sum()
f14_sw_nan = df_imputed.loc[f14_mask, 'SW'].isna().sum()
f14_total = f14_mask.sum()

other_phif_zero = df_imputed.loc[~f14_mask, 'PHIF'].eq(0).sum()
other_sw_zero = df_imputed.loc[~f14_mask, 'SW'].eq(0).sum()

print(f"F-14 well: {f14_total} samples")
print(f"  PHIF: {f14_phif_nan} NaN values ({f14_phif_nan/max(f14_total,1)*100:.1f}% of F-14)")
print(f"  SW: {f14_sw_nan} NaN values ({f14_sw_nan/max(f14_total,1)*100:.1f}% of F-14)")
print(f"Other wells:")
print(f"  PHIF: {other_phif_zero} zero values")
print(f"  SW: {other_sw_zero} zero values")

# Step 9: Global normalization using StandardScaler (features only, not labels)
# Handle NaN values in PHIF/SW for F-14 during normalization
print("\nStep 9: Applying global normalization to all features...")
print("  Note: F-14 PHIF/SW NaNs will be handled specially during normalization")

scaler = StandardScaler()
df_normalized = df_imputed.copy()

# Normalize features, handling NaNs properly
for feature in all_features:
    if feature in ['PHIF', 'SW'] or feature.startswith('Lithology_'):
        # For PHIF/SW and lithology columns, fit scaler only on non-NaN values, then transform
        non_nan_mask = df_imputed[feature].notna()
        if non_nan_mask.sum() > 0:  # Only if there are non-NaN values
            feature_scaler = StandardScaler()
            df_normalized.loc[non_nan_mask, feature] = feature_scaler.fit_transform(
                df_imputed.loc[non_nan_mask, feature].values.reshape(-1, 1)
            ).flatten()
        # NaN values remain NaN after normalization
    else:
        # Standard normalization for other features
        feature_scaler = StandardScaler()
        df_normalized[feature] = feature_scaler.fit_transform(
            df_imputed[feature].values.reshape(-1, 1)
        ).flatten()

# Final dataset
df_final = df_normalized.copy()
stats['final_rows'] = len(df_final)

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Initial rows: {stats['initial_rows']}")
print(f"After removing empty core values: {stats['after_empty_removal']} (dropped: {stats['initial_rows'] - stats['after_empty_removal']})")
print(f"After removing zeros in core features: {stats['after_zero_removal']} (dropped: {stats['after_empty_removal'] - stats['after_zero_removal']})")
print(f"After removing unrealistic core values: {stats['after_unrealistic_removal']} (dropped: {stats['after_zero_removal'] - stats['after_unrealistic_removal']})")
print(f"After imputation: {stats['after_imputation']} (dropped: {stats['after_unrealistic_removal'] - stats['after_imputation']})")
print(f"Facies labels: NOT dropped (keeping all rows)")
print(f"Lithology labels: NOT dropped (keeping all rows)")
print(f"Final rows: {stats['final_rows']}")
print(f"Total rows dropped: {stats['initial_rows'] - stats['final_rows']} ({(stats['initial_rows'] - stats['final_rows'])/stats['initial_rows']*100:.2f}%)")
print(f"Total rows retained: {stats['final_rows']} ({stats['final_rows']/stats['initial_rows']*100:.2f}%)")

# Save the ML-ready dataset
output_filename = 'ML_ready_10_base_plus_engineered_F14_special_litho.csv'
df_final.to_csv(output_filename, index=False)
print(f"\nML-ready dataset saved as: {output_filename}")

# Print sample of the final dataset
print("\nSample of final dataset:")
print(df_final.head())

# Print normalized value ranges (excluding NaNs)
print("\nNormalized value ranges (all features, excluding NaNs):")
for col in all_features:
    valid_values = df_final[col].dropna()
    if len(valid_values) > 0:
        print(f"{col}: min={valid_values.min():.3f}, max={valid_values.max():.3f}, mean={valid_values.mean():.3f}, std={valid_values.std():.3f}")
    else:
        print(f"{col}: All NaN values")

# Print feature summary
print(f"\n=== FEATURE SUMMARY ===")
print(f"Total features: {len(all_features)} ({len(base_feature_logs)} base + {len(engineered_features)} engineered + {len(lithology_features)} lithology)")
print(f"Base features (10): {base_feature_logs}")
print(f"Engineered features (7): {engineered_features}")
print(f"Lithology features (5): {lithology_features}")
print(f"Special handling: F-14 PHIF/SW set to NaN, OppositionIndex uses GR-NPHI for F-14")

# Facies distribution (if facies column exists)
if 'Equinor facies' in df_final.columns:
    print("\nFacies distribution in final dataset:")
    facies_counts = df_final['Equinor facies'].value_counts().sort_index()
    for facies, count in facies_counts.items():
        print(f"  {int(facies)}: {count} samples ({count/len(df_final)*100:.1f}%)")
    # Show NaN count
    nan_facies = df_final['Equinor facies'].isna().sum()
    if nan_facies > 0:
        print(f"  NaN: {nan_facies} samples ({nan_facies/len(df_final)*100:.1f}%)")

# Lithology distribution (raw values, plus one-hot encoding)
if 'Lithology' in df_final.columns:
    print("\nLithology distribution in final dataset (raw values + one-hot encoded):")
    litho_counts = df_final['Lithology'].value_counts().sort_index()
    for litho, count in litho_counts.items():
        print(f"  {int(litho)}: {count} samples ({count/len(df_final)*100:.1f}%)")
    # Show NaN count
    nan_litho = df_final['Lithology'].isna().sum()
    if nan_litho > 0:
        print(f"  NaN: {nan_litho} samples ({nan_litho/len(df_final)*100:.1f}%) - one-hot columns set to NaN")

# Wells and their sample counts
print("\nWells in final dataset:")
well_counts = df_final['Well'].value_counts().sort_index()
for well, count in well_counts.items():
    print(f"  {well}: {count} samples")

# Print correlation matrix for engineered features
print("\n=== Engineered Features Correlation Matrix ===")
eng_corr = df_final[engineered_features].corr()
print(eng_corr.round(3))

print("\n" + "="*60)
print("DATASET CREATION COMPLETE")
print("="*60)
print(f"Features: {len(all_features)} total ({len(base_feature_logs)} base + {len(engineered_features)} engineered + {len(lithology_features)} lithology)")
print(f"Special F-14 handling: PHIF and SW kept as NaN")
print(f"OppositionIndex: GR-PHIF for other wells, GR-NPHI for F-14")
print(f"NO facies/lithology filtering: All rows preserved")
print(f"Lithology one-hot encoding: ENABLED (Lithology_0 to Lithology_4)")
print(f"Output: {output_filename}")
print("="*60)
