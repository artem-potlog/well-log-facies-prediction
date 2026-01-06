"""
Feature Combination Definitions for V5 Multi-Scenario Approach - FINAL VERSION

Progressive strategy based on Phase 3 + Hypothesis Test results:
1. Establish baseline (FC_01 → FC_03)
2. Add lithology incrementally
3. Add engineered features incrementally
4. Add water/fluid features ONE AT A TIME to isolate problems
5. Test optimal combinations

Total: 18 feature combinations
"""

# FINAL Feature Combinations - Progressive Testing Strategy
FEATURE_COMBINATIONS = {
    # ============================================================================
    # GROUP 1: BASELINE PROGRESSION (3 combos)
    # ============================================================================
    
    'FC_01_rock_only': {
        'features': ['NPHI', 'RHOB', 'DT'],
        'description': 'Rock Physics Only (3 features)',
        'purpose': 'Minimal baseline',
        'result': '0.318 (custom_sand)',
        'category': 'baseline'
    },
    
    'FC_02_rock_clean': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT'],
        'description': 'Rock + Cleanliness (5 features)',
        'purpose': 'Add energy discrimination',
        'result': '0.390 (custom_sand)',
        'category': 'baseline'
    },
    
    'FC_03_add_perm': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH'],
        'description': 'Add Permeability (7 features)',
        'purpose': 'Clean baseline with reservoir quality indicators',
        'result': '0.395 (custom_sand) - Clean baseline winner',
        'category': 'baseline'
    },
    
    # ============================================================================
    # GROUP 2: ADD LITHOLOGY PROGRESSIVELY (3 combos)
    # ============================================================================
    
    'FC_04_fc3_lith': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4'],
        'description': 'FC_03 + Lithology (12 features)',
        'purpose': 'Test lithology impact on clean baseline',
        'result': '0.387 (custom_sand)',
        'category': 'lithology'
    },
    
    'FC_05_fc3_lith_relpos': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos'],
        'description': 'FC_03 + Lithology + RelPos (13 features)',
        'purpose': 'Add stratigraphic position context',
        'result': '0.389 (custom_sand)',
        'category': 'lithology'
    },
    
    'FC_06_fc3_lith_relpos_grtex': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration'],
        'description': 'FC_03 + Lith + RelPos + GR Texture (15 features)',
        'purpose': 'OPTIMAL CLEAN MODEL',
        'result': '0.400 (custom_sand) - WINNER!',
        'category': 'lithology'
    },
    
    # ============================================================================
    # GROUP 3: ADD ENGINEERED FEATURES PROGRESSIVELY (3 combos)
    # ============================================================================
    
    'FC_07_fc6_opposition': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'OppositionIndex'],
        'description': 'FC_06 + Opposition Index (16 features)',
        'purpose': 'Test opposition index (coarsening vs fining)',
        'category': 'engineered'
    },
    
    'FC_08_fc6_ntg': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'OppositionIndex', 'NTG_Slope'],
        'description': 'FC_07 + NTG Slope (17 features)',
        'purpose': 'Test net-to-gross trend impact',
        'category': 'engineered'
    },
    
    'FC_09_fc6_all_eng': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'OppositionIndex', 'NTG_Slope', 'BaseSharpness'],
        'description': 'FC_06 + All Engineered (18 features)',
        'purpose': 'Test all stratigraphic engineered features (no fluid)',
        'category': 'engineered'
    },
    
    # ============================================================================
    # GROUP 4: ADD WATER FEATURES ONE AT A TIME (3 combos)
    # ============================================================================
    
    'FC_10_fc3_rt': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'RT'],
        'description': 'FC_03 + RT only (8 features)',
        'purpose': 'ISOLATE RT impact - is resistivity the problem?',
        'category': 'fluid_test'
    },
    
    'FC_11_fc3_rt_rd': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'RT', 'RD'],
        'description': 'FC_03 + RT + RD (9 features)',
        'purpose': 'ISOLATE RD impact - does deep resistivity worsen it?',
        'category': 'fluid_test'
    },
    
    'FC_12_fc3_rt_rd_sw': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'RT', 'RD', 'SW'],
        'description': 'FC_03 + All Fluid (10 features)',
        'purpose': 'ISOLATE SW impact - reproduce FC_04 crash (-19% expected)',
        'category': 'fluid_test'
    },
    
    # ============================================================================
    # GROUP 5: OPTIMAL + FLUID (Test if fluid helps strong foundation) (3 combos)
    # ============================================================================
    
    'FC_13_fc6_rt': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'RT'],
        'description': 'FC_06 (winner) + RT only (16 features)',
        'purpose': 'Can RT help the optimal model?',
        'category': 'optimal_plus_fluid'
    },
    
    'FC_14_fc6_rt_rd': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'RT', 'RD'],
        'description': 'FC_06 + RT + RD (17 features)',
        'purpose': 'Test resistivity pair on strong foundation',
        'category': 'optimal_plus_fluid'
    },
    
    'FC_15_fc6_all_fluid': {
        'features': ['GR', 'VSH', 'NPHI', 'RHOB', 'DT', 'PHIF', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'RT', 'RD', 'SW'],
        'description': 'FC_06 + All Fluid (18 features)',
        'purpose': 'Maximum clean features + fluid - can it work?',
        'category': 'optimal_plus_fluid'
    },
    
    # ============================================================================
    # GROUP 6: SPECIAL CASES (3 combos)
    # ============================================================================
    
    'FC_16_measured_only': {
        'features': ['GR', 'RT', 'NPHI', 'RHOB', 'RD', 'DT',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'OppositionIndex', 'BaseSharpness'],
        'description': 'Measured Logs Only (11 features)',
        'purpose': 'For F-14 well (no PHIF/SW/KLOGH/VSH available)',
        'result': '0.261 (custom_sand) - Poor due to missing key features',
        'category': 'special'
    },
    
    'FC_17_max_features': {
        'features': ['GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH',
                     'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
                     'RelPos', 'GR_Slope', 'GR_Serration',
                     'OppositionIndex', 'NTG_Slope', 'BaseSharpness'],
        'description': 'Maximum Features (21 features)',
        'purpose': 'Test everything together (reference)',
        'result': '0.327 (custom_sand) - Overfitting',
        'category': 'special'
    },
    
    'FC_18_tidal_optimized': {
        'features': ['GR', 'RT', 'NPHI', 'RHOB', 'VSH', 'PHIF', 'SW',
                     'GR_Slope', 'GR_Serration', 'RelPos', 'BaseSharpness',
                     'OppositionIndex', 'NTG_Slope'],
        'description': 'Domain Expert Curated (13 features)',
        'purpose': 'Test domain knowledge vs data-driven',
        'result': '0.313 (custom_sand) - Data-driven wins',
        'category': 'special'
    },
    
    # ============================================================================
    # FORWARD SELECTION WINNER (Data-driven optimal)
    # ============================================================================
    
    'FC_19_forward_optimal': {
        'features': ['VSH', 'RelPos',
                     'Lithology_3', 'Lithology_0', 'OppositionIndex', 'NTG_Slope',
                     'Lithology_1', 'Lithology_4'],
        'description': 'Forward Selection Optimal (8 features)',
        'purpose': 'BEST DISCOVERED - Data-driven feature selection winner',
        'expected_accuracy': '48%',
        'forward_result': '0.480 (custom_sand) - Beats all manual designs',
        'category': 'forward_selection',
        'key_insight': 'VSH + 4 Lithology + 2 Stratigraphic trends = optimal'
    }
}


def get_feature_list(fc_name):
    """Get list of features for a given feature combination"""
    if fc_name not in FEATURE_COMBINATIONS:
        raise ValueError(f"Unknown feature combination: {fc_name}")
    return FEATURE_COMBINATIONS[fc_name]['features']


def get_all_feature_names():
    """Get list of all feature combination names"""
    return list(FEATURE_COMBINATIONS.keys())


def get_by_category(category):
    """Get feature combinations by category"""
    return [name for name, info in FEATURE_COMBINATIONS.items() 
            if info.get('category') == category]


def print_feature_summary():
    """Print summary of all feature combinations"""
    print("\n" + "="*80)
    print("FEATURE COMBINATIONS - FINAL PROGRESSIVE STRATEGY")
    print("Total: 19 combinations (18 manual + 1 forward selection)")
    print("="*80)
    
    categories = [
        ('baseline', 'GROUP 1: BASELINE PROGRESSION'),
        ('lithology', 'GROUP 2: ADD LITHOLOGY + ENGINEERED'),
        ('engineered', 'GROUP 3: ADD MORE ENGINEERED FEATURES'),
        ('fluid_test', 'GROUP 4: ADD WATER FEATURES (ONE AT A TIME)'),
        ('optimal_plus_fluid', 'GROUP 5: OPTIMAL + FLUID (Strong Foundation Test)'),
        ('forward_selection', 'GROUP 6: FORWARD SELECTION WINNER'),
        ('special', 'GROUP 7: SPECIAL CASES')
    ]
    
    for cat_key, cat_title in categories:
        combos = get_by_category(cat_key)
        if combos:
            print("\n" + "="*80)
            print(cat_title)
            print("="*80)
            
            for fc_name in combos:
                fc_info = FEATURE_COMBINATIONS[fc_name]
                n_features = len(fc_info['features'])
                result = fc_info.get('result', 'Not tested')
                print(f"\n{fc_name}:")
                print(f"  Description: {fc_info['description']}")
                print(f"  Features: {n_features}")
                print(f"  Purpose: {fc_info['purpose']}")
                if result != 'Not tested':
                    print(f"  Result: {result}")
    
    print("\n" + "="*80)
    print("TESTING STRATEGY:")
    print("="*80)
    print("Phase 1: Baseline (FC_01-03) - Establish foundation")
    print("Phase 2: Lithology (FC_04-06) - Test lithology value")
    print("Phase 3: Engineered (FC_07-09) - Test additional features")
    print("Phase 4: Fluid isolation (FC_10-12) - Find the culprit")
    print("Phase 5: Integration (FC_13-15) - Can fluid help optimal model?")
    print("Phase 6: Forward selection (FC_19) - Data-driven optimal")
    print("Phase 7: Special cases (FC_16-18) - Reference comparisons")
    print("\nTOTAL: 19 feature combos × 3 metrics = 57 models")
    print("\nRECOMMENDED FOR PRODUCTION:")
    print("  1. FC_19 (8 features) - Forward selection winner: 0.480")
    print("  2. FC_13 (16 features) - Manual design winner: 0.392")
    print("  3. Ensemble of top 10 models")
    print("="*80)


if __name__ == '__main__':
    print_feature_summary()
