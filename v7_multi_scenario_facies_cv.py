"""
Random Forest Classification - Version 7: Intelligent Feature Selection with Optuna

Revolutionary approach: Instead of testing predefined feature combinations,
use hyperparameter optimization to intelligently search the feature space.

Key Innovation:
- Feature subset = hyperparameter (21 binary decisions)
- Optuna/TPE samples intelligently (not exhaustively)
- Finds 30-40 best combinations from 2^21 = 2,097,152 possibilities
- Adds feature count penalty to prefer simpler models

Approach:
1. Encode feature subset as 21 booleans (include/exclude)
2. Objective = CV score with feature penalty
3. Run Optuna TPE optimization
4. Store top 30-40 unique feature masks

Benefits over V6:
- Explores 2+ million possibilities efficiently
- Data-driven feature discovery
- Avoids human bias in feature selection
- Finds optimal trade-off between performance and simplicity

Author: Based on V6, with Optuna optimization
Date: 2026-01-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json
import joblib
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import optuna
from optuna.samplers import TPESampler

# Import local config modules
import sys
sys.path.append('config')
from optimization_metrics import METRICS, custom_sand_weighted_f1

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

# Well naming (for display)
WELL_NAMES = {
    '15_9-F-1 B': 'Well A (F-1B)',
    '15_9-F-1 C': 'Well B (F-1C)',
    '15_9-F-4': 'Well C (F-4)',
    '15_9-F-11 A': 'Well D (F-11A)',
    '15_9-F-11 B': 'Well E (F-11B)',
    '15_9-F-14': 'Well F-14'
}

# ALL AVAILABLE FEATURES (21 total from FC_17_max_features)
ALL_FEATURES = [
    'GR', 'VSH', 'RT', 'RD', 'NPHI', 'RHOB', 'DT', 'PHIF', 'SW', 'KLOGH',
    'Lithology_0', 'Lithology_1', 'Lithology_2', 'Lithology_3', 'Lithology_4',
    'RelPos', 'GR_Slope', 'GR_Serration',
    'OppositionIndex', 'NTG_Slope', 'BaseSharpness'
]


class V7OptunaFeatureSelector:
    """
    Intelligent Feature Selection using Optuna
    
    Treats feature subset as hyperparameter:
    - 21 binary decisions (include/exclude each feature)
    - Objective = CV score - feature_penalty
    - TPE sampler explores space intelligently
    - Finds 30-40 best combinations
    """
    
    def __init__(self, dataset_path, metric='custom_sand', output_dir='v7_output'):
        """
        Initialize V7 Optuna feature selector
        
        Parameters:
        -----------
        dataset_path : str
            Path to ML-ready dataset
        metric : str
            Optimization metric: 'custom_sand' or 'f1_weighted'
        output_dir : str
            Output directory for results
        """
        self.dataset_path = dataset_path
        self.metric = metric
        self.output_dir = Path(output_dir)
        
        # Data containers
        self.df = None
        self.df_train = None
        self.train_wells = None
        self.test_wells = None
        self.groups_train = None
        self.n_splits = 4
        
        # Results containers
        self.study = None
        self.best_combinations = []
        self.trained_models = {}
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir,
            self.output_dir / 'models',
            self.output_dir / 'results',
            self.output_dir / 'predictions',
            self.output_dir / 'visualizations',
            self.output_dir / 'optuna_history'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directories created: {self.output_dir}")
    
    def load_data(self):
        """Load and prepare dataset"""
        print("\n" + "="*80)
        print("V7 INTELLIGENT FEATURE SELECTION WITH OPTUNA")
        print("Exploring 2^21 = 2,097,152 feature combinations intelligently")
        print("="*80)
        
        print(f"\nLoading dataset: {self.dataset_path}")
        
        try:
            self.df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Filter to samples with facies labels
        df_with_facies = self.df[self.df['Equinor facies'].notna()].copy()
        print(f"Samples with facies: {len(df_with_facies)} / {len(self.df)}")
        
        # Rename target column
        df_with_facies['Facies'] = df_with_facies['Equinor facies'].astype(int)
        
        self.df = df_with_facies
        
        # Show well distribution
        print("\nWell distribution:")
        well_counts = self.df['Well'].value_counts().sort_index()
        for well, count in well_counts.items():
            well_display = WELL_NAMES.get(well, well)
            print(f"  {well_display}: {count} samples")
        
        return self.df
    
    def configure_wells(self, train_wells=None, test_wells=None, interactive=True):
        """Configure training and test wells"""
        available_wells = sorted(self.df['Well'].unique())
        
        if interactive and train_wells is None:
            print("\n" + "="*80)
            print("WELL SELECTION")
            print("="*80)
            print(f"\nAvailable wells ({len(available_wells)}):")
            for i, well in enumerate(available_wells, 1):
                count = len(self.df[self.df['Well'] == well])
                print(f"  {i}. {well} ({count} samples)")
            
            print("\n" + "-"*80)
            print("SELECT TRAINING WELLS (for GroupKFold CV)")
            print("Recommended: 4-6 wells with most complete data")
            print("Enter comma-separated numbers (e.g., 1,2,3,4):")
            print("-"*80)
            
            train_input = input("Training wells: ").strip()
            train_indices = [int(x.strip()) - 1 for x in train_input.split(',') if x.strip().isdigit()]
            self.train_wells = [available_wells[i] for i in train_indices if 0 <= i < len(available_wells)]
            
            print("\n" + "-"*80)
            print("SELECT TEST WELLS (blind prediction)")
            print("Enter comma-separated numbers (e.g., 5,6):")
            print("-"*80)
            
            test_input = input("Test wells: ").strip()
            test_indices = [int(x.strip()) - 1 for x in test_input.split(',') if x.strip().isdigit()]
            self.test_wells = [available_wells[i] for i in test_indices if 0 <= i < len(available_wells)]
        
        else:
            if train_wells is not None:
                self.train_wells = train_wells
            else:
                self.train_wells = available_wells[:min(4, len(available_wells)-1)]
            
            if test_wells is not None:
                self.test_wells = test_wells
            else:
                self.test_wells = [w for w in available_wells if w not in self.train_wells]
        
        # Prepare training data
        self.df_train = self.df[self.df['Well'].isin(self.train_wells)].copy()
        self.groups_train = self.df_train['Well'].values
        self.n_splits = min(4, len(self.train_wells))
        
        # Display configuration
        print("\n" + "-"*80)
        print("WELL CONFIGURATION")
        print("-"*80)
        print(f"Training wells (GroupKFold CV with k={self.n_splits}):")
        for well in self.train_wells:
            count = len(self.df[self.df['Well'] == well])
            print(f"  - {well} ({count} samples)")
        
        print(f"\nTest wells (blind prediction):")
        for well in self.test_wells:
            count = len(self.df[self.df['Well'] == well])
            print(f"  - {well} ({count} samples)")
        
        print(f"\nTotal training samples: {len(self.df_train)}")
        print(f"Total test samples: {len(self.df[self.df['Well'].isin(self.test_wells)])}")
        print("-"*80)
    
    def _objective(self, trial):
        """
        Optuna objective function
        
        For each trial:
        1. Sample a feature subset (21 binary decisions)
        2. Train RF with CV
        3. Return score - feature_penalty
        """
        # Sample feature mask (21 booleans)
        feature_mask = []
        selected_features = []
        
        for feature in ALL_FEATURES:
            # Check if feature is available
            if feature not in self.df_train.columns:
                feature_mask.append(False)
                continue
            
            # Let Optuna decide: include or exclude
            include = trial.suggest_categorical(f'use_{feature}', [True, False])
            feature_mask.append(include)
            
            if include:
                selected_features.append(feature)
        
        # Require minimum 3 features
        if len(selected_features) < 3:
            return -1.0  # Invalid trial
        
        # Prepare data with selected features
        X_train = self.df_train[selected_features].copy()
        y_train = self.df_train['Facies'].values
        
        # Create pipeline
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=150,  # Fixed for speed
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            ))
        ])
        
        # Cross-validation
        gkf = GroupKFold(n_splits=self.n_splits)
        
        # Select scorer
        if self.metric == 'custom_sand':
            from sklearn.metrics import make_scorer
            scorer = make_scorer(custom_sand_weighted_f1, greater_is_better=True)
        else:  # f1_weighted
            from sklearn.metrics import make_scorer
            scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        # Evaluate
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                groups=self.groups_train,
                cv=gkf,
                scoring=scorer,
                n_jobs=1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            print(f"  ⚠ Trial failed: {e}")
            return -1.0
        
        # Feature penalty: encourage simpler models
        # Penalty = 0.01 * (n_features / 21)
        n_features = len(selected_features)
        feature_penalty = 0.01 * (n_features / 21.0)
        
        # Objective = CV score - penalty
        objective_value = cv_mean - feature_penalty
        
        # Store feature info in trial
        trial.set_user_attr('n_features', n_features)
        trial.set_user_attr('features', selected_features)
        trial.set_user_attr('cv_mean', float(cv_mean))
        trial.set_user_attr('cv_std', float(cv_std))
        trial.set_user_attr('feature_penalty', float(feature_penalty))
        
        return objective_value
    
    def run_optimization(self, n_trials=100, n_jobs=1):
        """
        Run Optuna optimization to find best feature combinations
        
        Parameters:
        -----------
        n_trials : int
            Number of trials to run (default: 100)
        n_jobs : int
            Number of parallel jobs (1 = sequential)
        """
        print("\n" + "="*80)
        print("STARTING OPTUNA OPTIMIZATION")
        print("="*80)
        print(f"Search space: 2^21 = 2,097,152 feature combinations")
        print(f"Trials: {n_trials}")
        print(f"Sampler: TPE (Tree-structured Parzen Estimator)")
        print(f"Metric: {self.metric}")
        print(f"CV folds: {self.n_splits}")
        print("="*80)
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'v7_feature_selection_{self.metric}'
        )
        
        # Run optimization
        start_time = datetime.now()
        
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"Total time: {elapsed:.1f} minutes")
        print(f"Trials completed: {len(self.study.trials)}")
        print(f"Best value: {self.study.best_value:.4f}")
        print(f"Best features ({len(self.study.best_trial.user_attrs['features'])}):")
        for feat in self.study.best_trial.user_attrs['features']:
            print(f"  - {feat}")
        print("="*80)
        
        # Save study
        study_path = self.output_dir / 'optuna_history' / 'study.pkl'
        joblib.dump(self.study, study_path)
        print(f"\n✓ Study saved to: {study_path}")
        
        # Extract top combinations
        self._extract_top_combinations()
        
        return self.study
    
    def _extract_top_combinations(self, top_n=40):
        """Extract top N unique feature combinations"""
        print(f"\nExtracting top {top_n} unique feature combinations...")
        
        # Sort trials by objective value
        valid_trials = [t for t in self.study.trials if t.value > 0]
        sorted_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)
        
        # Extract unique combinations
        seen_combinations = set()
        self.best_combinations = []
        
        for rank, trial in enumerate(sorted_trials, 1):
            features = trial.user_attrs['features']
            features_tuple = tuple(sorted(features))  # Make hashable
            
            # Skip duplicates
            if features_tuple in seen_combinations:
                continue
            
            seen_combinations.add(features_tuple)
            
            combo_info = {
                'rank': len(self.best_combinations) + 1,
                'trial_number': trial.number,
                'features': features,
                'n_features': trial.user_attrs['n_features'],
                'cv_mean': trial.user_attrs['cv_mean'],
                'cv_std': trial.user_attrs['cv_std'],
                'feature_penalty': trial.user_attrs['feature_penalty'],
                'objective_value': trial.value
            }
            
            self.best_combinations.append(combo_info)
            
            if len(self.best_combinations) >= top_n:
                break
        
        print(f"✓ Extracted {len(self.best_combinations)} unique combinations")
        
        # Save to CSV
        combos_df = pd.DataFrame(self.best_combinations)
        combos_path = self.output_dir / 'results' / 'best_feature_combinations.csv'
        combos_df.to_csv(combos_path, index=False)
        print(f"✓ Saved to: {combos_path}")
        
        # Print top 10
        print("\n" + "="*80)
        print("TOP 10 FEATURE COMBINATIONS")
        print("="*80)
        for combo in self.best_combinations[:10]:
            print(f"\nRank {combo['rank']}: Trial #{combo['trial_number']}")
            print(f"  Features: {combo['n_features']}")
            print(f"  CV Score: {combo['cv_mean']:.4f} ± {combo['cv_std']:.4f}")
            print(f"  Objective: {combo['objective_value']:.4f}")
            print(f"  Top features: {', '.join(combo['features'][:5])}{'...' if len(combo['features']) > 5 else ''}")
        print("="*80)
        
        return self.best_combinations
    
    def train_top_models(self, top_n=30, use_gridsearch=False):
        """
        Train final models for top N combinations with full hyperparameter tuning
        
        Parameters:
        -----------
        top_n : int
            Number of top combinations to train
        use_gridsearch : bool
            If True, run GridSearchCV for each. If False, use fixed params.
        """
        print("\n" + "="*80)
        print(f"TRAINING TOP {top_n} MODELS")
        print("="*80)
        
        if use_gridsearch:
            print("Mode: GridSearchCV (slow but thorough)")
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [15, 20, 25],
                'classifier__min_samples_split': [5, 8],
                'classifier__min_samples_leaf': [2, 3, 5]
            }
        else:
            print("Mode: Fixed parameters (fast)")
        
        # Select metric scorer
        if self.metric == 'custom_sand':
            from sklearn.metrics import make_scorer
            scorer = make_scorer(custom_sand_weighted_f1, greater_is_better=True)
        else:
            from sklearn.metrics import make_scorer
            scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        
        start_time = datetime.now()
        
        for i, combo in enumerate(self.best_combinations[:top_n], 1):
            print(f"\n[{i}/{top_n}] Training rank {combo['rank']}...")
            print(f"  Features ({combo['n_features']}): {', '.join(combo['features'][:3])}...")
            
            features = combo['features']
            X_train = self.df_train[features].copy()
            y_train = self.df_train['Facies'].values
            
            # Create pipeline
            pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ))
            ])
            
            if use_gridsearch:
                # Full hyperparameter search
                gkf = GroupKFold(n_splits=self.n_splits)
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=gkf,
                    n_jobs=1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train, groups=self.groups_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                # Fixed parameters
                pipeline.set_params(
                    classifier__n_estimators=200,
                    classifier__max_depth=20,
                    classifier__min_samples_split=5,
                    classifier__min_samples_leaf=3
                )
                pipeline.fit(X_train, y_train)
                best_model = pipeline
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3
                }
            
            # Save model
            model_id = f"rank_{combo['rank']:02d}_trial_{combo['trial_number']}"
            model_dir = self.output_dir / 'models' / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(best_model, model_dir / 'model.pkl', compress=3)
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'rank': combo['rank'],
                'trial_number': combo['trial_number'],
                'features': features,
                'n_features': combo['n_features'],
                'cv_mean': combo['cv_mean'],
                'cv_std': combo['cv_std'],
                'objective_value': combo['objective_value'],
                'best_params': best_params,
                'metric': self.metric,
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(model_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Feature importance
            rf_classifier = best_model.named_steps['classifier']
            fi_df = pd.DataFrame({
                'feature': features,
                'importance': rf_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            fi_df.to_csv(model_dir / 'feature_importance.csv', index=False)
            
            self.trained_models[model_id] = best_model
            
            print(f"  ✓ Saved to: {model_dir.name}")
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n✓ Training complete! Time: {elapsed:.1f} minutes")
    
    def predict_ensemble(self, target_well, voting_strategy='equal', top_n=None):
        """
        Generate ensemble predictions for target well
        
        Parameters:
        -----------
        target_well : str
            Well name
        voting_strategy : str
            'equal' or 'weighted' (by CV score)
        top_n : int or None
            Use only top N models
        """
        print("\n" + "="*80)
        print(f"ENSEMBLE PREDICTION: {WELL_NAMES.get(target_well, target_well)}")
        print("="*80)
        
        # Load target well data
        df_target = self.df[self.df['Well'] == target_well].copy()
        
        if len(df_target) == 0:
            print(f"Error: No data found for well {target_well}")
            return None
        
        # Get depth column
        if 'DEPTH_MD' in df_target.columns:
            depth = df_target['DEPTH_MD'].values
        elif 'Depth' in df_target.columns:
            depth = df_target['Depth'].values
        else:
            depth = np.arange(len(df_target))
        
        facies_true = df_target['Facies'].values if 'Facies' in df_target.columns else None
        
        print(f"Samples: {len(df_target)}")
        
        # Select models
        models_to_use = list(self.trained_models.keys())
        if top_n is not None:
            models_to_use = models_to_use[:top_n]
        
        print(f"Using {len(models_to_use)} models")
        
        # Collect predictions
        all_probs = []
        model_scores = []
        
        for model_id in models_to_use:
            model = self.trained_models[model_id]
            
            # Load metadata
            model_dir = self.output_dir / 'models' / model_id
            with open(model_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            features = metadata['features']
            
            # Check features
            missing = [f for f in features if f not in df_target.columns]
            if missing:
                print(f"⚠ Skipping {model_id} - missing features")
                continue
            
            # Predict
            X_target = df_target[features]
            probs = model.predict_proba(X_target)
            
            all_probs.append(probs)
            model_scores.append(metadata['cv_mean'])
        
        print(f"✓ Collected predictions from {len(all_probs)} models")
        
        # Stack predictions
        all_probs = np.stack(all_probs, axis=0)
        
        # Ensemble
        if voting_strategy == 'equal':
            avg_probs = np.mean(all_probs, axis=0)
        else:  # weighted
            weights = np.array(model_scores) / np.sum(model_scores)
            avg_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Predictions
        pred_facies = np.argmax(avg_probs, axis=1)
        
        # Uncertainty
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=1)
        
        pred_facies_per_model = np.argmax(all_probs, axis=2)
        agreement = np.array([
            np.sum(pred_facies_per_model[:, i] == pred_facies[i]) / len(all_probs)
            for i in range(len(pred_facies))
        ])
        
        sorted_probs = np.sort(avg_probs, axis=1)
        prob_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'DEPTH_MD': depth,
            'Facies_Predicted': pred_facies,
        })
        
        if facies_true is not None:
            results_df['Facies_True'] = facies_true
            accuracy = np.mean(pred_facies == facies_true)
            print(f"\n✓ Ensemble Accuracy: {accuracy:.4f}")
        
        # Add probabilities
        n_classes = avg_probs.shape[1]
        for i in range(n_classes):
            results_df[f'Prob_Class_{i}'] = avg_probs[:, i]
        
        for i in range(n_classes, 9):
            results_df[f'Prob_Class_{i}'] = 0.0
        
        # Add uncertainty
        results_df['Uncertainty_Entropy'] = entropy
        results_df['Uncertainty_Agreement'] = agreement
        results_df['Uncertainty_Margin'] = prob_margin
        
        # Save
        well_name_safe = target_well.replace('/', '_').replace('\\', '_')
        output_dir = self.output_dir / 'predictions' / well_name_safe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)
        print(f"✓ Saved to: {output_dir}")
        
        return results_df, all_probs
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        viz_dir = self.output_dir / 'visualizations'
        
        # 1. Optimization history
        self._plot_optimization_history(viz_dir)
        
        # 2. Feature importance analysis
        self._plot_feature_importance_analysis(viz_dir)
        
        # 3. Model comparison
        self._plot_model_comparison(viz_dir)
        
        print("✓ All visualizations created!")
    
    def _plot_optimization_history(self, viz_dir):
        """Plot Optuna optimization history"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract trial data
        trials_df = self.study.trials_dataframe()
        
        # Panel 1: Objective value over trials
        ax = axes[0, 0]
        ax.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6)
        ax.axhline(self.study.best_value, color='red', linestyle='--', label=f'Best: {self.study.best_value:.4f}')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Feature count distribution
        ax = axes[0, 1]
        n_features = [t.user_attrs.get('n_features', 0) for t in self.study.trials if t.value > 0]
        ax.hist(n_features, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(n_features), color='red', linestyle='--', label=f'Mean: {np.mean(n_features):.1f}')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Count')
        ax.set_title('Feature Count Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: CV score vs feature count
        ax = axes[1, 0]
        cv_scores = [t.user_attrs.get('cv_mean', 0) for t in self.study.trials if t.value > 0]
        ax.scatter(n_features, cv_scores, alpha=0.5, s=50)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('CV Score')
        ax.set_title('CV Score vs Feature Count')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Top features frequency
        ax = axes[1, 1]
        feature_counts = {}
        for combo in self.best_combinations[:20]:
            for feat in combo['features']:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        features, counts = zip(*sorted_features) if sorted_features else ([], [])
        
        ax.barh(range(len(features)), counts, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Frequency in Top 20')
        ax.set_title('Most Important Features')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        plt.suptitle('V7 Optuna Optimization Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / 'v7_optimization_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def _plot_feature_importance_analysis(self, viz_dir):
        """Aggregate feature importance across top models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Collect feature importances
        feature_importance_sum = {}
        feature_importance_count = {}
        
        for model_id in list(self.trained_models.keys())[:20]:
            model_dir = self.output_dir / 'models' / model_id
            fi_path = model_dir / 'feature_importance.csv'
            
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path)
                for _, row in fi_df.iterrows():
                    feat = row['feature']
                    importance = row['importance']
                    feature_importance_sum[feat] = feature_importance_sum.get(feat, 0) + importance
                    feature_importance_count[feat] = feature_importance_count.get(feat, 0) + 1
        
        # Average importance
        feature_importance_avg = {
            feat: feature_importance_sum[feat] / feature_importance_count[feat]
            for feat in feature_importance_sum
        }
        
        sorted_features = sorted(feature_importance_avg.items(), key=lambda x: x[1], reverse=True)
        
        # Panel 1: Average importance
        ax = axes[0]
        features, importances = zip(*sorted_features[:15]) if sorted_features else ([], [])
        ax.barh(range(len(features)), importances, alpha=0.7, color='coral')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Average Importance')
        ax.set_title('Feature Importance (Top 20 Models Average)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        # Panel 2: Feature selection frequency
        ax = axes[1]
        feature_frequency = {}
        for combo in self.best_combinations[:30]:
            for feat in combo['features']:
                feature_frequency[feat] = feature_frequency.get(feat, 0) + 1
        
        sorted_freq = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:15]
        features, freqs = zip(*sorted_freq) if sorted_freq else ([], [])
        
        ax.barh(range(len(features)), freqs, alpha=0.7, color='skyblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Selection Frequency (Top 30)')
        ax.set_title('Most Frequently Selected Features')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        plt.suptitle('V7 Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / 'v7_feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def _plot_model_comparison(self, viz_dir):
        """Compare top models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract top 30 combinations
        top_30 = self.best_combinations[:30]
        
        # Panel 1: CV scores
        ax = axes[0, 0]
        ranks = [c['rank'] for c in top_30]
        cv_means = [c['cv_mean'] for c in top_30]
        cv_stds = [c['cv_std'] for c in top_30]
        
        ax.errorbar(ranks, cv_means, yerr=cv_stds, fmt='o-', alpha=0.7, capsize=3)
        ax.set_xlabel('Rank')
        ax.set_ylabel('CV Score')
        ax.set_title('Top 30 Models CV Performance')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Feature count vs rank
        ax = axes[0, 1]
        n_features = [c['n_features'] for c in top_30]
        ax.plot(ranks, n_features, 'o-', alpha=0.7, color='coral')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Count vs Rank')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Score distribution
        ax = axes[1, 0]
        ax.hist(cv_means, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(cv_means), color='red', linestyle='--', label=f'Mean: {np.mean(cv_means):.4f}')
        ax.set_xlabel('CV Score')
        ax.set_ylabel('Count')
        ax.set_title('CV Score Distribution (Top 30)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = "V7 SUMMARY\n" + "="*40 + "\n\n"
        summary += f"Total Trials: {len(self.study.trials)}\n"
        summary += f"Valid Trials: {len([t for t in self.study.trials if t.value > 0])}\n"
        summary += f"Top Combinations: {len(self.best_combinations)}\n\n"
        summary += f"BEST MODEL (Rank 1):\n"
        summary += f"  CV Score: {top_30[0]['cv_mean']:.4f} ± {top_30[0]['cv_std']:.4f}\n"
        summary += f"  Features: {top_30[0]['n_features']}\n\n"
        summary += f"TOP 30 STATISTICS:\n"
        summary += f"  Mean CV: {np.mean(cv_means):.4f}\n"
        summary += f"  Std CV: {np.std(cv_means):.4f}\n"
        summary += f"  Avg Features: {np.mean(n_features):.1f}\n"
        
        ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace')
        
        plt.suptitle('V7 Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / 'v7_model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def analyze_feature_frequency(self, top_n=30):
        """Analyze and print feature frequency in top models"""
        print("\n" + "="*80)
        print(f"FEATURE FREQUENCY ANALYSIS (Top {top_n} Models)")
        print("="*80)
        
        feature_counts = {}
        for combo in self.best_combinations[:top_n]:
            for feat in combo['features']:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Selection Frequency:")
        for feat, count in sorted_features:
            pct = count / top_n * 100
            print(f"  {feat:.<25} {count:>2}/{top_n} ({pct:>5.1f}%)")
        
        # Categorize
        print("\n" + "-"*80)
        essential = [f for f, c in sorted_features if c / top_n >= 0.8]
        important = [f for f, c in sorted_features if 0.5 <= c / top_n < 0.8]
        optional = [f for f, c in sorted_features if 0.2 <= c / top_n < 0.5]
        
        if essential:
            print(f"\nEssential (≥80%): {', '.join(essential)}")
        if important:
            print(f"Important (50-79%): {', '.join(important)}")
        if optional:
            print(f"Optional (20-49%): {', '.join(optional)}")
    
    def export_summary_report(self):
        """Export text summary of results"""
        summary_path = self.output_dir / 'results' / 'analysis_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("V7 INTELLIGENT FEATURE SELECTION - RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Metric: {self.metric}\n")
            f.write(f"Total trials: {len(self.study.trials)}\n")
            f.write(f"Valid trials: {len([t for t in self.study.trials if t.value > 0])}\n")
            f.write(f"Unique combinations: {len(self.best_combinations)}\n\n")
            
            f.write("TOP 10 FEATURE COMBINATIONS\n")
            f.write("-"*80 + "\n")
            for combo in self.best_combinations[:10]:
                f.write(f"\nRank {combo['rank']}: Trial #{combo['trial_number']}\n")
                f.write(f"  CV Score: {combo['cv_mean']:.4f} ± {combo['cv_std']:.4f}\n")
                f.write(f"  Objective: {combo['objective_value']:.4f}\n")
                f.write(f"  Features ({combo['n_features']}): {', '.join(combo['features'])}\n")
        
        print(f"\n✓ Summary report: {summary_path}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("V7 INTELLIGENT FEATURE SELECTION - MAIN SCRIPT")
    print("Optuna/TPE Sampling of 2^21 = 2,097,152 Feature Combinations")
    print("="*80)
    
    # Configuration
    dataset_path = 'ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv'
    
    # Ask user for configuration
    print("\nSelect optimization mode:")
    print("  1. Quick Test - 50 trials (~30 min)")
    print("  2. Standard - 100 trials (~1 hour) - RECOMMENDED")
    print("  3. Thorough - 200 trials (~2 hours)")
    print("  4. Exhaustive - 500 trials (~5 hours)")
    
    mode = input("\nEnter mode (1-4): ").strip()
    
    if mode == '1':
        n_trials = 50
        output_dir = 'v7_quick_test'
    elif mode == '2':
        n_trials = 100
        output_dir = 'v7_standard'
    elif mode == '3':
        n_trials = 200
        output_dir = 'v7_thorough'
    elif mode == '4':
        n_trials = 500
        output_dir = 'v7_exhaustive'
    else:
        print("Invalid mode. Using Standard (100 trials).")
        n_trials = 100
        output_dir = 'v7_standard'
    
    # Select metric
    print("\nSelect optimization metric:")
    print("  1. custom_sand (Reservoir-focused) - RECOMMENDED")
    print("  2. f1_weighted (Balanced)")
    
    metric_choice = input("Enter choice (1-2): ").strip()
    metric = 'custom_sand' if metric_choice == '1' else 'f1_weighted'
    
    # Initialize
    selector = V7OptunaFeatureSelector(
        dataset_path=dataset_path,
        metric=metric,
        output_dir=output_dir
    )
    
    # Load data
    selector.load_data()
    
    # Configure wells
    selector.configure_wells(interactive=True)
    
    # Confirm
    print("\n" + "-"*80)
    print(f"Configuration:")
    print(f"  Trials: {n_trials}")
    print(f"  Metric: {metric}")
    print(f"  Output: {output_dir}")
    confirm = input("\nProceed with optimization? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Optimization cancelled.")
        return
    
    # Run Optuna optimization
    selector.run_optimization(n_trials=n_trials, n_jobs=1)
    
    # Train top models
    print("\n" + "-"*80)
    train_confirm = input("Train top 30 models with full parameters? (y/n): ").strip().lower()
    if train_confirm == 'y':
        selector.train_top_models(top_n=30, use_gridsearch=False)
    
    # Generate predictions for test wells
    if len(selector.trained_models) > 0:
        print("\n" + "="*80)
        print("GENERATING ENSEMBLE PREDICTIONS")
        print("="*80)
        
        for test_well in selector.test_wells:
            try:
                results_df, all_probs = selector.predict_ensemble(test_well, voting_strategy='equal')
            except Exception as e:
                print(f"\n⚠ Error predicting {test_well}: {e}")
    
    # Create visualizations
    selector.create_visualizations()
    
    # Analyze and print feature frequency
    selector.analyze_feature_frequency(top_n=30)
    
    # Export summary report
    selector.export_summary_report()
    
    print("\n" + "="*80)
    print("V7 COMPLETE! Intelligent feature selection finished.")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - results/best_feature_combinations.csv - Top combinations ranked")
    print("  - results/analysis_summary.txt - Detailed text report")
    print("  - optuna_history/study.pkl - Full Optuna study")
    print("  - models/ - Trained models for top combinations")
    print("  - visualizations/ - Analysis charts")
    print("\nKey insights:")
    print(f"  - Best CV score: {selector.best_combinations[0]['cv_mean']:.4f}")
    print(f"  - Optimal features: {selector.best_combinations[0]['n_features']}")
    print(f"  - Total unique combinations found: {len(selector.best_combinations)}")


if __name__ == '__main__':
    main()

