"""
Random Forest Classification - Version 7 Final: V7 Features + V6 Ensemble Method

This script takes feature combinations discovered by V7 (Optuna optimization)
and trains them using V6's full methodology:
- Load top N feature combinations from V7 results
- Train each with full GridSearchCV hyperparameter tuning
- Test with multiple optimization metrics (custom_sand, f1_weighted)
- Generate ensemble predictions with V6's proven approach
- Create comprehensive visualizations

Workflow:
1. Run v7_multi_scenario_facies_cv.py first (discovers features)
2. Run this script to train discovered features properly
3. Get best of both worlds: V7 discovery + V6 training

Author: Combines V7 and V6 approaches
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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import local config modules
import sys
sys.path.append('config')
from optimization_metrics import METRICS, get_scorer
from hyperparameter_grids import get_param_grid, count_combinations

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


class V7FinalEnsembleClassifier:
    """
    V7 Final: Load V7 feature combinations and train with V6 ensemble method
    
    Combines best of both worlds:
    - V7: Intelligent feature discovery (Optuna)
    - V6: Thorough training and ensemble predictions
    """
    
    def __init__(self, v7_results_dir, dataset_path, grid_type='standard', output_dir='v7_final_output'):
        """
        Initialize V7 Final classifier
        
        Parameters:
        -----------
        v7_results_dir : str
            Path to V7 output directory (contains best_feature_combinations.csv)
        dataset_path : str
            Path to ML-ready dataset
        grid_type : str
            Hyperparameter grid type: 'debug', 'quick', 'standard', 'full'
        output_dir : str
            Output directory for models and results
        """
        self.v7_results_dir = Path(v7_results_dir)
        self.dataset_path = dataset_path
        self.grid_type = grid_type
        self.output_dir = Path(output_dir)
        self.param_grid = get_param_grid(grid_type)
        
        # Data containers
        self.df = None
        self.train_wells = None
        self.test_wells = None
        
        # Feature combinations from V7
        self.v7_combinations = []
        
        # Results containers
        self.trained_models = {}
        self.model_rankings = []
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir,
            self.output_dir / 'models',
            self.output_dir / 'results',
            self.output_dir / 'predictions',
            self.output_dir / 'visualizations'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directories created: {self.output_dir}")
    
    def load_v7_results(self, top_n=30):
        """Load feature combinations from V7 results"""
        print("\n" + "="*80)
        print("LOADING V7 FEATURE COMBINATIONS")
        print("="*80)
        
        v7_file = self.v7_results_dir / 'results' / 'best_feature_combinations.csv'
        
        if not v7_file.exists():
            raise FileNotFoundError(
                f"V7 results not found: {v7_file}\n"
                f"Please run v7_multi_scenario_facies_cv.py first!"
            )
        
        df = pd.read_csv(v7_file)
        
        # Convert features from string to list
        if isinstance(df.iloc[0]['features'], str):
            df['features'] = df['features'].apply(eval)
        
        # Load top N combinations
        self.v7_combinations = []
        for i, row in df.head(top_n).iterrows():
            self.v7_combinations.append({
                'name': f"V7_Rank_{row['rank']:02d}",
                'features': row['features'],
                'v7_rank': row['rank'],
                'v7_cv_score': row['cv_mean'],
                'v7_n_features': row['n_features']
            })
        
        print(f"✓ Loaded {len(self.v7_combinations)} feature combinations from V7")
        print(f"\nV7 source: {self.v7_results_dir}")
        print(f"Top combination: Rank {self.v7_combinations[0]['v7_rank']}")
        print(f"  CV Score: {self.v7_combinations[0]['v7_cv_score']:.4f}")
        print(f"  Features ({self.v7_combinations[0]['v7_n_features']}): {', '.join(self.v7_combinations[0]['features'][:5])}...")
    
    def load_data(self):
        """Load and prepare dataset"""
        print("\n" + "="*80)
        print("V7 FINAL: V7 FEATURES + V6 ENSEMBLE METHOD")
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
        
        n_splits = min(4, len(self.train_wells))
        
        # Display configuration
        print("\n" + "-"*80)
        print("WELL CONFIGURATION")
        print("-"*80)
        print(f"Training wells (GroupKFold CV with k={n_splits}):")
        for well in self.train_wells:
            count = len(self.df[self.df['Well'] == well])
            print(f"  - {well} ({count} samples)")
        
        print(f"\nTest wells (blind prediction):")
        for well in self.test_wells:
            count = len(self.df[self.df['Well'] == well])
            print(f"  - {well} ({count} samples)")
        
        print(f"\nTotal training samples: {len(self.df[self.df['Well'].isin(self.train_wells)])}")
        print(f"Total test samples: {len(self.df[self.df['Well'].isin(self.test_wells)])}")
        print("-"*80)
    
    def train_all_scenarios(self, metrics=None, n_jobs=-1):
        """
        Train all V7 feature combinations with V6 methodology
        
        Parameters:
        -----------
        metrics : list or None
            List of metric names. If None, use ['custom_sand', 'f1_weighted'].
        n_jobs : int
            Number of parallel jobs for GridSearchCV (-1 = use all cores)
        """
        if metrics is None:
            metrics = ['custom_sand', 'f1_weighted']
        
        total_scenarios = len(self.v7_combinations) * len(metrics)
        
        # Prepare training data
        df_train = self.df[self.df['Well'].isin(self.train_wells)].copy()
        groups_train = df_train['Well'].values
        
        # Determine number of CV folds
        n_splits = min(4, len(self.train_wells))
        
        print("\n" + "="*80)
        print("V7 FINAL TRAINING CONFIGURATION")
        print("="*80)
        print(f"V7 feature combinations: {len(self.v7_combinations)}")
        print(f"Optimization metrics: {len(metrics)}")
        print(f"Total scenarios: {total_scenarios}")
        print(f"Hyperparameter grid: {self.grid_type}")
        print(f"  - Combinations per grid search: {count_combinations(self.param_grid)}")
        print(f"  - With {n_splits}-fold CV: {count_combinations(self.param_grid) * n_splits} fits per scenario")
        print(f"  - Total fits: {count_combinations(self.param_grid) * n_splits * total_scenarios}")
        print("="*80)
        
        print(f"\nTraining samples: {len(df_train)}")
        print(f"Training wells: {len(self.train_wells)} (for GroupKFold k={n_splits})")
        
        # Start training loop
        scenario_num = 0
        start_time = datetime.now()
        
        for combo in self.v7_combinations:
            combo_name = combo['name']
            features = combo['features']
            
            for metric_name in metrics:
                scenario_num += 1
                metric_info = METRICS[metric_name]
                scorer = metric_info['scorer']
                
                print("\n" + "="*80)
                print(f"SCENARIO {scenario_num}/{total_scenarios}: {combo_name} + {metric_name}")
                print("="*80)
                print(f"V7 Original Rank: {combo['v7_rank']}")
                print(f"V7 CV Score: {combo['v7_cv_score']:.4f}")
                print(f"Features: {len(features)}")
                print(f"Feature list: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
                print(f"Metric: {metric_info['description']}")
                
                # Check if all features available
                missing_features = [f for f in features if f not in df_train.columns]
                if missing_features:
                    print(f"⚠ Skipping - missing features: {missing_features}")
                    continue
                
                # Prepare data
                X_train = df_train[features].copy()
                y_train = df_train['Facies'].values
                
                # Create pipeline
                pipeline = Pipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=1
                    ))
                ])
                
                # GridSearchCV with GroupKFold
                gkf = GroupKFold(n_splits=n_splits)
                
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=self.param_grid,
                    scoring=scorer,
                    cv=gkf,
                    n_jobs=n_jobs,
                    verbose=1,
                    return_train_score=True
                )
                
                # Train
                print(f"\nTraining with GridSearchCV ({count_combinations(self.param_grid)} HP combinations)...")
                grid_search.fit(X_train, y_train, groups=groups_train)
                
                # Extract results
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_mean_score = grid_search.best_score_
                cv_std_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
                
                print(f"\n✓ Best CV score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
                print(f"✓ Best params: {best_params}")
                
                # Feature importance
                rf_classifier = best_model.named_steps['classifier']
                feature_importance = rf_classifier.feature_importances_
                
                # Save model and metadata
                scenario_id = f"{combo_name}_{metric_name}"
                model_dir = self.output_dir / 'models' / scenario_id
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model
                joblib.dump(best_model, model_dir / 'model.pkl', compress=3)
                
                # Save metadata
                metadata = {
                    'scenario_id': scenario_id,
                    'v7_combo_name': combo_name,
                    'v7_rank': combo['v7_rank'],
                    'v7_cv_score': combo['v7_cv_score'],
                    'metric': metric_name,
                    'n_features': len(features),
                    'features': features,
                    'cv_mean_score': float(cv_mean_score),
                    'cv_std_score': float(cv_std_score),
                    'best_params': best_params,
                    'training_wells': self.train_wells,
                    'n_cv_splits': n_splits,
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open(model_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # Save CV results
                cv_results_summary = {
                    'cv_mean_score': float(cv_mean_score),
                    'cv_std_score': float(cv_std_score),
                    'cv_scores_per_fold': [
                        float(grid_search.cv_results_[f'split{i}_test_score'][grid_search.best_index_])
                        for i in range(n_splits)
                    ]
                }
                
                with open(model_dir / 'cv_scores.json', 'w') as f:
                    json.dump(cv_results_summary, f, indent=4)
                
                # Save feature importance
                fi_df = pd.DataFrame({
                    'feature': features,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                fi_df.to_csv(model_dir / 'feature_importance.csv', index=False)
                
                # Store for ranking
                self.trained_models[scenario_id] = best_model
                self.model_rankings.append(metadata)
                
                print(f"✓ Saved to: {model_dir}")
                
                # Progress update
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                avg_time = elapsed / scenario_num
                remaining = avg_time * (total_scenarios - scenario_num)
                print(f"\nProgress: {scenario_num}/{total_scenarios} scenarios")
                print(f"Elapsed: {elapsed:.1f} min | Estimated remaining: {remaining:.1f} min")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Total scenarios trained: {len(self.trained_models)}")
        print(f"Total time: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
        
        # Create ranking table
        self._create_ranking_table()
    
    def _create_ranking_table(self):
        """Create and save model ranking table"""
        print("\nCreating model ranking table...")
        
        # Convert to DataFrame
        ranking_df = pd.DataFrame(self.model_rankings)
        
        # Sort by CV score
        ranking_df = ranking_df.sort_values('cv_mean_score', ascending=False).reset_index(drop=True)
        ranking_df['final_rank'] = range(1, len(ranking_df) + 1)
        
        # Save
        ranking_path = self.output_dir / 'results' / 'model_ranking.csv'
        ranking_df.to_csv(ranking_path, index=False)
        
        print(f"✓ Ranking saved to: {ranking_path}")
        
        # Print top 10
        print("\n" + "="*80)
        print("TOP 10 MODELS (V7 FINAL)")
        print("="*80)
        cols = ['final_rank', 'scenario_id', 'v7_rank', 'v7_cv_score', 'cv_mean_score', 'cv_std_score', 'n_features']
        print(ranking_df[cols].head(10).to_string(index=False))
        print("="*80)
        
        return ranking_df
    
    def predict_ensemble(self, target_well, voting_strategy='equal', top_n=None):
        """Generate ensemble predictions (V6 style)"""
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
        elif 'DEPTH' in df_target.columns:
            depth = df_target['DEPTH'].values
        else:
            depth = np.arange(len(df_target))
            
        facies_true = df_target['Facies'].values if 'Facies' in df_target.columns else None
        
        print(f"Samples: {len(df_target)}")
        
        # Load ranking
        ranking_df = pd.read_csv(self.output_dir / 'results' / 'model_ranking.csv')
        
        # Select models
        if top_n is not None:
            scenarios_to_use = ranking_df.head(top_n)['scenario_id'].tolist()
            print(f"Using top {top_n} models")
        else:
            scenarios_to_use = ranking_df['scenario_id'].tolist()
            print(f"Using all {len(scenarios_to_use)} models")
        
        # Collect predictions
        all_probs = []
        model_names = []
        
        for scenario_id in scenarios_to_use:
            # Load model
            model_path = self.output_dir / 'models' / scenario_id / 'model.pkl'
            
            if not model_path.exists():
                print(f"⚠ Model not found: {scenario_id}")
                continue
            
            model = joblib.load(model_path)
            
            # Load metadata to get features
            metadata_path = self.output_dir / 'models' / scenario_id / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            features = metadata['features']
            
            # Check if all features available
            missing = [f for f in features if f not in df_target.columns]
            if missing:
                print(f"⚠ Skipping {scenario_id} - missing features: {missing}")
                continue
            
            # Predict
            X_target = df_target[features]
            probs = model.predict_proba(X_target)
            
            all_probs.append(probs)
            model_names.append(scenario_id)
        
        print(f"\n✓ Collected predictions from {len(all_probs)} models")
        
        # Stack predictions
        all_probs = np.stack(all_probs, axis=0)
        
        # Ensemble prediction
        if voting_strategy == 'equal':
            avg_probs = np.mean(all_probs, axis=0)
        elif voting_strategy == 'weighted':
            weights = []
            for scenario_id in model_names:
                score = ranking_df[ranking_df['scenario_id'] == scenario_id]['cv_mean_score'].values[0]
                weights.append(score)
            weights = np.array(weights) / np.sum(weights)
            avg_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Predicted facies
        pred_facies = np.argmax(avg_probs, axis=1)
        
        # Uncertainty metrics
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=1)
        
        pred_facies_per_model = np.argmax(all_probs, axis=2)
        agreement = np.array([
            np.sum(pred_facies_per_model[:, i] == pred_facies[i]) / len(model_names)
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
        
        # Add probability columns
        n_classes = avg_probs.shape[1]
        for i in range(n_classes):
            results_df[f'Prob_Class_{i}'] = avg_probs[:, i]
        
        for i in range(n_classes, 9):
            results_df[f'Prob_Class_{i}'] = 0.0
        
        # Add uncertainty columns
        results_df['Uncertainty_Entropy'] = entropy
        results_df['Uncertainty_Agreement'] = agreement
        results_df['Uncertainty_Margin'] = prob_margin
        
        # Save predictions
        well_name_safe = target_well.replace('/', '_')
        output_dir = self.output_dir / 'predictions' / well_name_safe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)
        print(f"\n✓ Predictions saved to: {output_dir}")
        
        return results_df, all_probs
    
    def visualize_well_predictions(self, target_well, predictions_df, all_probs=None):
        """Create V6-style comprehensive visualizations"""
        well_name_safe = target_well.replace('/', '_').replace('\\', '_')
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating visualizations for {target_well}...")
        
        # 3-Panel Facies Log
        self._create_facies_log(predictions_df, target_well, well_name_safe, viz_dir)
        
        # 6-Panel Comprehensive Analysis
        self._create_comprehensive_analysis(predictions_df, target_well, well_name_safe, viz_dir)
    
    def _create_facies_log(self, df, well_name, well_name_safe, viz_dir):
        """Create 3-panel facies proportion log"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
        
        depth = df['DEPTH_MD'].values
        prob_cols = [col for col in df.columns if col.startswith('Prob_Class_')]
        n_classes = len(prob_cols)
        colors = plt.cm.tab10(np.arange(9))
        
        # Panel 1: Stacked facies proportions
        ax = axes[0]
        bottom = np.zeros(len(depth))
        
        for i in range(n_classes):
            props = df[f'Prob_Class_{i}'].values
            facies_name = FACIES_NAMES.get(i, f'F{i}')
            ax.fill_betweenx(depth, bottom, bottom + props,
                            alpha=0.7, label=facies_name, color=colors[i])
            bottom += props
        
        ax.set_xlabel('Facies Probability', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title('V7 Final Ensemble Facies Proportions')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Uncertainty profile
        ax = axes[1]
        entropy = df['Uncertainty_Entropy'].values
        agreement = df['Uncertainty_Agreement'].values
        max_entropy = np.log(9)
        entropy_norm = entropy / max_entropy
        
        ax.plot(entropy_norm, depth, 'r-', linewidth=1.5, label='Entropy (norm)')
        ax.fill_betweenx(depth, 0, entropy_norm, alpha=0.2, color='red')
        ax.plot(1 - agreement, depth, 'b-', linewidth=1.5, label='Disagreement')
        ax.fill_betweenx(depth, 0, 1 - agreement, alpha=0.2, color='blue')
        
        ax.set_xlabel('Uncertainty', fontweight='bold')
        ax.set_title('V7 Final Ensemble Uncertainty')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: True vs Predicted
        ax = axes[2]
        pred_facies = df['Facies_Predicted'].values
        
        for i in range(len(depth)-1):
            pred_color = colors[pred_facies[i]] if pred_facies[i] < 9 else 'gray'
            ax.fill_betweenx([depth[i], depth[i+1]], 0.55, 1,
                            color=pred_color, alpha=0.8)
        
        if 'Facies_True' in df.columns and not df['Facies_True'].isna().all():
            true_facies = df['Facies_True'].values
            for i in range(len(depth)-1):
                if not np.isnan(true_facies[i]):
                    true_color = colors[int(true_facies[i])]
                    ax.fill_betweenx([depth[i], depth[i+1]], 0, 0.45,
                                   color=true_color, alpha=0.8)
            
            ax.set_xticks([0.225, 0.775])
            ax.set_xticklabels(['True', 'Predicted'])
            accuracy = accuracy_score(true_facies[~np.isnan(true_facies)],
                                     pred_facies[~np.isnan(true_facies)])
            ax.text(0.5, 0.02, f'Accuracy: {accuracy:.3f}',
                   transform=ax.transAxes, ha='center', fontweight='bold')
        else:
            ax.set_xticks([0.775])
            ax.set_xticklabels(['Predicted'])
        
        ax.set_xlim([0, 1])
        ax.set_title('True vs V7 Final Ensemble')
        ax.invert_yaxis()
        
        plt.suptitle(f'V7 Final Facies Log - {well_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / f'{well_name_safe}_facies_log.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def _create_comprehensive_analysis(self, df, well_name, well_name_safe, viz_dir):
        """Create 6-panel comprehensive analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        depth = df['DEPTH_MD'].values
        pred_facies = df['Facies_Predicted'].values
        prob_cols = [col for col in df.columns if col.startswith('Prob_Class_')]
        n_classes = len(prob_cols)
        colors = plt.cm.tab10(np.arange(9))
        
        # Panel 1: Stacked proportions
        ax = axes[0, 0]
        bottom = np.zeros(len(depth))
        for i in range(n_classes):
            props = df[f'Prob_Class_{i}'].values
            ax.fill_betweenx(depth, bottom, bottom + props,
                            alpha=0.8, label=FACIES_NAMES.get(i, f'F{i}'), color=colors[i])
            bottom += props
        ax.set_xlabel('Probability')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Facies Proportions')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend(bbox_to_anchor=(1.05, 1), fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Entropy
        ax = axes[0, 1]
        entropy = df['Uncertainty_Entropy'].values
        ax.plot(entropy, depth, 'r-', linewidth=1.5)
        ax.fill_betweenx(depth, 0, entropy, alpha=0.3, color='red')
        ax.axvline(entropy.mean(), color='blue', linestyle='--', label=f'Mean: {entropy.mean():.2f}')
        ax.set_xlabel('Entropy')
        ax.set_title('Prediction Entropy')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Agreement
        ax = axes[0, 2]
        agreement = df['Uncertainty_Agreement'].values * 100
        ax.plot(agreement, depth, 'b-', linewidth=1.5)
        ax.fill_betweenx(depth, 0, agreement, alpha=0.3, color='blue')
        ax.axvline(agreement.mean(), color='red', linestyle='--',
                  label=f'Mean: {agreement.mean():.1f}%')
        ax.set_xlabel('Agreement (%)')
        ax.set_title('Model Agreement')
        ax.invert_yaxis()
        ax.set_xlim([0, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: True vs Predicted
        ax = axes[1, 0]
        for i in range(len(depth)-1):
            pred_color = colors[pred_facies[i]] if pred_facies[i] < 9 else 'gray'
            ax.fill_betweenx([depth[i], depth[i+1]], 0.55, 1, color=pred_color, alpha=0.8)
        
        if 'Facies_True' in df.columns and not df['Facies_True'].isna().all():
            true_facies = df['Facies_True'].values
            for i in range(len(depth)-1):
                if not np.isnan(true_facies[i]):
                    ax.fill_betweenx([depth[i], depth[i+1]], 0, 0.45,
                                   color=colors[int(true_facies[i])], alpha=0.8)
            ax.set_xticks([0.225, 0.775])
            ax.set_xticklabels(['True', 'Predicted'])
            accuracy = accuracy_score(true_facies[~np.isnan(true_facies)],
                                     pred_facies[~np.isnan(true_facies)])
            ax.text(0.5, 0.02, f'Accuracy: {accuracy:.3f}',
                   transform=ax.transAxes, ha='center', fontweight='bold')
        
        ax.set_xlim([0, 1])
        ax.set_title('True vs Predicted')
        ax.invert_yaxis()
        
        # Panel 5: Margin
        ax = axes[1, 1]
        margin = df['Uncertainty_Margin'].values
        ax.plot(margin, depth, 'g-', linewidth=1.5)
        ax.fill_betweenx(depth, 0, margin, alpha=0.3, color='green')
        ax.axvline(margin.mean(), color='red', linestyle='--',
                  label=f'Mean: {margin.mean():.3f}')
        ax.set_xlabel('Probability Margin')
        ax.set_title('Prediction Confidence')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary = f"V7 FINAL SUMMARY\n{'='*30}\n\n"
        summary += f"Well: {well_name}\n"
        summary += f"Samples: {len(df)}\n\n"
        
        if 'Facies_True' in df.columns and not df['Facies_True'].isna().all():
            true_facies = df['Facies_True'].values
            accuracy = accuracy_score(true_facies[~np.isnan(true_facies)],
                                     pred_facies[~np.isnan(true_facies)])
            summary += f"Accuracy: {accuracy:.4f}\n\n"
        
        summary += "UNCERTAINTY:\n"
        summary += f"  Entropy: {entropy.mean():.3f}\n"
        summary += f"  Agreement: {agreement.mean():.1f}%\n"
        summary += f"  Margin: {margin.mean():.3f}\n\n"
        
        high_conf = (agreement >= 80).sum()
        low_conf = (agreement < 60).sum()
        summary += f"CONFIDENCE:\n"
        summary += f"  High (≥80%): {high_conf/len(df)*100:.1f}%\n"
        summary += f"  Low (<60%): {low_conf/len(df)*100:.1f}%\n"
        
        ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace')
        
        plt.suptitle(f'V7 Final Comprehensive Analysis - {well_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / f'{well_name_safe}_comprehensive_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def create_summary_visualizations(self):
        """Create overall summary visualizations"""
        print("\nGenerating overall summary visualizations...")
        
        viz_dir = self.output_dir / 'visualizations'
        ranking_df = pd.read_csv(self.output_dir / 'results' / 'model_ranking.csv')
        ranking_valid = ranking_df[ranking_df['cv_mean_score'].notna()].copy()
        
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Model ranking
        ax1 = plt.subplot(2, 3, 1)
        y_pos = np.arange(min(20, len(ranking_valid)))
        ax1.barh(y_pos, ranking_valid.head(20)['cv_mean_score'],
                xerr=ranking_valid.head(20)['cv_std_score'], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ranking_valid.head(20)['scenario_id'], fontsize=8)
        ax1.set_xlabel('CV Mean Score')
        ax1.set_title('Model Ranking (Top 20)')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
        
        # Panel 2: V7 rank vs Final performance
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(ranking_valid['v7_rank'], ranking_valid['cv_mean_score'],
                   alpha=0.6, s=80, c=ranking_valid['final_rank'], cmap='viridis')
        ax2.set_xlabel('V7 Original Rank')
        ax2.set_ylabel('V7 Final CV Score')
        ax2.set_title('V7 Discovery vs Final Training')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Features vs Performance
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(ranking_valid['n_features'], ranking_valid['cv_mean_score'],
                   alpha=0.6, s=80)
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('CV Score')
        ax3.set_title('Features vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Metric comparison
        ax4 = plt.subplot(2, 3, 4)
        metrics = ranking_valid['metric'].unique()
        for metric in metrics:
            metric_scores = ranking_valid[ranking_valid['metric'] == metric]['cv_mean_score']
            ax4.hist(metric_scores, alpha=0.6, label=metric, bins=10)
        ax4.set_xlabel('CV Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Score Distribution by Metric')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: CV std distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(ranking_valid['cv_std_score'], bins=15, color='coral', alpha=0.7)
        ax5.axvline(ranking_valid['cv_std_score'].mean(), color='red', linestyle='--',
                   label=f"Mean: {ranking_valid['cv_std_score'].mean():.3f}")
        ax5.set_xlabel('CV Std Score')
        ax5.set_ylabel('Count')
        ax5.set_title('Model Stability Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary = "V7 FINAL SUMMARY\n" + "="*35 + "\n\n"
        summary += f"V7 Combinations Used: {len(self.v7_combinations)}\n"
        summary += f"Total Models: {len(ranking_df)}\n"
        summary += f"Valid Scores: {len(ranking_valid)}\n\n"
        summary += "TOP 3 MODELS:\n"
        for idx, row in ranking_valid.head(3).iterrows():
            summary += f"  {row['final_rank']}. {row['scenario_id']}\n"
            summary += f"     {row['cv_mean_score']:.4f} ± {row['cv_std_score']:.4f}\n"
            summary += f"     (V7 Rank: {row['v7_rank']})\n"
        summary += f"\nBest Score: {ranking_valid['cv_mean_score'].max():.4f}\n"
        summary += f"Mean Score: {ranking_valid['cv_mean_score'].mean():.4f}\n"
        
        ax6.text(0.1, 0.95, summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', family='monospace')
        
        plt.suptitle('V7 Final Multi-Scenario Summary (V7 Discovery + V6 Training)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = viz_dir / 'v7_final_summary_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("V7 FINAL - V7 DISCOVERY + V6 ENSEMBLE METHOD")
    print("="*80)
    
    # Configuration
    dataset_path = 'ML_ready_F14_PHIF_SW_predicted_with_confidence_fin.csv'
    
    # Ask for V7 results directory
    print("\nEnter V7 results directory:")
    print("  (e.g., v7_output, v7_standard, v7_exhaustive)")
    v7_dir = input("V7 directory: ").strip()
    
    if not v7_dir:
        v7_dir = 'v7_output'
        print(f"Using default: {v7_dir}")
    
    # Check if directory exists
    if not Path(v7_dir).exists():
        print(f"\n❌ Error: Directory not found: {v7_dir}")
        print("Please run v7_multi_scenario_facies_cv.py first!")
        return
    
    # Ask for top N combinations
    print("\nHow many V7 combinations to train?")
    print("  Recommended: 20-30 (balance between diversity and time)")
    top_n_input = input("Number of combinations (default=20): ").strip()
    top_n = int(top_n_input) if top_n_input.isdigit() else 20
    
    # Grid type
    print("\nSelect hyperparameter grid:")
    print("  1. Quick - 4 combinations (~5 min per scenario)")
    print("  2. Standard - 72 combinations (~15 min per scenario) - RECOMMENDED")
    print("  3. Full - 324 combinations (~30 min per scenario)")
    grid_choice = input("Enter choice (1-3, default=2): ").strip()
    
    grid_type = {
        '1': 'quick',
        '2': 'standard',
        '3': 'full'
    }.get(grid_choice, 'standard')
    
    output_dir = 'v7_final_output'
    
    # Initialize classifier
    classifier = V7FinalEnsembleClassifier(
        v7_results_dir=v7_dir,
        dataset_path=dataset_path,
        grid_type=grid_type,
        output_dir=output_dir
    )
    
    # Load V7 results
    classifier.load_v7_results(top_n=top_n)
    
    # Load data
    classifier.load_data()
    
    # Configure wells
    classifier.configure_wells(interactive=True)
    
    # Confirm
    print("\n" + "-"*80)
    total_scenarios = top_n * 2  # 2 metrics
    print(f"Configuration:")
    print(f"  V7 combinations: {top_n}")
    print(f"  Metrics: 2 (custom_sand, f1_weighted)")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Grid type: {grid_type}")
    print(f"  Output: {output_dir}")
    
    confirm = input("\nProceed with training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Train all scenarios
    classifier.train_all_scenarios(
        metrics=['custom_sand', 'f1_weighted'],
        n_jobs=-1
    )
    
    # Generate ensemble predictions for test wells
    print("\n" + "="*80)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("="*80)
    
    for test_well in classifier.test_wells:
        try:
            results_df, all_probs = classifier.predict_ensemble(test_well, voting_strategy='equal')
            
            if results_df is not None:
                classifier.visualize_well_predictions(test_well, results_df, all_probs)
                
        except Exception as e:
            print(f"\n⚠ Error predicting {test_well}: {e}")
    
    # Generate summary visualizations
    print("\n" + "="*80)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*80)
    
    try:
        classifier.create_summary_visualizations()
    except Exception as e:
        print(f"\n⚠ Error generating summary: {e}")
    
    print("\n" + "="*80)
    print("V7 FINAL COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - results/model_ranking.csv - All models ranked by final performance")
    print("  - predictions/{well}/ - Ensemble predictions per well")
    print("  - visualizations/ - All charts and plots")
    print("  - models/ - All trained models")
    print("\nKey visualizations:")
    print("  - *_facies_log.png - 3-panel facies proportion logs")
    print("  - *_comprehensive_analysis.png - 6-panel detailed analysis")
    print("  - v7_final_summary_analysis.png - Overall model comparison")


if __name__ == '__main__':
    main()

