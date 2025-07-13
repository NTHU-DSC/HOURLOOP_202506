import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import warnings
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CatBoostModelOptimizer:
    def __init__(self, ship_method: str, top_n_feats: int = 5, data_path: Optional[str] = None):
        """
        Initialize the CatBoost optimizer.
        
        Args:
            ship_method: Shipping method identifier
            top_n_feats: Number of top features to select
            data_path: Custom path to data file (optional)
        """
        self.ship_method = ship_method
        self.top_n_feats = top_n_feats
        self.target_col = "log_cost"
        
        # Load data
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv(f'Data/train/{ship_method}_train.csv')
        
        # Model components
        self.model = None
        self.best_params = None
        self.selected_features = None
        self.feature_importance_scores = None
        
        # Feature definitions
        self.all_features = [
            "vendor_name", "fc_code", "from_state", "to_state", 
            "from_city", "to_city", "log_TVP", "log_weight", 
            "log_volume", "log_Hdis", "across_state"
        ]
        
        self.cat_features = [
            "vendor_name", "fc_code", "from_state", "to_state", 
            "from_city", "to_city", "across_state"
        ]
        
        # Validation
        self._validate_data()
        
        logger.info(f"Initialized optimizer for {ship_method} with {len(self.df)} samples")
    
    def _validate_data(self):
        """Validate input data and features."""
        missing_features = set(self.all_features) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not in data")
        
        # Check for missing values
        missing_vals = self.df[self.all_features + [self.target_col]].isnull().sum()
        if missing_vals.any():
            logger.warning(f"Missing values detected:\n{missing_vals[missing_vals > 0]}")
    
    def adj_r2(self, y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        """Calculate adjusted R-squared."""
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2
    
    def select_top_features(self, method: str = 'catboost') -> List[str]:
        """
        Select top features using various methods.
        
        Args:
            method: Feature selection method ('catboost', 'correlation', 'mutual_info')
        """
        X = self.df[self.all_features]
        y = self.df[self.target_col]
        
        if method == 'catboost':
            return self._select_features_catboost(X, y)
        elif method == 'correlation':
            return self._select_features_correlation(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _select_features_catboost(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using CatBoost feature importance."""
        cat_feature_indices = [X.columns.get_loc(col) for col in self.cat_features if col in X.columns]
        train_pool = Pool(data=X, label=y, cat_features=cat_feature_indices)
        
        # Train base model for feature selection
        base_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_state=42
        )
        base_model.fit(train_pool)
        
        # Get feature importances
        importances = base_model.get_feature_importance()
        self.feature_importance_scores = dict(zip(self.all_features, importances))
        
        # Select top features
        top_indices = np.argsort(importances)[-self.top_n_feats:]
        selected_features = [self.all_features[idx] for idx in top_indices]
        
        logger.info(f"Selected features by CatBoost importance: {selected_features}")
        return selected_features
    
    def _select_features_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using correlation with target."""
        # Calculate correlations for numerical features only
        num_features = [col for col in self.all_features if col not in self.cat_features]
        correlations = X[num_features].corrwith(y).abs().sort_values(ascending=False)
        
        # Add categorical features (assume they're important)
        top_num = min(self.top_n_feats - len(self.cat_features), len(num_features))
        selected_features = list(correlations.head(top_num).index) + self.cat_features
        selected_features = selected_features[:self.top_n_feats]
        
        logger.info(f"Selected features by correlation: {selected_features}")
        return selected_features
    
    def create_pools(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Pool, Pool]:
        """Create CatBoost Pool objects."""
        cat_feature_indices = [i for i, col in enumerate(X_train.columns) if col in self.cat_features]
        
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature_indices)
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feature_indices)
        
        return train_pool, val_pool
    
    def objective(self, trial, X_train, X_val, y_train, y_val) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 1, 4),
            'verbose': False,
            'random_state': 42,
            'eval_metric': 'RMSE',
            'od_type': 'Iter',
            'od_wait': 50
        }
        
        # Create pools
        train_pool, val_pool = self.create_pools(X_train, y_train, X_val, y_val)
        
        # Train model with early stopping
        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False
        )
        
        # Calculate validation score
        y_pred = model.predict(val_pool)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        return rmse
    
    def optimize_hyperparameters(self, X_train, X_val, y_train, y_val, 
                                n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            timeout: Maximum optimization time in seconds
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation with the best parameters."""
        if not self.best_params:
            raise ValueError("Must optimize hyperparameters first")
        
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Prepare parameters
        cv_params = self.best_params.copy()
        cv_params.update({'verbose': False, 'random_state': 42})
        
        # Create categorical feature indices
        cat_feature_indices = [i for i, col in enumerate(X.columns) if col in self.cat_features]
        
        # Cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {'rmse': [], 'mae': [], 'adj_r2': []}
        
        for train_idx, val_idx in kfold.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create pools
            train_pool = Pool(data=X_train_cv, label=y_train_cv, cat_features=cat_feature_indices)
            val_pool = Pool(data=X_val_cv, cat_features=cat_feature_indices)
            
            # Train model
            model = CatBoostRegressor(**cv_params)
            model.fit(train_pool, verbose=False)
            
            # Predict and evaluate
            y_pred = model.predict(val_pool)
            cv_scores['rmse'].append(mean_squared_error(y_val_cv, y_pred, squared=False))
            cv_scores['mae'].append(mean_absolute_error(y_val_cv, y_pred))
            cv_scores['adj_r2'].append(self.adj_r2(y_val_cv, y_pred, len(X.columns)))
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        logger.info("Cross-validation results:")
        for metric, value in cv_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return cv_results
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
        """Train final model with best parameters."""
        logger.info("Training final model with best parameters")
        
        cat_feature_indices = [i for i, col in enumerate(X.columns) if col in self.cat_features]
        train_pool = Pool(data=X, label=y, cat_features=cat_feature_indices)
        
        final_params = self.best_params.copy()
        final_params.update({
            'random_state': 42,
            'eval_metric': 'RMSE',
            'verbose': False
        })
        
        self.model = CatBoostRegressor(**final_params)
        self.model.fit(train_pool)
        
        return self.model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate trained model performance."""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        cat_feature_indices = [i for i, col in enumerate(X.columns) if col in self.cat_features]
        eval_pool = Pool(data=X, cat_features=cat_feature_indices)
        y_pred = self.model.predict(eval_pool)
        
        metrics = {
            'RMSE': mean_squared_error(y, y_pred, squared=False),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
            'adj_R2': self.adj_r2(y, y_pred, len(X.columns))
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, save_path: str = 'feature_importance.png'):
        """Plot feature importance."""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        importances = self.model.get_feature_importance()
        feature_names = self.selected_features
        
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feat_imp_df)))
        bars = plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color=colors)
        
        plt.title(f'Feature Importances - {self.ship_method}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Feature importance plot saved to {save_path}")
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.Series, 
                        save_path: str = 'predictions_vs_actual.png'):
        """Plot actual vs predicted values."""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        cat_feature_indices = [i for i, col in enumerate(X.columns) if col in self.cat_features]
        eval_pool = Pool(data=X, cat_features=cat_feature_indices)
        y_pred = self.model.predict(eval_pool)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        adj_r2 = self.adj_r2(y, y_pred, len(X.columns))
        rmse = mean_squared_error(y, y_pred, squared=False)
        
        plt.figure(figsize=(12, 10))
        
        # Scatter plot
        plt.scatter(y, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Statistics text box
        textstr = f'R² = {r2:.3f}\nAdj-R² = {adj_r2:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.xlabel(f'Actual {self.target_col}', fontsize=12)
        plt.ylabel(f'Predicted {self.target_col}', fontsize=12)
        plt.title(f'Actual vs Predicted Values - {self.ship_method}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Predictions plot saved to {save_path}")
    
    def generate_shap_analysis(self, X: pd.DataFrame, max_samples: int = 1000):
        """Generate SHAP analysis with sample limitation for performance."""
        if not self.model:
            raise ValueError("Model must be trained first")
        
        logger.info("Generating SHAP analysis...")
        
        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows for SHAP analysis")
        else:
            X_sample = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary Plot - {self.ship_method}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {self.ship_method}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("SHAP analysis completed and saved")
    
    def save_model_and_artifacts(self, output_dir: str = 'model_artifacts'):
        """Save model and related artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model
        model_path = output_path / f'{self.ship_method}_catboost_model.cbm'
        self.model.save_model(str(model_path))
        
        # Save parameters and metadata
        metadata = {
            'ship_method': self.ship_method,
            'selected_features': self.selected_features,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance_scores,
            'target_col': self.target_col,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path / f'{self.ship_method}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and artifacts saved to {output_dir}")
        return output_path
    
    def load_model_and_artifacts(self, model_path: str, metadata_path: str):
        """Load saved model and metadata."""
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.selected_features = metadata['selected_features']
        self.best_params = metadata['best_params']
        self.feature_importance_scores = metadata.get('feature_importance')
        
        logger.info("Model and artifacts loaded successfully")


def run_complete_pipeline(ship_method: str, top_n_feats: int = 5, 
                         n_trials: int = 50, cv_folds: int = 5,
                         data_path: Optional[str] = None) -> Tuple[CatBoostModelOptimizer, Dict[str, Any]]:
    """
    Run the complete modeling pipeline.
    
    Args:
        ship_method: Shipping method identifier
        top_n_feats: Number of top features to select
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        data_path: Custom path to data file
    
    Returns:
        Tuple of optimizer instance and final metrics
    """
    logger.info(f"Starting complete pipeline for {ship_method}")
    
    # Initialize optimizer
    optimizer = CatBoostModelOptimizer(ship_method, top_n_feats, data_path)
    
    # Feature selection
    selected_features = optimizer.select_top_features()
    optimizer.selected_features = selected_features
    
    # Prepare data
    X = optimizer.df[selected_features]
    y = optimizer.df[optimizer.target_col]
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Hyperparameter optimization
    best_params = optimizer.optimize_hyperparameters(X_train, X_val, y_train, y_val, n_trials)
    
    # Cross-validation
    cv_results = optimizer.cross_validate_model(X, y, cv_folds)
    
    # Train final model on full training data
    final_model = optimizer.train_final_model(X_train, y_train)
    
    # Evaluate on validation set
    val_metrics = optimizer.evaluate_model(X_val, y_val)
    
    # Generate visualizations
    optimizer.plot_feature_importance()
    optimizer.plot_predictions(X_val, y_val)
    optimizer.generate_shap_analysis(X)
    
    # Save everything
    output_path = optimizer.save_model_and_artifacts()
    
    # Combine all results
    final_results = {
        'validation_metrics': val_metrics,
        'cv_results': cv_results,
        'best_params': best_params,
        'selected_features': selected_features,
        'output_path': str(output_path)
    }
    
    logger.info("Pipeline completed successfully!")
    return optimizer, final_results


def main():
    """Main execution function with error handling."""
    try:
        optimizer, results = run_complete_pipeline(
            ship_method="AMAZON_FREIGHT",
            top_n_feats=5,
            n_trials=100,  # Increased for better optimization
            cv_folds=5
        )
        
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"Validation RMSE: {results['validation_metrics']['RMSE']:.4f}")
        print(f"Validation Adj-R²: {results['validation_metrics']['adj_R2']:.4f}")
        print(f"CV RMSE (mean±std): {results['cv_results']['rmse_mean']:.4f}±{results['cv_results']['rmse_std']:.4f}")
        print(f"Selected Features: {results['selected_features']}")
        print(f"Output saved to: {results['output_path']}")
        
        return optimizer, results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    optimizer, results = main()