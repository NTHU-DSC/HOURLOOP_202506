import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(self, 
                 n_numerical_features: int,
                 categorical_cardinalities: list,
                 d_token: int = 64,
                 n_blocks: int = 3,
                 attention_heads: int = 8,
                 attention_dropout: float = 0.2,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0,
                 activation: str = 'reglu',
                 prenormalization: bool = True,
                 initialization: str = 'kaiming',
                 kv_compression_ratio: Optional[float] = None,
                 kv_compression_sharing: Optional[str] = None,
                 d_out: int = 1):
        
        super().__init__()
        
        self.n_numerical_features = n_numerical_features
        self.categorical_cardinalities = categorical_cardinalities
        self.n_categorical_features = len(categorical_cardinalities)
        self.d_token = d_token
        
        # Feature tokenizers
        if n_numerical_features > 0:
            self.numerical_tokenizer = nn.Linear(n_numerical_features, d_token)
        
        if categorical_cardinalities:
            self.categorical_tokenizers = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in categorical_cardinalities
            ])
        
        # CLS token for final prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_token,
                n_heads=attention_heads,
                d_ff=d_token * 4,
                dropout=ffn_dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token // 2, d_out)
        )
        
        # Initialize weights
        self._initialize_weights(initialization)
        
    def _initialize_weights(self, initialization: str):
        """Initialize model weights."""
        if initialization == 'kaiming':
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Embedding):
                    nn.init.kaiming_normal_(module.weight)
    
    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        batch_size = numerical.size(0) if numerical is not None else categorical.size(0)
        
        tokens = []
        
        # Process numerical features
        if self.n_numerical_features > 0 and numerical is not None:
            num_tokens = self.numerical_tokenizer(numerical).unsqueeze(1)  # [batch, 1, d_token]
            tokens.append(num_tokens)
        
        # Process categorical features
        if self.categorical_cardinalities and categorical is not None:
            for i, tokenizer in enumerate(self.categorical_tokenizers):
                cat_token = tokenizer(categorical[:, i]).unsqueeze(1)  # [batch, 1, d_token]
                tokens.append(cat_token)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_token]
        tokens.insert(0, cls_tokens)
        
        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)  # [batch, n_tokens, d_token]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Use CLS token for prediction
        cls_output = x[:, 0]  # [batch, d_token]
        output = self.head(cls_output)
        
        return output.squeeze(-1)


class FTTransformerPredictor:
    """Load trained FT-Transformer model and make predictions."""
    
    def __init__(self, artifacts_dir: str):
        """
        Initialize predictor with saved artifacts.
        
        Args:
            artifacts_dir: Directory containing saved model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.metadata = None
        self.preprocessing_components = None
        
        # Load artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all saved artifacts."""
        # Find artifact files
        metadata_files = list(self.artifacts_dir.glob("*_metadata.json"))
        preprocessing_files = list(self.artifacts_dir.glob("*_preprocessing.pkl"))
        model_files = list(self.artifacts_dir.glob("*_ft_transformer.pth"))
        
        if not metadata_files or not preprocessing_files or not model_files:
            raise FileNotFoundError(f"Missing artifact files in {self.artifacts_dir}")
        
        # Load metadata
        with open(metadata_files[0], 'r') as f:
            self.metadata = json.load(f)
        
        # Load preprocessing components
        with open(preprocessing_files[0], 'rb') as f:
            self.preprocessing_components = pickle.load(f)
        
        # Initialize and load model
        n_numerical = len([f for f in self.metadata['selected_features'] 
                          if f in self.metadata['num_features']])
        
        self.model = FTTransformer(
            n_numerical_features=n_numerical,
            categorical_cardinalities=self.preprocessing_components['categorical_cardinalities'],
            d_token=self.metadata['best_params']['d_token'],
            n_blocks=self.metadata['best_params']['n_blocks'],
            attention_heads=self.metadata['best_params']['attention_heads'],
            attention_dropout=self.metadata['best_params']['attention_dropout'],
            ffn_dropout=self.metadata['best_params']['ffn_dropout']
        ).to(device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_files[0], map_location=device))
        self.model.eval()
        
        logger.info(f"Loaded FT-Transformer model for {self.metadata['ship_method']}")
        logger.info(f"Selected features: {self.metadata['selected_features']}")
    
    def preprocess_data(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data using saved preprocessing components.
        
        Args:
            X: Input DataFrame with features
            
        Returns:
            Tuple of (numerical_data, categorical_data)
        """
        # Separate numerical and categorical features
        selected_num_features = self.metadata['num_features']
        selected_cat_features = self.metadata['cat_features']
        
        # Process numerical features
        numerical_data = None
        if selected_num_features:
            X_num = X[selected_num_features].fillna(0)
            numerical_data = self.preprocessing_components['numerical_scaler'].transform(X_num)
        
        # Process categorical features
        categorical_data = None
        if selected_cat_features:
            X_cat = X[selected_cat_features].fillna('unknown')
            cat_data_list = []
            
            for col in selected_cat_features:
                encoder = self.preprocessing_components['categorical_encoders'][col]
                # Handle unseen categories
                encoded = []
                for val in X_cat[col].astype(str):
                    if val in encoder.classes_:
                        encoded.append(encoder.transform([val])[0])
                    else:
                        encoded.append(0)  # Use 0 for unknown categories
                        logger.warning(f"Unknown category '{val}' in column '{col}', using default encoding")
                
                cat_data_list.append(np.array(encoded))
            
            categorical_data = np.column_stack(cat_data_list)
        
        return numerical_data, categorical_data
    
    def predict(self, X: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input DataFrame with features
            batch_size: Batch size for prediction
            
        Returns:
            Array of predictions in original scale
        """
        logger.info(f"Making predictions on {len(X)} samples")
        
        # Validate input features
        missing_features = set(self.metadata['selected_features']) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Preprocess data
        numerical_data, categorical_data = self.preprocess_data(X)
        
        # Handle None values for model input
        if numerical_data is None:
            numerical_data = np.zeros((len(X), 0))
        if categorical_data is None:
            categorical_data = np.zeros((len(X), 0), dtype=int)
        
        # Make predictions in batches
        predictions = []
        n_samples = len(X)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            # Prepare batch
            batch_num = torch.FloatTensor(numerical_data[i:end_idx]).to(device)
            batch_cat = torch.LongTensor(categorical_data[i:end_idx]).to(device)
            
            # Predict
            with torch.no_grad():
                batch_pred = self.model(batch_num, batch_cat)
                predictions.extend(batch_pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Inverse transform to original scale
        predictions_original = self.preprocessing_components['target_scaler'].inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        logger.info("Predictions completed")
        return predictions_original
    
    def evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Adjusted R-squared
        n = len(y_true)
        p = len(self.metadata['selected_features'])
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Adjusted_R2': adj_r2,
            'n_samples': n,
            'n_features': p
        }
        
        return metrics
    
    def predict_and_evaluate(self, test_data_path: str, target_col: str = "log_cost", 
                           output_csv_path: str = "test_predictions.csv") -> Dict[str, Any]:
        """
        Complete prediction and evaluation pipeline.
        
        Args:
            test_data_path: Path to test data CSV
            target_col: Target column name
            output_csv_path: Path to save predictions CSV
            
        Returns:
            Dictionary containing predictions and evaluation metrics
        """
        logger.info(f"Loading test data from {test_data_path}")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        
        # Validate target column exists
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Extract features and target
        X_test = test_df[self.metadata['selected_features']]
        y_test = test_df[target_col]
        
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Target shape: {y_test.shape}")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = self.evaluate_predictions(y_test, y_pred)
        
        # Create results DataFrame
        results_df = test_df.copy()
        results_df['predictions'] = y_pred
        results_df['residuals'] = y_test - y_pred
        results_df['absolute_error'] = np.abs(results_df['residuals'])
        results_df['squared_error'] = results_df['residuals'] ** 2
        
        # Save predictions to CSV
        results_df.to_csv(output_csv_path, index=False)
        
        # Create summary results
        summary_results = {
            'model_info': {
                'ship_method': self.metadata['ship_method'],
                'selected_features': self.metadata['selected_features'],
                'model_params': self.metadata['best_params']
            },
            'evaluation_metrics': metrics,
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist(),
            'predictions_csv_path': output_csv_path
        }
        
        # Log results
        logger.info("="*60)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Ship Method: {self.metadata['ship_method']}")
        logger.info(f"Test Samples: {metrics['n_samples']}")
        logger.info(f"Features Used: {metrics['n_features']}")
        logger.info("-"*40)
        logger.info("METRICS:")
        for metric, value in metrics.items():
            if metric not in ['n_samples', 'n_features']:
                logger.info(f"{metric}: {value:.6f}")
        logger.info("-"*40)
        logger.info(f"Predictions saved to: {output_csv_path}")
        logger.info("="*60)
        
        return summary_results


def load_and_predict(artifacts_dir: str, test_data_path: str, 
                    target_col: str = "log_cost", 
                    output_csv_path: str = "test_predictions.csv") -> Dict[str, Any]:
    """
    Convenience function to load model and make predictions.
    
    Args:
        artifacts_dir: Directory containing saved model artifacts
        test_data_path: Path to test data CSV file
        target_col: Name of target column in test data
        output_csv_path: Path to save predictions CSV
        
    Returns:
        Dictionary containing evaluation results and metrics
    """
    try:
        # Initialize predictor
        predictor = FTTransformerPredictor(artifacts_dir)
        
        # Make predictions and evaluate
        results = predictor.predict_and_evaluate(
            test_data_path=test_data_path,
            target_col=target_col,
            output_csv_path=output_csv_path
        )
        
        # Save detailed metrics to JSON
        metrics_output_path = output_csv_path.replace('.csv', '_metrics.json')
        with open(metrics_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {metrics_output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    # Example usage - modify paths as needed
    ship_methods = ["AMAZON_LTL", "AMAZON_UPS_PARCEL",  "AMAZON_FREIGHT", "WWE_PARCEL", "WWE_LTL", "HOUR_LOOP_FEDEX_PARCEL" "UBER_LTL", "ESTES", ]
    
    for ship_method in ship_methods:
        # Path to saved model artifacts directory
        ARTIFACTS_DIR = f"Result_FT_Transformer\\{ship_method}\\model_artifacts"  # Change this to your artifacts directory
        
        # Path to test data
        TEST_DATA_PATH = f"Data\\test\\{ship_method}_test.csv"  # Change this to your test data path
        
        # Output path for predictions
        OUTPUT_CSV_PATH = "AMAZON_LTL_test_predictions.csv"
        
        try:
            # Load model and make predictions
            results = load_and_predict(
                artifacts_dir=ARTIFACTS_DIR,
                test_data_path=TEST_DATA_PATH,
                target_col="log_cost",
                output_csv_path=OUTPUT_CSV_PATH
            )
            
            print("\n" + "="*50)
            print("PREDICTION SUMMARY")
            print("="*50)
            print(f"ship_method: {ship_method}")
            print(f"MSE: {results['evaluation_metrics']['MSE']:.6f}")
            print(f"RMSE: {results['evaluation_metrics']['RMSE']:.6f}")
            print(f"MAE: {results['evaluation_metrics']['MAE']:.6f}")
            print(f"R²: {results['evaluation_metrics']['R2']:.6f}")
            print(f"Adjusted R²: {results['evaluation_metrics']['Adjusted_R2']:.6f}")
            print(f"Test Samples: {results['evaluation_metrics']['n_samples']}")
            print(f"Predictions saved to: {OUTPUT_CSV_PATH}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please check your file paths and ensure all required files exist.")