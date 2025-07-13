import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import json
from pathlib import Path
import logging
from datetime import datetime
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class TabularDataset(Dataset):
    """Custom dataset for tabular data."""
    
    def __init__(self, numerical_data: np.ndarray, categorical_data: np.ndarray, targets: np.ndarray):
        self.numerical_data = torch.FloatTensor(numerical_data)
        self.categorical_data = torch.LongTensor(categorical_data)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'numerical': self.numerical_data[idx],
            'categorical': self.categorical_data[idx],
            'target': self.targets[idx]
        }


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
                 categorical_cardinalities: List[int],
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


class FTTransformerOptimizer:
    """FT-Transformer optimizer for tabular regression."""
    
    def __init__(self, ship_method: str, top_n_feats: int = 5, data_path: Optional[str] = None):
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
        
        # Preprocessing components
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders = {}
        self.target_scaler = StandardScaler()
        
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
        
        self.num_features = [f for f in self.all_features if f not in self.cat_features]
        
        # Validation
        self._validate_data()
        
        logger.info(f"Initialized FT-Transformer optimizer for {ship_method} with {len(self.df)} samples")
    
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
    
    def select_top_features(self, method: str = 'correlation') -> List[str]:
        """Select top features using correlation analysis."""
        X = self.df[self.all_features]
        y = self.df[self.target_col]
        
        if method == 'correlation':
            return self._select_features_correlation(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _select_features_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using correlation with target."""
        # Calculate correlations for numerical features only
        num_features = [col for col in self.all_features if col not in self.cat_features]
        correlations = X[num_features].corrwith(y).abs().sort_values(ascending=False)
        
        # Add categorical features (assume they're important)
        available_cat_features = [f for f in self.cat_features if f in X.columns]
        
        # Balance numerical and categorical features
        n_cat = min(len(available_cat_features), self.top_n_feats // 2)
        n_num = min(self.top_n_feats - n_cat, len(correlations))
        
        selected_features = (list(correlations.head(n_num).index) + 
                           available_cat_features[:n_cat])
        selected_features = selected_features[:self.top_n_feats]
        
        logger.info(f"Selected features by correlation: {selected_features}")
        return selected_features
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess features and target."""
        # Separate numerical and categorical features
        selected_num_features = [f for f in self.selected_features if f in self.num_features]
        selected_cat_features = [f for f in self.selected_features if f in self.cat_features]
        
        # Process numerical features
        numerical_data = None
        if selected_num_features:
            X_num = X[selected_num_features].fillna(0)
            if fit_scalers:
                numerical_data = self.numerical_scaler.fit_transform(X_num)
            else:
                numerical_data = self.numerical_scaler.transform(X_num)
        
        # Process categorical features
        categorical_data = None
        categorical_cardinalities = []
        
        if selected_cat_features:
            X_cat = X[selected_cat_features].fillna('unknown')
            cat_data_list = []
            
            for col in selected_cat_features:
                if fit_scalers:
                    # Fit encoder
                    encoder = LabelEncoder()
                    encoded = encoder.fit_transform(X_cat[col].astype(str))
                    self.categorical_encoders[col] = encoder
                    categorical_cardinalities.append(len(encoder.classes_))
                else:
                    # Transform using existing encoder
                    encoder = self.categorical_encoders[col]
                    # Handle unseen categories
                    encoded = []
                    for val in X_cat[col].astype(str):
                        if val in encoder.classes_:
                            encoded.append(encoder.transform([val])[0])
                        else:
                            encoded.append(0)  # Use 0 for unknown categories
                    encoded = np.array(encoded)
                
                cat_data_list.append(encoded)
            
            categorical_data = np.column_stack(cat_data_list)
        
        # Process target
        y_processed = y.values.reshape(-1, 1)
        if fit_scalers:
            y_processed = self.target_scaler.fit_transform(y_processed)
        else:
            y_processed = self.target_scaler.transform(y_processed)
        y_processed = y_processed.flatten()
        
        # Store cardinalities for model initialization
        if fit_scalers:
            self.categorical_cardinalities = categorical_cardinalities
        
        return numerical_data, categorical_data, y_processed
    
    def create_data_loader(self, numerical_data: np.ndarray, categorical_data: np.ndarray, 
                          targets: np.ndarray, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        # Handle None values
        if numerical_data is None:
            numerical_data = np.zeros((len(targets), 0))
        if categorical_data is None:
            categorical_data = np.zeros((len(targets), 0), dtype=int)
        
        dataset = TabularDataset(numerical_data, categorical_data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def objective(self, trial, X_train, X_val, y_train, y_val) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Hyperparameters to optimize
        params = {
            'd_token': trial.suggest_categorical('d_token', [32, 64, 128, 192]),
            'n_blocks': trial.suggest_int('n_blocks', 2, 6),
            'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 12]),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        }
        
        try:
            # Preprocess data
            X_train_num, X_train_cat, y_train_processed = self.preprocess_data(X_train, y_train, fit_scalers=True)
            X_val_num, X_val_cat, y_val_processed = self.preprocess_data(X_val, y_val, fit_scalers=False)
            
            # Create data loaders
            train_loader = self.create_data_loader(X_train_num, X_train_cat, y_train_processed, 
                                                 params['batch_size'], shuffle=True)
            val_loader = self.create_data_loader(X_val_num, X_val_cat, y_val_processed, 
                                               params['batch_size'], shuffle=False)
            
            # Initialize model
            n_numerical = X_train_num.shape[1] if X_train_num is not None else 0
            categorical_cardinalities = getattr(self, 'categorical_cardinalities', [])
            
            model = FTTransformer(
                n_numerical_features=n_numerical,
                categorical_cardinalities=categorical_cardinalities,
                d_token=params['d_token'],
                n_blocks=params['n_blocks'],
                attention_heads=params['attention_heads'],
                attention_dropout=params['attention_dropout'],
                ffn_dropout=params['ffn_dropout']
            ).to(device)
            
            # Optimizer and loss
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=params['learning_rate'], 
                                  weight_decay=params['weight_decay'])
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            n_epochs = 50
            patience = 10
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(n_epochs):
                # Training
                train_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    outputs = model(batch['numerical'].to(device), batch['categorical'].to(device))
                    loss = criterion(outputs, batch['target'].to(device))
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(batch['numerical'].to(device), batch['categorical'].to(device))
                        loss = criterion(outputs, batch['target'].to(device))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                
                model.train()
            
            return best_val_loss
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def optimize_hyperparameters(self, X_train, X_val, y_train, y_val, 
                                n_trials: int = 50, timeout: int = 3600) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            timeout=timeout
        )
        
        self.best_params = study.best_params
        logger.info(f"Best validation loss: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame, y_val: pd.Series) -> FTTransformer:
        """Train final model with best parameters."""
        logger.info("Training final model with best parameters")
        
        # Preprocess data
        X_train_num, X_train_cat, y_train_processed = self.preprocess_data(X_train, y_train, fit_scalers=True)
        X_val_num, X_val_cat, y_val_processed = self.preprocess_data(X_val, y_val, fit_scalers=False)
        
        # Create data loaders
        train_loader = self.create_data_loader(X_train_num, X_train_cat, y_train_processed, 
                                             self.best_params['batch_size'], shuffle=True)
        val_loader = self.create_data_loader(X_val_num, X_val_cat, y_val_processed, 
                                           self.best_params['batch_size'], shuffle=False)
        
        # Initialize model
        n_numerical = X_train_num.shape[1] if X_train_num is not None else 0
        categorical_cardinalities = getattr(self, 'categorical_cardinalities', [])
        
        self.model = FTTransformer(
            n_numerical_features=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            d_token=self.best_params['d_token'],
            n_blocks=self.best_params['n_blocks'],
            attention_heads=self.best_params['attention_heads'],
            attention_dropout=self.best_params['attention_dropout'],
            ffn_dropout=self.best_params['ffn_dropout']
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), 
                              lr=self.best_params['learning_rate'], 
                              weight_decay=self.best_params['weight_decay'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        n_epochs = 200
        patience = 20
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch['numerical'].to(device), batch['categorical'].to(device))
                loss = criterion(outputs, batch['target'].to(device))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(batch['numerical'].to(device), batch['categorical'].to(device))
                    loss = criterion(outputs, batch['target'].to(device))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History - {self.ship_method}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate trained model performance."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Preprocess data
        X_num, X_cat, y_processed = self.preprocess_data(X, y, fit_scalers=False)
        
        # Create data loader
        data_loader = self.create_data_loader(X_num, X_cat, y_processed, 
                                            batch_size=512, shuffle=False)
        
        # Get predictions
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(batch['numerical'].to(device), batch['categorical'].to(device))
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform predictions and targets
        predictions_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets_original = self.target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = {
            'RMSE': mean_squared_error(targets_original, predictions_original, squared=False),
            'MAE': mean_absolute_error(targets_original, predictions_original),
            'R2': r2_score(targets_original, predictions_original),
            'adj_R2': self.adj_r2(targets_original, predictions_original, len(self.selected_features))
        }
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.Series, 
                        save_path: str = 'predictions_vs_actual.png'):
        """Plot actual vs predicted values."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get predictions
        X_num, X_cat, y_processed = self.preprocess_data(X, y, fit_scalers=False)
        data_loader = self.create_data_loader(X_num, X_cat, y_processed, 
                                            batch_size=512, shuffle=False)
        
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = self.model(batch['numerical'].to(device), batch['categorical'].to(device))
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_true = self.target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        adj_r2 = self.adj_r2(y_true, y_pred, len(self.selected_features))
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        plt.figure(figsize=(12, 10))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
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
    
    def plot_attention_weights(self, X: pd.DataFrame, sample_idx: int = 0):
        """Plot attention weights for a sample."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Preprocess single sample
        X_sample = X.iloc[sample_idx:sample_idx+1]
        X_num, X_cat, _ = self.preprocess_data(X_sample, pd.Series([0]), fit_scalers=False)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            numerical = torch.FloatTensor(X_num).to(device) if X_num is not None else None
            categorical = torch.LongTensor(X_cat).to(device) if X_cat is not None else None
            
            # Forward pass through model to get attention weights
            # This is a simplified version - you might need to modify the model to return attention weights
            logger.info("Attention visualization would require model modification to return attention weights")
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, n_permutations: int = 10) -> Dict[str, float]:
        """Calculate feature importance using permutation importance."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        logger.info("Calculating permutation feature importance...")
        
        # Get baseline performance
        baseline_metrics = self.evaluate_model(X, y)
        baseline_rmse = baseline_metrics['RMSE']
        
        feature_importance = {}
        
        for feature in self.selected_features:
            importance_scores = []
            
            for _ in range(n_permutations):
                # Create copy of data and permute feature
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                # Evaluate with permuted feature
                permuted_metrics = self.evaluate_model(X_permuted, y)
                permuted_rmse = permuted_metrics['RMSE']
                
                # Calculate importance as increase in RMSE
                importance_scores.append(permuted_rmse - baseline_rmse)
            
            feature_importance[feature] = float(np.mean(importance_scores))
        
        # Sort by importance
        self.feature_importance_scores = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        logger.info("Feature importance calculated")
        return self.feature_importance_scores
    
    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, save_path: str = 'feature_importance.png'):
        """Plot feature importance."""
        if self.feature_importance_scores is None:
            self.get_feature_importance(X, y)
        
        feat_imp_df = pd.DataFrame(list(self.feature_importance_scores.items()), 
                                  columns=['feature', 'importance'])
        feat_imp_df = feat_imp_df.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feat_imp_df)))
        bars = plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color=colors)
        
        plt.title(f'Feature Importances - {self.ship_method}', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score (RMSE Increase)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Feature importance plot saved to {save_path}")
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        if not self.best_params:
            raise ValueError("Must optimize hyperparameters first")
        
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {'rmse': [], 'mae': [], 'adj_r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            cv_model = self._train_fold_model(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
            
            # Evaluate
            fold_metrics = self._evaluate_fold_model(cv_model, X_val_cv, y_val_cv)
            
            cv_scores['rmse'].append(fold_metrics['RMSE'])
            cv_scores['mae'].append(fold_metrics['MAE'])
            cv_scores['adj_r2'].append(fold_metrics['adj_R2'])
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        logger.info("Cross-validation results:")
        for metric, value in cv_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return cv_results
    
    def _train_fold_model(self, X_train, y_train, X_val, y_val) -> FTTransformer:
        """Train model for a single CV fold."""
        # Preprocess data
        X_train_num, X_train_cat, y_train_processed = self.preprocess_data(X_train, y_train, fit_scalers=True)
        X_val_num, X_val_cat, y_val_processed = self.preprocess_data(X_val, y_val, fit_scalers=False)
        
        # Create data loaders
        train_loader = self.create_data_loader(X_train_num, X_train_cat, y_train_processed, 
                                             self.best_params['batch_size'], shuffle=True)
        val_loader = self.create_data_loader(X_val_num, X_val_cat, y_val_processed, 
                                           self.best_params['batch_size'], shuffle=False)
        
        # Initialize model
        n_numerical = X_train_num.shape[1] if X_train_num is not None else 0
        categorical_cardinalities = getattr(self, 'categorical_cardinalities', [])
        
        model = FTTransformer(
            n_numerical_features=n_numerical,
            categorical_cardinalities=categorical_cardinalities,
            d_token=self.best_params['d_token'],
            n_blocks=self.best_params['n_blocks'],
            attention_heads=self.best_params['attention_heads'],
            attention_dropout=self.best_params['attention_dropout'],
            ffn_dropout=self.best_params['ffn_dropout']
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.AdamW(model.parameters(), 
                              lr=self.best_params['learning_rate'], 
                              weight_decay=self.best_params['weight_decay'])
        criterion = nn.MSELoss()
        
        # Training loop (shorter for CV)
        n_epochs = 150
        patience = 20
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['numerical'].to(device), batch['categorical'].to(device))
                loss = criterion(outputs, batch['target'].to(device))
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['numerical'].to(device), batch['categorical'].to(device))
                    loss = criterion(outputs, batch['target'].to(device))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return model
    
    def _evaluate_fold_model(self, model: FTTransformer, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model for CV fold."""
        # Preprocess data
        X_num, X_cat, y_processed = self.preprocess_data(X, y, fit_scalers=False)
        
        # Create data loader
        data_loader = self.create_data_loader(X_num, X_cat, y_processed, 
                                            batch_size=512, shuffle=False)
        
        # Get predictions
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(batch['numerical'].to(device), batch['categorical'].to(device))
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch['target'].cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Inverse transform
        predictions_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        targets_original = self.target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(targets_original, predictions_original)),
            'MAE': mean_absolute_error(targets_original, predictions_original),
            'R2': r2_score(targets_original, predictions_original),
            'adj_R2': self.adj_r2(targets_original, predictions_original, len(self.selected_features))
        }
        
        return metrics
    
    def save_model_and_artifacts(self, output_dir: str = 'model_artifacts'):
        """Save model and related artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = output_path / f'{self.ship_method}_ft_transformer.pth'
            torch.save(self.model.state_dict(), model_path)
        
        # Save preprocessing components
        preprocessing_path = output_path / f'{self.ship_method}_preprocessing.pkl'
        preprocessing_components = {
            'numerical_scaler': self.numerical_scaler,
            'categorical_encoders': self.categorical_encoders,
            'target_scaler': self.target_scaler,
            'categorical_cardinalities': getattr(self, 'categorical_cardinalities', [])
        }
        
        with open(preprocessing_path, 'wb') as f:
            pickle.dump(preprocessing_components, f)
        
        # Save parameters and metadata
        metadata = {
            'ship_method': self.ship_method,
            'selected_features': self.selected_features,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance_scores,
            'target_col': self.target_col,
            'num_features': [f for f in self.selected_features if f in self.num_features],
            'cat_features': [f for f in self.selected_features if f in self.cat_features],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path / f'{self.ship_method}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and artifacts saved to {output_dir}")
        return output_path
    
    def load_model_and_artifacts(self, model_path: str, preprocessing_path: str, metadata_path: str):
        """Load saved model and metadata."""
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.selected_features = metadata['selected_features']
        self.best_params = metadata['best_params']
        self.feature_importance_scores = metadata.get('feature_importance')
        
        # Load preprocessing components
        with open(preprocessing_path, 'rb') as f:
            preprocessing_components = pickle.load(f)
        
        self.numerical_scaler = preprocessing_components['numerical_scaler']
        self.categorical_encoders = preprocessing_components['categorical_encoders']
        self.target_scaler = preprocessing_components['target_scaler']
        self.categorical_cardinalities = preprocessing_components['categorical_cardinalities']
        
        # Load model
        n_numerical = len([f for f in self.selected_features if f in self.num_features])
        
        self.model = FTTransformer(
            n_numerical_features=n_numerical,
            categorical_cardinalities=self.categorical_cardinalities,
            d_token=self.best_params['d_token'],
            n_blocks=self.best_params['n_blocks'],
            attention_heads=self.best_params['attention_heads'],
            attention_dropout=self.best_params['attention_dropout'],
            ffn_dropout=self.best_params['ffn_dropout']
        ).to(device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        logger.info("Model and artifacts loaded successfully")


def run_complete_pipeline(ship_method: str, top_n_feats: int = 5, 
                         n_trials: int = 20, cv_folds: int = 5,
                         data_path: Optional[str] = None) -> Tuple[FTTransformerOptimizer, Dict[str, Any]]:
    """
    Run the complete FT-Transformer modeling pipeline.
    
    Args:
        ship_method: Shipping method identifier
        top_n_feats: Number of top features to select
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        data_path: Custom path to data file
    
    Returns:
        Tuple of optimizer instance and final metrics
    """
    logger.info(f"Starting complete FT-Transformer pipeline for {ship_method}")
    
    # Initialize optimizer
    optimizer = FTTransformerOptimizer(ship_method, top_n_feats, data_path)
    
    # Feature selection
    selected_features = optimizer.select_top_features()
    optimizer.selected_features = selected_features
    
    # Prepare data
    X = optimizer.df[selected_features]
    y = optimizer.df[optimizer.target_col]
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Hyperparameter optimization
    best_params = optimizer.optimize_hyperparameters(X_train, X_val, y_train, y_val, n_trials)
    
    # Cross-validation
    cv_results = optimizer.cross_validate_model(X, y, cv_folds)
    
    # Train final model
    final_model = optimizer.train_final_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    val_metrics = optimizer.evaluate_model(X_val, y_val)
    
    # Generate visualizations
    optimizer.plot_predictions(X_val, y_val)
    optimizer.plot_feature_importance(X_val, y_val)
    
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
    
    logger.info("FT-Transformer pipeline completed successfully!")
    return optimizer, final_results


def main():
    """Main execution function with error handling."""
    try:
        optimizer, results = run_complete_pipeline(
            ship_method="AMAZON_LTL",
            top_n_feats=5,
            n_trials=150,  # Reduced for faster execution
            cv_folds=5
        )
        
        print("\n" + "="*50)
        print("FINAL FT-TRANSFORMER RESULTS SUMMARY")
        print("="*50)
        print(f"Validation RMSE: {results['validation_metrics']['RMSE']:.4f}")
        print(f"Validation Adj-R²: {results['validation_metrics']['adj_R2']:.4f}")
        print(f"CV RMSE (mean±std): {results['cv_results']['rmse_mean']:.4f}±{results['cv_results']['rmse_std']:.4f}")
        print(f"Selected Features: {results['selected_features']}")
        print(f"Output saved to: {results['output_path']}")
        print(f"Device used: {device}")
        
        return optimizer, results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    optimizer, results = main()