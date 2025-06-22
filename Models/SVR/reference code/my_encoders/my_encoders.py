from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pickle

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = {}
        self.defaults_ = {}
        self.columns = []

    def fit(self, X, y):
        # 計算每個類別的平均 target（log_cost）
        X = pd.DataFrame(X)
        self.columns = X.columns.tolist()
        for col in self.columns:
            df = pd.DataFrame({col: X[col], 'target': y})
            self.mapping_[col] = df.groupby(col)['target'].mean().to_dict()
            self.defaults_[col] = df['target'].mean()
        return self

    def transform(self, X):
        # 將欄位值轉換為對應的平均 target 值
        X = pd.DataFrame(X)
        return np.hstack([
            X[col].map(self.mapping_[col]).fillna(self.defaults_[col]).values.reshape(-1, 1)
            for col in self.columns
        ])
    
    def get_feature_names_out(self, input_features=None):
        return [f"{col}_target_encoded" for col in self.columns]

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    將分類特徵編碼為相對頻率 (0~1)。
    可同時處理多欄位，支援 get_feature_names_out。
    """
    def __init__(self):
        self.freq_maps_ = {}
        self.columns_ = []
        self.global_freq_ = {}
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()
        n_samples = len(X)
        for col in self.columns_:
            freq = X[col].value_counts(dropna=False) / n_samples
            self.freq_maps_[col] = freq.to_dict()
            self.global_freq_[col] = 1.0 / n_samples      # fallback ≈最小頻率
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        encoded = []
        for col in self.columns_:
            encoded_col = X[col].map(self.freq_maps_[col]).fillna(self.global_freq_[col])
            encoded.append(encoded_col.values.reshape(-1, 1))
        return np.hstack(encoded)
    
    # 讓 ColumnTransformer 能自動抓欄位名
    def get_feature_names_out(self, input_features=None):
        return [f"{col}_freq_encoded" for col in self.columns_]

# Save encoders to pickle file
def save_encoders():
    target_encoder = TargetEncoder()
    freq_encoder = FrequencyEncoder()
    
    with open('my_encoders.pkl', 'wb') as f:
        pickle.dump({
            'target_encoder': target_encoder,
            'freq_encoder': freq_encoder
        }, f)

if __name__ == '__main__':
    save_encoders()