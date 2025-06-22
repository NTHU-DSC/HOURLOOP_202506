from sklearn.compose import ColumnTransformer
from my_encoders import TargetEncoder, FrequencyEncoder
import pickle

# -----------------------------------------------------------
# 1. 產生 feature_cols：只告訴你有哪些欄位
# -----------------------------------------------------------
def build_feature_cols(
        numeric_cols,
        target_encode_cols=None,
        freq_encode_cols=None
    ):
    """回傳最終要用來建模的欄位清單（list）。"""
    feature_cols = list(numeric_cols)
    if target_encode_cols:
        feature_cols += target_encode_cols
    if freq_encode_cols:
        feature_cols += freq_encode_cols
    return feature_cols


# -----------------------------------------------------------
# 2. 產生 preprocessor：給 Pipeline 用的 ColumnTransformer
# -----------------------------------------------------------
def build_preprocessor(
        numeric_cols,
        target_encode_cols=None,
        freq_encode_cols=None
    ):
    """回傳 ColumnTransformer（直接丟進 Pipeline）。"""
    transformers = []
    if target_encode_cols:
        transformers.append(('te', TargetEncoder(), target_encode_cols))
    if freq_encode_cols:
        transformers.append(('freq', FrequencyEncoder(), freq_encode_cols))

    # numeric_cols 沒有指定 transformer ⇒ remainder='passthrough' 直接帶出
    return ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )

# Save preprocessor function to pickle file
def save_preprocessor():
    with open('my_functions.pkl', 'wb') as f:
        pickle.dump({
            'build_feature_cols': build_feature_cols,
            'build_preprocessor': build_preprocessor
        }, f)

if __name__ == '__main__':
    save_preprocessor()