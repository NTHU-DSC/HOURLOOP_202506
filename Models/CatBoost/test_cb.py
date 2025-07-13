from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


for ship_method in ["WWE_PARCEL", "WWE_LTL", "UBER_LTL", "ESTES", "HOUR_LOOP_FEDEX_PARCEL", "AMAZON_UPS_PARCEL", "AMAZON_LTL", "AMAZON_FREIGHT"]:
    with open(f"Result\\{ship_method}\\model_artifacts\\{ship_method}_metadata.json", 'r') as f:
        metadata = json.load(f)
    selected_features = metadata['selected_features']
    test_df = pd.read_csv(f"Data\\test\\{ship_method}_test.csv")
    X_test = test_df[selected_features]
    y_test = test_df['log_cost']

    model = CatBoostRegressor()
    model.load_model(f"Result\\{ship_method}\\model_artifacts\\{ship_method}_catboost_model.cbm")

    y_pred = model.predict(X_test)

    def adj_r2(y_true, y_pred):
            """Calculate adjusted R-squared."""
            n = len(y_true)
            r2 = r2_score(y_true, y_pred)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 5 - 1)
            return adj_r2

    def plot_predictions(X: pd.DataFrame, y: pd.Series, save_path: str = 'predictions_vs_actual.png'):
        """Plot actual vs predicted values."""
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        adj__r2 = adj_r2(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        plt.figure(figsize=(12, 10))
        
        # Scatter plot
        plt.scatter(y, y_pred, alpha=0.6, s=30, color='steelblue', edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Statistics text box
        textstr = f'R² = {r2:.3f}\nAdj-R² = {adj__r2:.3f}\nRMSE = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.xlabel('Actual log_cost', fontsize=12)
        plt.ylabel('Predicted log_cost', fontsize=12)
        plt.title(f'Actual vs Predicted Values - {ship_method}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    print(ship_method)
    print(adj_r2(y_test, y_pred))
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    plot_predictions(X_test, y_test, save_path=f"{ship_method}_pred_vs_act.png")