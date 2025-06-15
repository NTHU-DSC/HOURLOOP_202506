import json
import pickle, cloudpickle
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")


class FinalModel:
    def __init__(self):
        self.ship_methods = ["WWE_PARCEL", "WWE_LTL", "UBER_LTL", "ESTES", "HOUR_LOOP_FEDEX_PARCEL", "AMAZON_UPS_PARCEL", "AMAZON_LTL", "AMAZON_FREIGHT"]

        with open(f"utils/featuredata.json", 'r') as f:
            self.featuredata = json.load(f)

        self.models = self._load_models()

    def _load_models(self):
        models = {ship_method: {} for ship_method in self.ship_methods}
        for ship_method in self.ship_methods:
            for model in self.featuredata[ship_method].keys():
                if model == "SVR":
                    with open(f"models/SVR/{ship_method}_model.pkl", 'rb') as f:
                        models[ship_method][model] = cloudpickle.load(f)
                
                elif model == "CatBoost":
                    cb_model = CatBoostRegressor()
                    cb_model.load_model(f"models/CatBoost/{ship_method}/{ship_method}_catboost_model.cbm")
                    models[ship_method][model] = cb_model
                
                elif model == "RandomForest":
                    with open(f"models/RandomForest/rf_{ship_method}.pkl", 'rb') as f:
                        models[ship_method][model] = pickle.load(f)
                
                elif model == "Bayesian":
                    with open(f"models/Bayesian/{ship_method}.pkl", 'rb') as f:
                        models[ship_method][model] = pickle.load(f)
        return models

    def predict(self, data: pd.DataFrame):
        results = {ship_method: [] for ship_method in self.ship_methods}

        # Predict cost
        for ship_method in self.ship_methods:
            for model in self.featuredata[ship_method].keys():
                input_data = data[self.featuredata[ship_method][model]]

                if model in ["CatBoost", "SVR", "Bayesian"]:
                    results[ship_method].append(pd.Series(self.models[ship_method][model].predict(input_data)))

                elif model == "RandomForest":
                    try:
                        input_data = input_data.rename(columns={f'vendor_name_encoded_{ship_method}': 'vendor_name_encoded'})
                    except: pass
                    results[ship_method].append(pd.Series(self.models[ship_method][model].predict(input_data)))

        # Calculate mean of outputs from each model
        for ship_method in self.ship_methods:
            results[ship_method] = pd.concat(results[ship_method], axis=1).mean(axis=1)
            results[ship_method] = np.expm1(results[ship_method])

        # Convert format and sort predicted cost value
        result_shipments = [[] for _ in range(len(data))]
        result_df_list = []
        for i in range(len(data)):
            for ship_method in self.ship_methods:
                result_shipments[i].append({
                    "Shipment ID": data["Shipment ID"].iloc[i],
                    "ship_method": ship_method,
                    "predicted_cost": results[ship_method][i]
                })
            result_shipments[i] = sorted(result_shipments[i], key=lambda x: x['predicted_cost'])
            result_df_list.extend(result_shipments[i])
        result_df = pd.DataFrame(result_df_list)

        return result_df
            

if __name__ == "__main__":
    from datapipeline import DataPipeline

    sample = pd.DataFrame({
        "Shipment ID": ["0001", "0002", "0003", "0004", "0005"],
        "vendor_name": ["Two's Company", "R&R Corp.", "Glimmer Goddess", "Pet Lou", "Glimmer Goddess"],
        "fc_code": ["ICT2", "TMB8", "TMB8", "SCK8", "TMB8"],
        "from_postal_code": ["10523", "20852", "6795", "14151-0888", "L2E 0A6"],
        "total_vendor_price": [1.2, 3.5, 4.6, 2.36, 23.5],
        "weight": [123., 456., 789., 1211., 659.],
        "volume": [1000., 2000., 3.5, 4., 502.6]
    })

    pipeline = DataPipeline()
    processed_sample = pipeline.process(sample)

    final_BIG_model = FinalModel()
    result = final_BIG_model.predict(processed_sample)
    print(result)