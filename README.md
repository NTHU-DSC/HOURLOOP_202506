# Cost Prediction System for Domestic US Shipping

## 🚚 Project Overview
This repository contains a multi-model machine learning pipeline to predict shipping cost for different shipping methods in the US domestic logistics market. The models are tailored to each `ship_method`, improving forecasting accuracy and aiding operational decisions.

## 🎯 Goal
- Predict shipping cost using tabular order features
- Support multiple shipping methods (e.g., AMAZON_FREIGHT, UPS_PARCEL...)
- Provide ranked suggestions for lowest estimated cost

---
# 📦 Hour Loop Shipping Cost Predictor v1.0

A machine learning system developed by **Hour Loop × NTHU DSC** to predict domestic U.S. shipping costs across various shipment methods using real order features.

The app supports both **single shipment input** and **batch CSV uploads**, and automatically returns predicted costs and method rankings.

---

## 🚀 Quick Start

### ▶️ Run the App

```bash
cd HOURLOOP-1
streamlit run App/main.py
````

### 🪟 Windows Users

Double-click:

```
main.bat
```

---

## 📂 Project Structure

```
HOURLOOP-1/
├── App/                     # Streamlit app and integration logic
│   ├── main.py              # Entry point
│   └── utils/               # Core logic and pre-processing
│       ├── check_data.py
│       ├── datapipeline.py
│       ├── predict.py
│       ├── fc.csv
│       ├── featuredata.json
│       └── vendors.json
│
├── models/                  # Pre-trained models (by algorithm)
│   ├── SVR/
│   ├── Bayesian/
│   ├── RandomForest/
│   ├── CatBoost/
│   └── FT-Transformer/
│
├── encoders/                # Target encoders for categorical fields
│   ├── from_state/
│   ├── to_state/
│   └── vendor_name/
│
├── data/                    # Sample input data
│   └── ESTES_test.csv
│
├── reference_code/          # Model training and tuning notebooks
│   ├── SVR_final_release.ipynb
│   └── SVR_optuna_RBF.ipynb
│
├── .gitignore
├── main.bat
└── README.md
```

---

## 🧠 Model Overview

We trained a dedicated model for each `ship_method`, using:

* Linear Regression (OLS, Ridge, Lasso)
* **Bayesian Linear Regression**
* **Support Vector Regression (SVR)**
* **Random Forest**
* **CatBoost**
* (In Progress) FT-Transformer

Each model was tuned using grid search or Optuna and validated on hold-out time ranges.

Encoded features include:

* `log_weight`, `log_volume`, `log_TVP/log_weight`, `log_distance`
* Target encoding for vendor\_name
* Frequency encoding for locations (state, city)

---

## 💻 UI Features

### 🔹 Single Shipment Query

Fill in 6 required fields:

* Weight
* Total Vendor Price
* Volume
* Vendor Name
* FC Code (destination)
* From Postal Code

→ Predict shipping cost and ranking across all methods.

### 🔸 Batch Upload

Upload a CSV file with multiple shipments:

* Auto-validates format, data types, and missing values
* Returns downloadable predictions with ranked costs

---

## 📤 Output Format

| Shipment ID | ship\_method | predicted\_cost |
| ----------- | ------------ | --------------- |
| 0001        | AMAZON\_LTL  | 12.40           |
| 0001        | UBER\_LTL    | 14.80           |
| 0001        | ESTES        | 100.20          |

All results are downloadable as `.csv`.

---

## 📚 reference\_code/

* `SVR_final_release.ipynb` – Full SVR pipeline
* `SVR_optuna_RBF.ipynb` – Optuna tuning log and search history
* Used for traceable, reproducible ML development

---

## ⚠️ Disclaimer

This project is intended for research and educational purposes.
All data used in this project is anonymized or internal to Hour Loop.

---

## 🤝 Credits

Developed by:
Dennis Pai, Howard Sun, Ellina Tsao, Hailey Chang, Jonathan Lee, Amity Wu
Hour Loop × National Tsing Hua University Data Science Club

---

## 📩 Contact

For inquiries or requests, contact:
📬 `nthu.dsc@gmail.com`
