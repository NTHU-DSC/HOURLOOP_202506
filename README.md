# Cost Prediction System for Domestic US Shipping

## 🚚 Project Overview
This repository contains a multi-model machine learning pipeline to predict shipping cost for different shipping methods in the US domestic logistics market. The models are tailored to each `ship_method`, improving forecasting accuracy and aiding operational decisions.

## 🎯 Goal
- Predict shipping cost using tabular order features
- Support multiple shipping methods (e.g., AMAZON_FREIGHT, UPS_PARCEL...)
- Provide ranked suggestions for lowest estimated cost

## 🧠 Models Used
- Linear Regression (OLS, Ridge, Lasso)
- Bayesian Ridge Regression
- Random Forest
- CatBoost
- Support Vector Regression (SVR)

## 📂 Project Structure

- `App/` – Streamlit app and all related code
    - `main.py` – App entry point
    - `utils/` – Feature engineering, prediction pipeline, validation
    - `encoders/` – Encoded categorical features
    - `models/` – Trained models by algorithm
- `data/` – Sample inputs or test cases
- `.gitignore`, `README.md` – Project metadata

## 🚀 Run the app

```bash
streamlit run App/main.py

## 🔁 Data Flow
1. User input/order batch uploaded
2. `DataPipeline` transforms features
3. `FinalModel` predicts cost using top-3 models per method
4. Sorted results returned for decision making

## 📦 Install & Run
```bash
git clone https://github.com/your-org/cost-prediction-project.git
cd cost-prediction-project
pip install -r requirements.txt
streamlit run app/main.py
