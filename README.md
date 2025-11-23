# Customer Churn Prediction

An end-to-end customer churn prediction system built on the Telco Customer Churn dataset.  
The project includes a full machine learning pipeline (data cleaning, feature engineering, model training) and a production-style Streamlit dashboard for real-time churn risk scoring.

## 🔍 Project Overview

- Predict whether a telecom customer is likely to churn (leave the service).
- Use customer demographics, contract details, billing information and service usage.
- Provide real-time churn risk with **probability scores** and **risk levels** (Low / Medium / High).
- Expose the model through a professional, dark-themed **Streamlit web UI**.

## 🧠 Model & Approach

- **Algorithm**: Support Vector Machine (SVM) with linear kernel  
- **Target**: `Churn` (Yes / No)  
- **Key steps**:
  - Drop `customerID`
  - Convert `TotalCharges` to numeric + handle missing values
  - Normalize label variants like `"No internet service"` → `"No"` and `"No phone service"` → `"No"`
  - One-hot encode relevant categorical variables
  - Ordinal encode `InternetService` and `Contract`
  - Scale numerical features using Min-Max scaling
- **Train / Test split**: 80 / 20 with stratification  
- **Metric**: Accuracy (plus churn probability from `predict_proba`)

## 🖥️ Streamlit Dashboard

The dashboard allows business users to:

- Input customer attributes (tenure, contract type, services, payment method, etc.)
- Get:
  - Churn / No-churn prediction
  - Churn probability (0–100%)
  - Risk level: **Low**, **Medium**, or **High**
- View a compact snapshot of the customer profile used for the prediction.

The UI is built with:
- Dark theme styling via custom CSS
- Metric cards for model overview
- Probability bar visualisation
- Clear separation of input and prediction sections

## 📂 Project Structure

```text
customer-churn-prediction/
├── churn_dashboard.py            # Streamlit app (UI + prediction)
├── customer_churn_model.ipynb    # Jupyter notebook (EDA + model training)
├── Telco-Customer-Churn.csv      # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
