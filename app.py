import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# =========================
# 1. GLOBAL PAGE CONFIG & STYLES
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark, modern UI
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #151c2c 0, #050816 45%, #02040a 100%);
            color: #f5f5f5;
        }
        h1, h2, h3, h4 {
            font-family: "Segoe UI", Roboto, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 700 !important;
        }
        .main-card {
            background: rgba(15, 23, 42, 0.95);
            border-radius: 18px;
            padding: 24px 28px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 650;
            letter-spacing: 0.03em;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 0.92rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: linear-gradient(120deg, #1d283a, #0f172a);
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #e5e7eb;
        }
        .badge span.dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.9);
        }
        .metric-card {
            background: radial-gradient(circle at top, #1e293b 0, #020617 85%);
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid rgba(148, 163, 184, 0.32);
        }
        .prob-bar {
            height: 10px;
            border-radius: 999px;
            background: #0f172a;
            overflow: hidden;
            border: 1px solid rgba(30, 64, 175, 0.7);
        }
        .prob-fill {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
        }
        .risk-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: white;
        }
        .risk-low { background: linear-gradient(135deg, #16a34a, #22c55e); }
        .risk-medium { background: linear-gradient(135deg, #eab308, #f59e0b); }
        .risk-high { background: linear-gradient(135deg, #b91c1c, #ef4444); }
        .risk-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: white;
        }
        .footer-note {
            color: #64748b;
            font-size: 0.78rem;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.5rem 1.6rem;
            border: 1px solid rgba(148, 163, 184, 0.4);
            background: linear-gradient(120deg, #2563eb, #7c3aed);
            font-weight: 600;
        }
        .stButton>button:hover {
            border-color: rgba(248, 250, 252, 0.8);
            box-shadow: 0 0 18px rgba(96, 165, 250, 0.7);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 2. TRAINING PIPELINE
# =========================

@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Drop ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # TotalCharges numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Replace "No internet/phone service"
    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_cols:
        if col in X.columns:
            X[col] = X[col].replace("No internet service", "No")

    if "MultipleLines" in X.columns:
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")

    # One-hot columns
    ohe_cols = ["gender", "PaymentMethod"] + [
        col
        for col in X.columns
        if X[col].dtype == "object" and "Yes" in X[col].unique()
    ]

    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe_array = ohe.fit_transform(X[ohe_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(ohe_cols),
        index=X.index,
    )

    # Ordinal for InternetService & Contract
    ord_internet = OrdinalEncoder(categories=[["No", "DSL", "Fiber optic"]])
    if "InternetService" in X.columns:
        X["InternetService"] = ord_internet.fit_transform(X[["InternetService"]])

    ord_contract = OrdinalEncoder(
        categories=[["Month-to-month", "One year", "Two year"]]
    )
    if "Contract" in X.columns:
        X["Contract"] = ord_contract.fit_transform(X[["Contract"]])

    # Final matrix
    X_model = pd.concat(
        [X.drop(ohe_cols, axis=1).reset_index(drop=True),
         ohe_df.reset_index(drop=True)],
        axis=1,
    )

    # Encode target
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_enc, train_size=0.8, random_state=42, stratify=y_enc
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "scaler": scaler,
        "ohe": ohe,
        "ord_internet": ord_internet,
        "ord_contract": ord_contract,
        "ohe_cols": ohe_cols,
        "label_encoder": label_encoder,
        "feature_columns": X_model.columns.tolist(),
        "accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_model.shape[1],
    }


def build_input_df(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return pd.DataFrame([data])


def preprocess_single_row(input_df, artifacts):
    ohe = artifacts["ohe"]
    ord_internet = artifacts["ord_internet"]
    ord_contract = artifacts["ord_contract"]
    ohe_cols = artifacts["ohe_cols"]
    feature_columns = artifacts["feature_columns"]
    scaler = artifacts["scaler"]

    X = input_df.copy()

    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_cols:
        if col in X.columns:
            X[col] = X[col].replace("No internet service", "No")

    if "MultipleLines" in X.columns:
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")

    ohe_array = ohe.transform(X[ohe_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(ohe_cols),
        index=X.index,
    )

    if "InternetService" in X.columns:
        X["InternetService"] = ord_internet.transform(X[["InternetService"]])

    if "Contract" in X.columns:
        X["Contract"] = ord_contract.transform(X[["Contract"]])

    X_model = pd.concat(
        [X.drop(ohe_cols, axis=1).reset_index(drop=True),
         ohe_df.reset_index(drop=True)],
        axis=1,
    )

    X_model = X_model[feature_columns]
    X_scaled = scaler.transform(X_model)
    return X_scaled


# =========================
# 3. MAIN UI
# =========================

def main():
    # HEADER
    left, right = st.columns([0.75, 0.25])
    with left:
        st.markdown(
            """
            <div class="badge">
                <span class="dot"></span>
                Telco Customer Intelligence
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h1 style="margin-top: 0.4rem;">Customer Churn Prediction</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="subtitle">Score churn risk in real time using an SVM-based model with production-style preprocessing.</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.write("")

    # TRAIN / LOAD MODEL
    with st.spinner("Loading model & preprocessing pipeline..."):
        artifacts = train_model()

    # TOP METRICS
    acc = artifacts["accuracy"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Accuracy**")
        st.markdown(f"<h2 style='margin-top: 2px;'>{acc*100:.2f}%</h2>", unsafe_allow_html=True)
        st.caption("Evaluated on held-out test set.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Training Samples**")
        st.markdown(f"<h2 style='margin-top: 2px;'>{artifacts['n_train']}</h2>", unsafe_allow_html=True)
        st.caption("Used to fit the churn model.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Model Features**")
        st.markdown(f"<h2 style='margin-top: 2px;'>{artifacts['n_features']}</h2>", unsafe_allow_html=True)
        st.caption("After all encodings & transformations.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Enter Customer Details</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle" style="margin-bottom: 1rem;">Fill in the contract, billing and service information to estimate churn probability.</p>',
        unsafe_allow_html=True,
    )

    # INPUT LAYOUT
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["No", "DSL", "Fiber optic"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

    with col3:
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    with c2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    with c3:
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=1000000.0, value=2500.0
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close main-card

    st.markdown("")
    predict_col, _ = st.columns([0.25, 0.75])
    with predict_col:
        predict_clicked = st.button("🚀 Run Churn Prediction", use_container_width=True)

    if predict_clicked:
        input_df = build_input_df(
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        )

        X_scaled = preprocess_single_row(input_df, artifacts)
        model = artifacts["model"]
        label_encoder = artifacts["label_encoder"]

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]  # [prob_no, prob_yes]
        churn_prob = float(proba[1])

        pred_label = label_encoder.inverse_transform([pred])[0]

        # Determine risk level
        if churn_prob < 0.30:
            risk_level = "Low Risk"
            risk_class = "risk-low"
            narrative = "Customer is unlikely to churn based on current profile."
        elif churn_prob < 0.60:
            risk_level = "Medium Risk"
            risk_class = "risk-medium"
            narrative = "Customer shows some churn indicators. Monitor and engage."
        else:
            risk_level = "High Risk"
            risk_class = "risk-high"
            narrative = "Customer is highly likely to churn. Immediate action recommended."

        st.markdown("---")
        res_left, res_right = st.columns([0.52, 0.48])

        with res_left:
            st.markdown(
                '<div class="main-card">',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-title">Prediction Summary</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if pred_label == "Yes":
                st.markdown(
                    "<h2 style='color:#f97373;margin-bottom:0.3rem;'>Customer is likely to CHURN</h2>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<h2 style='color:#4ade80;margin-bottom:0.3rem;'>Customer is likely to STAY</h2>",
                    unsafe_allow_html=True,
                )

            # Risk chip
            st.markdown(
                f"""
                <div class="risk-chip {risk_class}">
                    <span class="risk-dot"></span>
                    {risk_level}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br><br>", unsafe_allow_html=True)

            # Probability bar
            st.markdown("**Churn Probability**")
            st.markdown(
                f"""
                <div class="prob-bar">
                    <div class="prob-fill" style="width: {churn_prob*100:.2f}%"></div>
                </div>
                <p style="margin-top:6px;font-size:0.9rem;color:#e5e7eb;">
                    {churn_prob*100:.2f}% chance that this customer will churn.
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='subtitle' style='margin-top:4px;'>{narrative}</p>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<p class='footer-note' style='margin-top:18px;'>Probabilities are estimated using an SVM classifier with calibrated decision function. Use alongside business rules and domain judgement.</p>",
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with res_right:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-title">Customer Snapshot</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="subtitle" style="margin-bottom:0.8rem;">Key attributes used by the model for this prediction.</p>',
                unsafe_allow_html=True,
            )

            # Show a compact summary
            st.markdown(
                f"""
                - **Demographics:** {gender}, Senior Citizen: {senior_citizen}, Partner: {partner}, Dependents: {dependents}  
                - **Tenure:** `{tenure}` months  
                - **Phone Service:** {phone_service}, Multiple Lines: {multiple_lines}  
                - **Internet:** {internet_service}  
                - **Security / Backup:** OnlineSecurity: {online_security}, OnlineBackup: {online_backup}  
                - **Device / Support:** DeviceProtection: {device_protection}, TechSupport: {tech_support}  
                - **Streaming:** TV: {streaming_tv}, Movies: {streaming_movies}  
                - **Contract:** {contract}, Paperless Billing: {paperless_billing}  
                - **Payment Method:** {payment_method}  
                - **Charges:** Monthly: `{monthly_charges}`, Total: `{total_charges}`  
                """,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.header("ℹ️ About this Model")
        st.write(
            """
            This app runs a **Support Vector Machine (SVM)** model trained on the
            Telco Customer Churn dataset.  
            
            Preprocessing includes:
            - Cleaning special category values  
            - One-hot encoding service flags  
            - Ordinal encoding contract & internet type  
            - Min–Max scaling of numerical features  
            """
        )
        st.caption("You can extend this with feature importance, SHAP explanations, or your own business logic.")


if __name__ == "__main__":
    main()
