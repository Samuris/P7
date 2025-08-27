import streamlit as st
import pandas as pd
import numpy as np
import os

# ============================
# 1. Chargement du modèle
# ============================
def load_model_any(model_path="best_model.joblib"):
    try:
        skops_path = os.path.splitext(model_path)[0] + ".skops"
        if os.path.exists(skops_path):
            import skops.io as sio
            model = sio.load(skops_path)
        else:
            import joblib
            model = joblib.load(model_path)

        expected_features = getattr(model, "feature_names_in_", [])
        if expected_features is None:
            expected_features = []
        else:
            expected_features = list(expected_features)

        st.success("✅ Modèle chargé avec succès")
        return model, expected_features

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, []

def ensure_features(df: pd.DataFrame, expected_features, reference_row: dict) -> pd.DataFrame:
    if not expected_features:
        return df
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = reference_row.get(feat, 0)
    return df[[f for f in expected_features]]

def predict_proba_safely(mdl, X: pd.DataFrame):
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X)
        return float(proba[0, 1])
    elif hasattr(mdl, "decision_function"):
        scores = mdl.decision_function(X)
        return float(1.0 / (1.0 + np.exp(-scores[0])))
    else:
        y = mdl.predict(X)
        return float(y[0])

# ============================
# 2. App Streamlit
# ============================
st.title("💳 Dashboard de Credit Scoring")

MODEL_FILE = "best_model.joblib"
REFERENCE_ROW_FILE = "reference_row.csv"

model, expected_features = load_model_any(MODEL_FILE)

try:
    reference_row = pd.read_csv(REFERENCE_ROW_FILE).iloc[0].to_dict()
except:
    reference_row = {}

mode = st.radio("Choisissez un mode :", ["Client existant", "Nouveau client"])

# --- Client existant ---
if mode == "Client existant":
    uploaded_file = st.file_uploader("Uploader un dataset (ex: application_train.csv)", type=["csv"])
    if uploaded_file and model:
        df_app = pd.read_csv(uploaded_file)
        row_index = st.number_input("Index du client", min_value=0, max_value=len(df_app)-1, value=0)
        client_row = df_app.iloc[row_index].drop("TARGET", errors="ignore")
        client_dict = client_row.replace([np.nan, np.inf, -np.inf], 0).to_dict()

        st.write("📊 Données du client :", client_dict)

        if st.button("⚡ Prédire ce client"):
            input_df = pd.DataFrame([client_dict])
            input_df = ensure_features(input_df, expected_features, reference_row)

            proba = predict_proba_safely(model, input_df)
            repayment = 1 - proba

            st.metric("Probabilité de remboursement", f"{repayment*100:.1f}%")
            st.metric("Probabilité de défaut", f"{proba*100:.1f}%")

# --- Nouveau client ---
elif mode == "Nouveau client" and model:
    st.subheader("Créer un nouveau client")
    new_client = {
        "AMT_INCOME_TOTAL": st.number_input("Revenu total", value=200000.0),
        "DAYS_BIRTH": st.number_input("Âge (jours négatifs)", value=-15000.0),
        "DAYS_EMPLOYED": st.number_input("Jours d'emploi (négatifs si en cours)", value=-3000.0),
    }

    st.write("📊 Données saisies :", new_client)

    if st.button("⚡ Prédire ce nouveau client"):
        input_df = pd.DataFrame([new_client])
        input_df = ensure_features(input_df, expected_features, reference_row)

        proba = predict_proba_safely(model, input_df)
        repayment = 1 - proba

        st.metric("Probabilité de remboursement", f"{repayment*100:.1f}%")
        st.metric("Probabilité de défaut", f"{proba*100:.1f}%")
