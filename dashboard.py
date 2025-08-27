import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is None:
            expected_features = []
        else:
            expected_features = list(expected_features)

        st.success("✅ Modèle chargé avec succès")
        return model, expected_features

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, []

def load_reference_row(path="reference_row.csv"):
    try:
        row = pd.read_csv(path).iloc[0].to_dict()
        st.info("Ligne de référence chargée avec succès.")
        return row
    except Exception as e:
        st.warning(f"Impossible de charger la ligne de référence : {e}")
        return {}

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
# 2. Streamlit App
# ============================
st.title("💳 Dashboard de Credit Scoring")

MODEL_FILE = "best_model.joblib"
REFERENCE_ROW_FILE = "reference_row.csv"
APPLICATION_TRAIN_CSV = "application_train.csv"

# Charger modèle + reference row
model, expected_features = load_model_any(MODEL_FILE)
reference_row = load_reference_row(REFERENCE_ROW_FILE)

if model is None:
    st.stop()  # Stop si modèle non chargé

# ============================
# 3. Choix du mode
# ============================
mode = st.radio("Choisissez un mode :", ["Client existant", "Nouveau client"])

# ============================
# Mode : Client existant
# ============================
if mode == "Client existant":
    uploaded_file = st.file_uploader("Uploader un dataset (ex: application_train.csv)", type=["csv"])
    if uploaded_file:
        df_app = pd.read_csv(uploaded_file)
        if "TARGET" in df_app.columns:
            st.success("Dataset avec TARGET chargé ✅")
        else:
            st.warning("⚠️ TARGET manquant (pas de vérité terrain).")

        row_index = st.number_input("Index du client", min_value=0, max_value=len(df_app)-1, value=0)
        client_row = df_app.iloc[row_index].drop("TARGET", errors="ignore")
        client_dict = client_row.replace([np.nan, np.inf, -np.inf], 0).to_dict()

        st.subheader("📊 Données du client")
        st.json(client_dict)

        if st.button("⚡ Prédire ce client"):
            input_df = pd.DataFrame([client_dict])
            input_df = ensure_features(input_df, expected_features, reference_row)
            proba = predict_proba_safely(model, input_df)
            repayment = 1 - proba
            st.metric("Probabilité de remboursement", f"{repayment*100:.1f}%")
            st.metric("Probabilité de défaut", f"{proba*100:.1f}%")

        # --- Graphique : distribution univariée ---
        st.subheader("📈 Comparaison univariée avec la population")
        if st.checkbox("Afficher histogramme"):
            cols = df_app.columns.tolist()
            if "TARGET" in cols: cols.remove("TARGET")
            feature = st.selectbox("Choisir une feature", cols, index=0)
            fig, ax = plt.subplots()
            if "TARGET" in df_app.columns:
                sns.histplot(df_app, x=feature, hue="TARGET", bins=50, kde=False, ax=ax,
                             palette={0:"green",1:"red"}, alpha=0.6)
                ax.legend(title="TARGET", labels=["0 = Accepté", "1 = Refusé"])
            else:
                sns.histplot(df_app[feature].dropna(), bins=50, kde=True, ax=ax)
            ax.axvline(client_dict.get(feature,0), color="yellow", linewidth=2, label="Client")
            ax.set_title(f"Distribution de {feature}")
            ax.legend()
            st.pyplot(fig)

        # --- Graphique : scatter bivarié ---
        st.subheader("📊 Analyse bivariée")
        if st.checkbox("Afficher scatterplot"):
            cols = df_app.columns.tolist()
            if "TARGET" in cols: cols.remove("TARGET")
            f1 = st.selectbox("Feature X", cols, index=0, key="x")
            f2 = st.selectbox("Feature Y", cols, index=1, key="y")
            fig, ax = plt.subplots()
            if "TARGET" in df_app.columns:
                sns.scatterplot(data=df_app.sample(min(5000,len(df_app))), x=f1, y=f2, hue="TARGET",
                                palette={0:"green",1:"red"}, alpha=0.5, ax=ax)
            else:
                sns.scatterplot(data=df_app.sample(min(5000,len(df_app))), x=f1, y=f2, color="blue", alpha=0.5, ax=ax)
            ax.scatter(client_dict.get(f1,np.nan), client_dict.get(f2,np.nan),
                       color="yellow", s=120, label="Client")
            ax.legend()
            st.pyplot(fig)

# ============================
# Mode : Nouveau client
# ============================
elif mode == "Nouveau client":
    st.subheader("Créer un nouveau client")
    new_client = {
        "AMT_INCOME_TOTAL": st.number_input("Revenu total", value=200000.0),
        "DAYS_BIRTH": st.number_input("Âge (jours négatifs)", value=-15000.0),
        "DAYS_EMPLOYED": st.number_input("Jours d'emploi (négatifs si en cours)", value=-3000.0),
    }

    st.json(new_client)

    if st.button("⚡ Prédire ce nouveau client"):
        input_df = pd.DataFrame([new_client])
        input_df = ensure_features(input_df, expected_features, reference_row)
        proba = predict_proba_safely(model, input_df)
        repayment = 1 - proba
        st.metric("Probabilité de remboursement", f"{repayment*100:.1f}%")
        st.metric("Probabilité de défaut", f"{proba*100:.1f}%")

        # Comparaison avec population si dataset dispo
        if os.path.exists(APPLICATION_TRAIN_CSV):
            df_app = pd.read_csv(APPLICATION_TRAIN_CSV)
            st.subheader("📈 Comparaison avec population (application_train.csv)")
            feature = st.selectbox("Feature à comparer", [c for c in df_app.columns if c not in ["TARGET"]], index=0, key="newf")
            fig, ax = plt.subplots()
            if "TARGET" in df_app.columns:
                sns.histplot(df_app, x=feature, hue="TARGET", bins=50, kde=False, ax=ax,
                             palette={0:"green",1:"red"}, alpha=0.6)
                ax.legend(title="TARGET", labels=["0 = Accepté", "1 = Refusé"])
            else:
                sns.histplot(df_app[feature].dropna(), bins=50, kde=True, ax=ax)
            ax.axvline(new_client.get(feature,0), color="yellow", linewidth=2, label="Nouveau client")
            ax.legend()
            st.pyplot(fig)
