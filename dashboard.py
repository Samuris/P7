import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# ------------------------------
# Fonctions utilitaires et cache
# ------------------------------
@st.cache_data
def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(model_path="best_model.joblib"):
    """Charge le modèle ML sauvegardé"""
    try:
        model = joblib.load(model_path)
        expected_features = getattr(model, "feature_names_in_", [])
        return model, list(expected_features)
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None, []

def sample_feature(df: pd.DataFrame, feature: str, sample_size: int = 10000):
    data = df[feature].dropna().replace([np.nan, np.inf, -np.inf], 0)
    if len(data) > sample_size:
        return data.sample(sample_size, random_state=42)
    return data

def append_to_csv(record: dict, filename: str):
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, mode='w', index=False)

def display_probability(prob_default):
    prob_default_pct = prob_default * 100
    prob_repayment_pct = (1 - prob_default) * 100
    if (1 - prob_default) >= 0.8:
        color_rep = "green"
    elif (1 - prob_default) >= 0.4:
        color_rep = "orange"
    else:
        color_rep = "red"
    rep_str = f"<span style='color: {color_rep}; font-size: 20px;'>Probabilité de remboursement : {prob_repayment_pct:.1f} %</span>"
    def_str = f"<span style='color: {'red' if (1 - prob_default) < 0.4 else 'black'}; font-size: 20px;'>Probabilité de défaut : {prob_default_pct:.1f} %</span>"
    return rep_str, def_str

# --- Fonctions prédiction ---
def ensure_features(df: pd.DataFrame, expected_features, reference_row=None) -> pd.DataFrame:
    if not expected_features:
        return df
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = reference_row.get(feat, 0) if reference_row else 0
    return df[[f for f in expected_features]]

def predict_proba_safely(mdl, X: pd.DataFrame):
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X)
        return float(proba[0, 1])  # proba défaut
    elif hasattr(mdl, "decision_function"):
        scores = mdl.decision_function(X)
        return float(1.0 / (1.0 + np.exp(-scores[0])))
    else:
        y = mdl.predict(X)
        return float(y[0])

# ------------------------------
# 1. Chargement
# ------------------------------

st.title("📊 Dashboard de Credit Scoring")

GRADIENT_BOOSTING_MODEL = "best_model.joblib"
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"

HISTORY_FILE = "history.csv"
NEW_CLIENTS_FILE = "new_clients.csv"

# Charger modèle
model, expected_features = load_model(GRADIENT_BOOSTING_MODEL)

# Charger dataset global
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("application_train.csv introuvable.")
    df_app = pd.DataFrame()

# Charger artefacts
fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV) if os.path.exists(FEATURE_IMPORTANCE_CSV) else pd.DataFrame()
th_df = pd.read_csv(THRESHOLDS_CSV) if os.path.exists(THRESHOLDS_CSV) else pd.DataFrame()
data_drift_available = os.path.exists(DATA_DRIFT_REPORT_HTML)

# Historique
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------
# 2. Sélection du Mode
# ------------------------------

st.header("⚙️ Sélection du Mode de Test")
mode = st.radio("Choisissez un mode :", options=["Client existant", "Nouveau client"])

# ------------------------------
# 3. Mode Client existant
# ------------------------------
if mode == "Client existant" and model is not None:
    st.subheader("🔍 Rechercher et Modifier un Client Existant")
    if not df_app.empty:
        max_index = len(df_app) - 1
        row_index = st.number_input(f"Index du client (0 à {max_index})", min_value=0, max_value=max_index, value=0)

        client_row = df_app.iloc[row_index].copy()
        true_target = client_row.get("TARGET", None)
        st.write(f"**Vérité terrain (TARGET)** : {true_target}")
        if "TARGET" in client_row:
            client_row = client_row.drop("TARGET")
        client_row = client_row.replace([np.nan, np.inf, -np.inf], 0)
        client_dict = client_row.to_dict()
        st.write("Données du client :", client_dict)

        if st.button("⚡ Prédire avec le modèle"):
            input_df = pd.DataFrame([client_dict])
            input_df = ensure_features(input_df, expected_features)
            default_prob = predict_proba_safely(model, input_df)
            rep_str, def_str = display_probability(default_prob)
            st.markdown(rep_str, unsafe_allow_html=True)
            st.markdown(def_str, unsafe_allow_html=True)
            decision = bool(default_prob >= 0.5)
            if true_target is not None:
                if bool(true_target) == decision:
                    st.success("✅ Le modèle a correctement prédit.")
                else:
                    st.warning("⚠️ Le modèle s'est trompé.")
            record = {"Mode": "Client existant", "Index": row_index,
                      "Vérité terrain": true_target,
                      "default_probability": default_prob,
                      "decision": decision}
            st.session_state["history"].append(record)
            append_to_csv(record, HISTORY_FILE)

        # Comparaison univariée
        st.subheader("📈 Comparaison univariée avec la Population")
        if st.checkbox("Afficher histogramme de comparaison"):
            columns_list = [c for c in df_app.columns if c != "TARGET"]
            selected_feature = st.selectbox("Feature à comparer", columns_list, index=0)
            fig, ax = plt.subplots()
            sns.histplot(df_app, x=selected_feature, hue="TARGET", kde=True,
                         palette={0: "blue", 1: "red"}, ax=ax)
            client_value = client_dict.get(selected_feature, 0)
            ax.axvline(client_value, color='yellow', linewidth=2, label='Client sélectionné')
            ax.legend()
            st.pyplot(fig)

        # Comparaison bivariée
        st.subheader("📊 Analyse bivariée")
        if st.checkbox("Afficher graphique bivarié"):
            cols = [c for c in df_app.columns if c != "TARGET"]
            feature_x = st.selectbox("Feature X", cols, index=0)
            feature_y = st.selectbox("Feature Y", cols, index=1)
            sample_size = 5000
            df_sample = df_app[[feature_x, feature_y, "TARGET"]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample) > sample_size:
                df_sample = df_sample.sample(sample_size, random_state=42)
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=df_sample, x=feature_x, y=feature_y, hue="TARGET",
                            palette={0: "blue", 1: "red"}, alpha=0.5, ax=ax_scatter)
            client_x = client_dict.get(feature_x, np.nan)
            client_y = client_dict.get(feature_y, np.nan)
            ax_scatter.scatter(client_x, client_y, color="yellow", s=100, label="Client")
            ax_scatter.legend()
            st.pyplot(fig_scatter)

# ------------------------------
# 4. Mode Nouveau client
# ------------------------------
elif mode == "Nouveau client" and model is not None:
    st.subheader("🆕 Créer un Nouveau Client")
    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0)
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0)

    st.write("Données saisies :", new_client)

    if st.button("⚡ Prédire (Nouveau Client)"):
        input_df = pd.DataFrame([new_client])
        input_df = ensure_features(input_df, expected_features)
        default_prob = predict_proba_safely(model, input_df)
        rep_str, def_str = display_probability(default_prob)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = bool(default_prob >= 0.5)
        record = {"Mode": "Nouveau client", "Données": new_client,
                  "default_probability": default_prob,
                  "decision": decision}
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)

# ------------------------------
# 5. Historique des Tests
# ------------------------------
st.header("🗂️ Historique des Tests (Permanent)")
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()

if st.session_state["history"]:
    session_history = pd.DataFrame(st.session_state["history"])
    combined_history = pd.concat([persistent_history, session_history], ignore_index=True)
else:
    combined_history = persistent_history

with st.expander("Afficher l'historique complet des tests", expanded=True):
    st.dataframe(combined_history, height=500)
    probs = combined_history.get("default_probability", [])
    if len(probs) > 0:
        avg_repayment = (1 - combined_history["default_probability"]).mean()
        st.metric("Performance Moyenne (Probabilité remboursement)", f"{avg_repayment*100:.1f}%")
    else:
        st.info("Pas encore de prédictions sauvegardées.")

# ------------------------------
# 6. Données Générales
# ------------------------------
with st.expander("📑 Données Générales", expanded=False):
    if not fi_df.empty:
        st.subheader("Feature Importance")
        st.dataframe(fi_df)
    if not th_df.empty:
        st.subheader("Thresholds")
        st.dataframe(th_df)
    if data_drift_available:
        st.subheader("Rapport Data Drift")
        with open(DATA_DRIFT_REPORT_HTML, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
