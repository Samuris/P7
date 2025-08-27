import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

# ======================================================
# Utils
# ======================================================
@st.cache_data
def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

@st.cache_data
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

# ======================================================
# Préparation des features (encodage catégorielles)
# ======================================================
def prepare_input(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or isinstance(out[col].iloc[0], str):
            # encoder catégoriel en int
            if ref_df is not None and col in ref_df.columns:
                categories = ref_df[col].astype(str).unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(np.float32)
            else:
                out[col], _ = pd.factorize(out[col].astype(str))
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out.astype(np.float32)

# ======================================================
# ONNX Model
# ======================================================
@st.cache_resource
def load_onnx_model(path="best_model.onnx"):
    try:
        sess = ort.InferenceSession(path)
        st.success("✅ Modèle ONNX chargé avec succès.")
        return sess
    except Exception as e:
        st.error(f"Erreur chargement modèle ONNX : {e}")
        return None

def predict_proba(sess, X: pd.DataFrame):
    input_name = sess.get_inputs()[0].name
    inputs = {input_name: X.astype(np.float32).to_numpy()}
    outputs = sess.run(None, inputs)
    # Comme on a mis zipmap=False, la sortie est un tableau numpy (n, 2)
    proba = outputs[0]
    return float(proba[0, 1])  # probabilité de défaut (classe 1)




# ======================================================
# 1. Config
# ======================================================
st.title("📊 Dashboard de Credit Scoring")

MODEL_FILE = "best_model_proba.onnx"
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
HISTORY_FILE = "history.csv"

sess = load_onnx_model(MODEL_FILE)

# dataset global
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("❌ application_train.csv introuvable.")
    df_app = pd.DataFrame()

# artefacts
fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV) if os.path.exists(FEATURE_IMPORTANCE_CSV) else pd.DataFrame()
th_df = pd.read_csv(THRESHOLDS_CSV) if os.path.exists(THRESHOLDS_CSV) else pd.DataFrame()
data_drift_available = os.path.exists(DATA_DRIFT_REPORT_HTML)

# historique
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state["history"] = []

# ======================================================
# 2. Mode
# ======================================================
st.header("⚙️ Sélection du Mode de Test")
mode = st.radio("Choisissez :", ["Client existant", "Nouveau client"])

# ======================================================
# 3. Client existant
# ======================================================
if sess and mode == "Client existant":
    if not df_app.empty:
        max_index = len(df_app) - 1
        idx = st.number_input("Index du client :", min_value=0, max_value=max_index, value=0)
        client_row = df_app.iloc[idx].copy()
        true_target = client_row.get("TARGET", None)
        st.write("**Vérité terrain (TARGET)** :", true_target)
        if "TARGET" in client_row:
            client_row = client_row.drop("TARGET")
        client_row = client_row.replace([np.nan, np.inf, -np.inf], 0)
        client_dict = client_row.to_dict()

        if st.button("⚡ Prédire ce client"):
            X = pd.DataFrame([client_dict])
            prob_default = predict_proba(sess, X, df_app)
            rep_str, def_str = display_probability(prob_default)
            st.markdown(rep_str, unsafe_allow_html=True)
            st.markdown(def_str, unsafe_allow_html=True)
            decision = prob_default >= 0.5
            if true_target is not None:
                if bool(true_target) == decision:
                    st.success("✅ Bonne prédiction")
                else:
                    st.warning("⚠️ Mauvaise prédiction")
            record = {"Mode": "Client existant", "Index": idx,
                      "Vérité terrain": true_target,
                      "default_probability": prob_default,
                      "decision": decision}
            st.session_state["history"].append(record)
            append_to_csv(record, HISTORY_FILE)

        # --- Comparaison univariée
        st.subheader("📈 Comparaison univariée")
        if st.checkbox("Afficher histogramme comparaison"):
            columns_list = [c for c in df_app.columns if c != "TARGET"]
            feature = st.selectbox("Feature", columns_list, index=0)
            fig, ax = plt.subplots()
            sns.histplot(df_app, x=feature, hue="TARGET",
                         palette={0: "blue", 1: "red"}, kde=True, ax=ax)
            client_value = client_dict.get(feature, 0)
            ax.axvline(client_value, color="yellow", linewidth=2, label="Client sélectionné")
            ax.legend()
            st.pyplot(fig)

        # --- Comparaison bivariée
        st.subheader("📊 Analyse bivariée")
        if st.checkbox("Afficher scatterplot bivarié"):
            cols = [c for c in df_app.columns if c != "TARGET"]
            feature_x = st.selectbox("X", cols, index=0)
            feature_y = st.selectbox("Y", cols, index=1)
            df_sample = df_app[[feature_x, feature_y, "TARGET"]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample) > 5000:
                df_sample = df_sample.sample(5000, random_state=42)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_sample, x=feature_x, y=feature_y, hue="TARGET",
                            palette={0: "blue", 1: "red"}, alpha=0.5, ax=ax)
            ax.scatter(client_dict.get(feature_x, np.nan),
                       client_dict.get(feature_y, np.nan),
                       color="orange", s=120, label="Client sélectionné")
            ax.legend()
            st.pyplot(fig)

# ======================================================
# 4. Nouveau client
# ======================================================
elif sess and mode == "Nouveau client":
    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0)
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0)

    if st.button("⚡ Prédire nouveau client"):
        X = pd.DataFrame([new_client])
        prob_default = predict_proba(sess, X, df_app)
        rep_str, def_str = display_probability(prob_default)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        record = {"Mode": "Nouveau client", "Données": new_client,
                  "default_probability": prob_default,
                  "decision": decision}
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)

    # Comparaisons
    if not df_app.empty:
        st.subheader("📈 Comparaison univariée")
        if st.checkbox("Comparer histogramme (nouveau client)"):
            cols = [c for c in df_app.columns if c != "TARGET"]
            feature = st.selectbox("Feature", cols, index=0, key="new_univar")
            fig, ax = plt.subplots()
            sns.histplot(df_app, x=feature, hue="TARGET",
                         palette={0: "blue", 1: "red"}, kde=True, ax=ax)
            ax.axvline(new_client.get(feature, 0), color="orange", linewidth=2, label="Nouveau client")
            ax.legend()
            st.pyplot(fig)

        st.subheader("📊 Analyse bivariée")
        if st.checkbox("Comparer scatterplot (nouveau client)"):
            cols = [c for c in df_app.columns if c != "TARGET"]
            fx = st.selectbox("X", cols, index=0, key="new_x")
            fy = st.selectbox("Y", cols, index=1, key="new_y")
            df_sample = df_app[[fx, fy, "TARGET"]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample) > 5000:
                df_sample = df_sample.sample(5000, random_state=42)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_sample, x=fx, y=fy, hue="TARGET",
                            palette={0: "blue", 1: "red"}, alpha=0.5, ax=ax)
            ax.scatter(new_client.get(fx, np.nan),
                       new_client.get(fy, np.nan),
                       color="orange", s=120, label="Nouveau client")
            ax.legend()
            st.pyplot(fig)

# ======================================================
# 5. Historique
# ======================================================
st.header("🗂️ Historique des Tests")
if st.session_state["history"]:
    hist = pd.DataFrame(st.session_state["history"])
    combined = pd.concat([persistent_history, hist], ignore_index=True)
else:
    combined = persistent_history

with st.expander("Afficher l'historique complet", expanded=True):
    st.dataframe(combined, height=500)
    if "default_probability" in combined:
        avg_repayment = (1 - combined["default_probability"]).mean()
        st.metric("Performance Moyenne (Probabilité remboursement)", f"{avg_repayment*100:.1f}%")

# ======================================================
# 6. Données générales
# ======================================================
with st.expander("📑 Données Générales"):
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





