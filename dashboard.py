import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, pickle

# ONNX est optionnel (seulement si le joblib contient onnx_bytes)
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

# ============================================
# 0) Unpickler tolérant (si jamais sklearn traîne)
# ============================================
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "sklearn._loss._loss" in module and name.startswith("__pyx_unpickle_"):
            def dummy(*args, **kwargs):
                raise AttributeError(f"Incompatible attribute {name}")
            return dummy
        return super().find_class(module, name)

# ============================================
# 1) Utils
# ============================================
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

# --- Prépare un vecteur numérique dans l'ordre des colonnes numériques du dataset de réf,
#     puis ajuste (pad/troncature) à n_features attendu par l'ONNX.
def vectorize_numeric(row_dict: dict, ref_df: pd.DataFrame, n_features: int) -> np.ndarray:
    # ordre déterministe: colonnes numériques du dataset d'entraînement (hors TARGET)
    num_cols = [c for c in ref_df.select_dtypes(include=[np.number]).columns if c != "TARGET"]
    vals = []
    for c in num_cols:
        v = row_dict.get(c, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=np.float32)[None, :]  # shape (1, len(num_cols))

    # ajuste dimensions
    cur = arr.shape[1]
    if cur < n_features:
        arr = np.pad(arr, ((0,0),(0, n_features-cur)), mode="constant")
    elif cur > n_features:
        arr = arr[:, :n_features]
    return arr.astype(np.float32)

# --- Pour les modèles sklearn: aligne les colonnes si possible
def df_for_sklearn(row_dict: dict, expected_features: list | None) -> pd.DataFrame:
    if expected_features:
        data = {c: row_dict.get(c, 0) for c in expected_features}
        X = pd.DataFrame([data])
    else:
        # fallback: DataFrame brut, cast numérique quand possible
        X = pd.DataFrame([row_dict])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0)

# ============================================
# 2) Chargement modèle (sklearn OU wrapper ONNX)
# ============================================
@st.cache_resource
def load_model_any(path="best_model.joblib"):
    try:
        with open(path, "rb") as f:
            obj = SafeUnpickler(f).load()
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return {"kind": "error", "err": str(e)}

    # Cas ONNX wrapper (dict avec onnx_bytes)
    if isinstance(obj, dict) and "onnx_bytes" in obj:
        if not HAS_ORT:
            st.error("Le fichier contient un modèle ONNX mais onnxruntime n'est pas installé.")
            return {"kind": "error", "err": "onnxruntime missing"}
        try:
            sess = ort.InferenceSession(obj["onnx_bytes"], providers=["CPUExecutionProvider"])
            n_in = sess.get_inputs()[0].shape[-1]
            if not isinstance(n_in, int):  # si None ou symbolique
                # on essaie de deviner avec un essai à blanc si besoin, sinon on laisse None
                n_in = int(n_in) if n_in is not None else None
            st.success("✅ Modèle ONNX chargé via wrapper joblib.")
            return {"kind": "onnx", "sess": sess, "n_features": n_in}
        except Exception as e:
            st.error(f"Erreur création session ONNX : {e}")
            return {"kind": "error", "err": str(e)}

    # Cas sklearn natif
    expected_features = getattr(obj, "feature_names_in_", None)
    n_in = getattr(obj, "n_features_in_", None)
    if expected_features is not None:
        expected_features = list(expected_features)
    st.success("✅ Modèle sklearn chargé.")
    return {"kind": "sklearn", "model": obj, "expected_features": expected_features, "n_features": n_in}

def predict_proba_generic(model_bundle: dict, row_dict: dict, ref_df: pd.DataFrame) -> float:
    kind = model_bundle.get("kind")
    if kind == "onnx":
        sess = model_bundle["sess"]
        n_in = model_bundle.get("n_features")
        if n_in is None:
            # on tente de lire dynamiquement depuis le premier input à nouveau
            try:
                n_in = sess.get_inputs()[0].shape[-1]
            except Exception:
                pass
        if not isinstance(n_in, int):
            st.error("Impossible de déterminer n_features attendu par le modèle ONNX.")
            return 0.0
        X = vectorize_numeric(row_dict, ref_df, n_in)
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: X})
        proba = out[0]
        # (N,2) → classe 1 en colonne 1 ; sinon on suppose (N,1)
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(np.ravel(proba)[0])

    elif kind == "sklearn":
        mdl = model_bundle["model"]
        X = df_for_sklearn(row_dict, model_bundle.get("expected_features"))
        # Sécurise cast numeric
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)
            return float(p[0, 1] if p.shape[1] > 1 else p[0, 0])
        elif hasattr(mdl, "decision_function"):
            s = mdl.decision_function(X)
            return float(1.0 / (1.0 + np.exp(-s[0])))
        else:
            y = mdl.predict(X)
            return float(np.ravel(y)[0])

    else:
        st.error("Modèle non disponible.")
        return 0.0

# ============================================
# 3) App
# ============================================
st.title("📊 Dashboard de Credit Scoring")

MODEL_FILE = "best_model.joblib"  # <-- on garde l'extension, comme demandé
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
HISTORY_FILE = "history.csv"

bundle = load_model_any(MODEL_FILE)

# Dataset global
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("❌ application_train.csv introuvable.")
    df_app = pd.DataFrame()

# Artefacts
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

# ============================================
# 4) UI
# ============================================
st.header("⚙️ Sélection du Mode de Test")
mode = st.radio("Choisissez :", ["Client existant", "Nouveau client"])

# ---------- Client existant ----------
if bundle.get("kind") != "error" and mode == "Client existant":
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
            prob_default = predict_proba_generic(bundle, client_dict, df_app)
            rep_str, def_str = display_probability(prob_default)
            st.markdown(rep_str, unsafe_allow_html=True)
            st.markdown(def_str, unsafe_allow_html=True)
            decision = prob_default >= 0.5
            if true_target is not None:
                st.success("✅ Bonne prédiction" if bool(true_target) == decision else "⚠️ Mauvaise prédiction")
            record = {
                "Mode": "Client existant",
                "Index": idx,
                "Vérité terrain": true_target,
                "default_probability": prob_default,
                "decision": decision
            }
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

# ---------- Nouveau client ----------
elif bundle.get("kind") != "error" and mode == "Nouveau client":
    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0)
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0)

    if st.button("⚡ Prédire nouveau client"):
        prob_default = predict_proba_generic(bundle, new_client, df_app if not df_app.empty else pd.DataFrame())
        rep_str, def_str = display_probability(prob_default)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        record = {"Mode": "Nouveau client", "Données": new_client,
                  "default_probability": prob_default, "decision": decision}
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)

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

# ---------- Historique ----------
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

# ---------- Données générales ----------
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
