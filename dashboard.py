# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, pickle

# onnxruntime est optionnel (utilisé si best_model.joblib contient des onnx_bytes)
try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False

# ==============================
# Unpickler tolérant (sklearn)
# ==============================
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "sklearn._loss._loss" in module and name.startswith("__pyx_unpickle_"):
            def dummy(*args, **kwargs):
                raise AttributeError(f"Incompatible attribute {name}")
            return dummy
        return super().find_class(module, name)

# ==============================
# Utils généraux
# ==============================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def append_to_csv(record: dict, filename: str):
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, mode='w', index=False)

def display_probability(prob_default: float):
    prob_default_pct = prob_default * 100.0
    prob_repayment_pct = (1.0 - prob_default) * 100.0
    if (1 - prob_default) >= 0.8:
        color_rep = "green"
    elif (1 - prob_default) >= 0.4:
        color_rep = "orange"
    else:
        color_rep = "red"
    rep_str = f"<span style='color: {color_rep}; font-size: 20px;'>Probabilité de remboursement : {prob_repayment_pct:.1f} %</span>"
    def_str = f"<span style='color: {'red' if (1 - prob_default) < 0.4 else 'black'}; font-size: 20px;'>Probabilité de défaut : {prob_default_pct:.1f} %</span>"
    return rep_str, def_str

@st.cache_data
def sample_feature(df: pd.DataFrame, feature: str, sample_size: int = 10000):
    data = df[feature].dropna().replace([np.nan, np.inf, -np.inf], 0)
    if len(data) > sample_size:
        return data.sample(sample_size, random_state=42)
    return data

# ==============================
# Chargement reference_row
# ==============================
def load_reference_row(path="reference_row.csv"):
    if os.path.exists(path):
        try:
            ref_df = pd.read_csv(path, nrows=1)
            # garde l'ordre exact des colonnes du CSV
            ref_row = ref_df.iloc[0].to_dict()
            feature_order = [c for c in ref_df.columns if c != "TARGET"]
            st.success("Ligne de référence chargée.")
            return ref_row, feature_order
        except Exception as e:
            st.warning(f"Impossible de lire {path} : {e}")
    return {}, []

# ==============================
# Chargement du modèle (sklearn ou ONNX wrapper)
# ==============================
@st.cache_resource
def load_model_any(path="best_model.joblib"):
    try:
        with open(path, "rb") as f:
            obj = SafeUnpickler(f).load()
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return {"kind": "error", "err": str(e)}

    # Wrapper ONNX (dict avec onnx_bytes)
    if isinstance(obj, dict) and "onnx_bytes" in obj:
        if not HAS_ORT:
            st.error("Le fichier contient un modèle ONNX mais onnxruntime n'est pas installé.")
            return {"kind": "error", "err": "onnxruntime_missing"}
        try:
            sess = ort.InferenceSession(obj["onnx_bytes"], providers=["CPUExecutionProvider"])
            # essaie de récupérer la dimension attendue
            shape = sess.get_inputs()[0].shape
            n_in = shape[-1] if isinstance(shape, (list, tuple)) else None
            if not isinstance(n_in, int):
                n_in = None
            st.success("✅ Modèle ONNX chargé (via best_model.joblib).")
            # On accepte un éventuel 'feature_order' dans le dict wrapper si tu l'avais mis
            feat_order = obj.get("feature_order", None)
            if feat_order is not None and isinstance(feat_order, (list, tuple)):
                feat_order = list(feat_order)
            else:
                feat_order = None
            return {"kind": "onnx", "sess": sess, "n_features": n_in, "feature_order": feat_order}
        except Exception as e:
            st.error(f"Erreur ONNXRuntime : {e}")
            return {"kind": "error", "err": str(e)}

    # sklearn natif
    expected_features = getattr(obj, "feature_names_in_", None)
    if expected_features is not None:
        expected_features = list(expected_features)
    n_in = getattr(obj, "n_features_in_", None)
    st.success("✅ Modèle sklearn chargé.")
    return {"kind": "sklearn", "model": obj, "expected_features": expected_features, "n_features": n_in}

# ==============================
# Préparation des features
# ==============================
def ensure_features_sklearn(row: dict, expected_features: list | None, reference_row: dict | None):
    """
    Reproduit la logique Flask:
    - si expected_features connu: on crée un DF avec ces colonnes
    - on complète les manquantes via reference_row (sinon 0)
    - on cast en numérique quand possible (sinon on laisse tel quel si le pipeline gère)
    """
    if expected_features:
        data = {}
        for feat in expected_features:
            if row.get(feat) is not None:
                data[feat] = row.get(feat)
            elif reference_row and (feat in reference_row):
                data[feat] = reference_row[feat]
            else:
                data[feat] = 0
        X = pd.DataFrame([data])
    else:
        X = pd.DataFrame([row])
        # complète avec reference_row pour les colonnes connues
        if reference_row:
            for k, v in reference_row.items():
                if k not in X.columns:
                    X[k] = v

    # cast numérique si possible
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            pass
    return X.fillna(0)

def vectorize_for_onnx(row: dict, feature_order: list, reference_row: dict, n_features: int):
    """
    Construit un vecteur float32 (1, n_features) dans l'ordre des colonnes de référence.
    - Si une feature manque dans row, prend la valeur de reference_row, sinon 0.
    - Cast float (les categoriels deviennent 0 si non numériques).
    - Pad/tronque pour matcher n_features du modèle ONNX.
    """
    vals = []
    for feat in feature_order:
        v = row.get(feat, reference_row.get(feat, 0))
        try:
            v = float(v)
        except Exception:
            v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=np.float32)[None, :]  # (1, len(feature_order))

    if isinstance(n_features, int):
        cur = arr.shape[1]
        if cur < n_features:
            arr = np.pad(arr, ((0, 0), (0, n_features - cur)), mode="constant")
        elif cur > n_features:
            arr = arr[:, :n_features]
    return arr.astype(np.float32)

# ==============================
# Prédiction (sklearn / onnx)
# ==============================
def predict_proba_generic(bundle: dict, row_dict: dict, ref_df: pd.DataFrame, reference_row: dict, ref_feature_order: list):
    kind = bundle.get("kind")

    if kind == "sklearn":
        mdl = bundle["model"]
        expected = bundle.get("expected_features")
        X = ensure_features_sklearn(row_dict, expected, reference_row)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)
            return float(p[0, 1] if p.shape[1] > 1 else p[0, 0])
        elif hasattr(mdl, "decision_function"):
            s = mdl.decision_function(X)
            return float(1.0 / (1.0 + np.exp(-s[0])))
        else:
            y = mdl.predict(X)
            return float(np.ravel(y)[0])

    if kind == "onnx":
        if not HAS_ORT:
            st.error("onnxruntime non disponible.")
            return 0.0
        sess = bundle["sess"]
        n_in = bundle.get("n_features")
        # ordre priorité: ordre fourni dans le wrapper → reference_row.csv → colonnes numériques du df_app
        feature_order = bundle.get("feature_order") or ref_feature_order
        if not feature_order:
            # fallback: colonnes numériques du dataset global, hors TARGET
            feature_order = [c for c in ref_df.select_dtypes(include=[np.number]).columns if c != "TARGET"]
            if not feature_order:
                st.error("Impossible de déterminer l'ordre des features pour ONNX.")
                return 0.0

        X = vectorize_for_onnx(row_dict, feature_order, reference_row, n_in if isinstance(n_in, int) else len(feature_order))
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: X})
        # ONNX sklearn (zipmap=False) retourne typiquement proba en premier ou second output selon l’opérateur
        # On récupère le premier array "probable" (ndim>=2) sinon on aplatit.
        proba_arr = None
        for out in outputs:
            if hasattr(out, "ndim") and out.ndim >= 2:
                proba_arr = out
                break
        if proba_arr is None:
            proba_arr = outputs[0]

        proba_arr = np.asarray(proba_arr)
        if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
            return float(proba_arr[0, 1])  # colonne 1 = classe positive
        return float(np.ravel(proba_arr)[0])

    st.error("Modèle non disponible.")
    return 0.0

# ==============================
# App
# ==============================
st.title("📊 Dashboard de Credit Scoring")

MODEL_FILE = "best_model.joblib"  # on garde cette extension
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
REFERENCE_ROW_CSV = "reference_row.csv"
HISTORY_FILE = "history.csv"

bundle = load_model_any(MODEL_FILE)
reference_row, ref_feature_order = load_reference_row(REFERENCE_ROW_CSV)

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

# Historique (persistant + session)
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state["history"] = []

# ==============================
# UI
# ==============================
st.header("⚙️ Sélection du Mode de Test")
mode = st.radio("Choisissez :", ["Client existant", "Nouveau client"])

# -------- Client existant --------
if bundle.get("kind") != "error" and mode == "Client existant":
    if not df_app.empty:
        max_index = len(df_app) - 1
        idx = st.number_input("Index du client (0 → {max_index})", min_value=0, max_value=max_index, value=0)
        client_row = df_app.iloc[idx].copy()
        true_target = client_row.get("TARGET", None)
        st.write("**Vérité terrain (TARGET)** :", true_target)

        # dict sans TARGET
        if "TARGET" in client_row:
            client_row = client_row.drop("TARGET")
        client_row = client_row.replace([np.nan, np.inf, -np.inf], 0)
        client_dict = client_row.to_dict()

        if st.button("⚡ Prédire ce client"):
            prob_default = predict_proba_generic(bundle, client_dict, df_app, reference_row, ref_feature_order)
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
                "decision": decision,
            }
            st.session_state["history"].append(record)
            append_to_csv(record, HISTORY_FILE)

        # --- Comparaison univariée
        st.subheader("📈 Comparaison univariée")
        if st.checkbox("Afficher histogramme comparaison"):
            columns_list = [c for c in df_app.columns if c != "TARGET"]
            feature = st.selectbox("Feature", columns_list, index=0)
            fig, ax = plt.subplots()
            # Histogramme coloré par TARGET (0 = bleu, 1 = rouge)
            sns.histplot(df_app, x=feature, hue="TARGET", palette={0: "blue", 1: "red"}, kde=True, ax=ax)
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

# -------- Nouveau client --------
elif bundle.get("kind") != "error" and mode == "Nouveau client":
    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0)
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0)

    if st.button("⚡ Prédire nouveau client"):
        prob_default = predict_proba_generic(bundle, new_client, df_app, reference_row, ref_feature_order)
        rep_str, def_str = display_probability(prob_default)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        record = {
            "Mode": "Nouveau client",
            "Données": new_client,
            "default_probability": prob_default,
            "decision": decision,
        }
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)

    # Comparaisons
    if not df_app.empty:
        st.subheader("📈 Comparaison univariée")
        if st.checkbox("Comparer histogramme (nouveau client)"):
            cols = [c for c in df_app.columns if c != "TARGET"]
            feature = st.selectbox("Feature", cols, index=0, key="new_univar")
            fig, ax = plt.subplots()
            sns.histplot(df_app, x=feature, hue="TARGET", palette={0: "blue", 1: "red"}, kde=True, ax=ax)
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

# -------- Historique --------
st.header("🗂️ Historique des Tests")
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()

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

# -------- Données générales --------
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"

fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV) if os.path.exists(FEATURE_IMPORTANCE_CSV) else pd.DataFrame()
th_df = pd.read_csv(THRESHOLDS_CSV) if os.path.exists(THRESHOLDS_CSV) else pd.DataFrame()
data_drift_available = os.path.exists(DATA_DRIFT_REPORT_HTML)

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
