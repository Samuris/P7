import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# essaye d'importer onnxruntime proprement
try:
    import onnxruntime as ort
except Exception as _e:
    ort = None

# ======================================================
# Utils
# ======================================================
@st.cache_data
def load_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def append_to_csv(record: dict, filename: str):
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, mode='w', index=False)

def display_probability(prob_default: float):
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
# Loader qui GARDE l’extension .joblib et tente ONNX
# ======================================================
def _bytes_look_like_onnx(b: bytes) -> bool:
    # heuristique très simple
    return (b[:4] == b"ONNX") or (b.find(b"ir_version") != -1 and b.find(b"graph") != -1)

def _try_build_session_from_bytes(b: bytes):
    if ort is None:
        raise RuntimeError("onnxruntime n'est pas installé/disponible.")
    return ort.InferenceSession(b, providers=["CPUExecutionProvider"])

@st.cache_resource
def load_session_from_joblib(joblib_path: str):
    """
    1) Essaye directement comme si c'était un ONNX (extension gardée .joblib)
    2) Sinon, unpickle SAFE pour extraire onnx_bytes / onnx_path si présent
    3) Sinon -> erreur claire
    """
    if ort is None:
        raise RuntimeError("onnxruntime n'est pas disponible dans l'environnement.")

    # Tentative 1: le fichier .joblib EST en fait un ONNX renommé
    try:
        sess = ort.InferenceSession(joblib_path, providers=["CPUExecutionProvider"])
        return ("onnx_file_path", sess)
    except Exception:
        pass

    # Tentative 2: lire les bytes et re-tester comme bytes ONNX
    try:
        with open(joblib_path, "rb") as f:
            raw = f.read()
        if _bytes_look_like_onnx(raw):
            sess = _try_build_session_from_bytes(raw)
            return ("onnx_bytes_in_joblib", sess)
    except Exception:
        pass

    # Tentative 3: unpickle sécurisé pour trouver un dict simple
    # (sans charger sklearn : on n'autorise QUE des objets builtin)
    import pickle

    class OnlyBuiltinsUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Interdit tout import exotique
            raise pickle.UnpicklingError("Import bloqué: %s.%s" % (module, name))

    try:
        with open(joblib_path, "rb") as f:
            obj = OnlyBuiltinsUnpickler(f).load()
        # on accepte uniquement dict/bytes/str simples
        if isinstance(obj, dict):
            # 3a) onnx_bytes
            if "onnx_bytes" in obj and isinstance(obj["onnx_bytes"], (bytes, bytearray)):
                sess = _try_build_session_from_bytes(obj["onnx_bytes"])
                return ("onnx_bytes_key", sess)
            # 3b) onnx_path relatif (à côté du joblib)
            if "onnx_path" in obj and isinstance(obj["onnx_path"], str):
                onnx_path = obj["onnx_path"]
                if not os.path.isabs(onnx_path):
                    onnx_path = os.path.join(os.path.dirname(joblib_path), onnx_path)
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                return ("onnx_path_key", sess)
    except Exception:
        pass

    raise RuntimeError(
        "Impossible d'utiliser best_model.joblib sans sklearn.\n"
        "Soit le fichier est un vrai pickle sklearn (non portable), soit il ne contient pas d'ONNX interne.\n"
        "Solution robuste: exporter le pipeline complet en ONNX, ou emballer les bytes ONNX dans le joblib "
        "via {'onnx_bytes': <...>}."
    )

# ======================================================
# Prétraitement (get_dummies gabarit depuis df_app)
# ======================================================
def fit_dummy_template(df_app: pd.DataFrame, target_col: str = "TARGET"):
    base = df_app.drop(columns=[target_col], errors="ignore").copy()
    cat_cols = base.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in base.columns if c not in cat_cols]
    dummies = pd.get_dummies(base[cat_cols].astype(str), dummy_na=False)
    template_cols = num_cols + list(dummies.columns)
    return num_cols, cat_cols, template_cols

def transform_with_template(X: pd.DataFrame, num_cols, cat_cols, template_cols) -> pd.DataFrame:
    Xc = X.copy()
    # numériques
    num_part = {}
    for c in num_cols:
        v = pd.to_numeric(Xc[c], errors="coerce") if c in Xc.columns else pd.Series([0], index=Xc.index)
        num_part[c] = v.fillna(0.0).astype(np.float32)
    num_df = pd.DataFrame(num_part, index=X.index)
    # catégorielles
    cat_df_input = pd.DataFrame(index=X.index)
    for c in cat_cols:
        cat_df_input[c] = Xc[c].astype(str) if c in Xc.columns else ""
    dummies = pd.get_dummies(cat_df_input.astype(str), dummy_na=False)
    out = pd.concat([num_df, dummies], axis=1)
    out = out.reindex(columns=template_cols, fill_value=0).astype(np.float32)
    return out

def _align_to_expected_dim(sess, X_mat: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = X_mat.to_numpy(dtype=np.float32) if isinstance(X_mat, pd.DataFrame) else X_mat.astype(np.float32)
    exp_dim = sess.get_inputs()[0].shape[1]
    if exp_dim is None:
        return arr
    cur_dim = arr.shape[1]
    if cur_dim == exp_dim:
        return arr
    if cur_dim < exp_dim:
        pad = np.zeros((arr.shape[0], exp_dim - cur_dim), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)
    return arr[:, :exp_dim]

def run_onnx_raw(sess, X_np: np.ndarray):
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: X_np})

def pick_proba(outputs, proba_col: int, debug: bool=False) -> float:
    proba = outputs[0]
    if isinstance(proba, dict):
        # zipmap=True → dict {label: prob}
        # on essaye clé "1" puis 1
        for key in ("1", 1):
            if key in proba:
                return float(proba[key])
        # sinon prend le max
        return float(list(proba.values())[1 if proba_col == 1 and len(proba) > 1 else 0])

    if isinstance(proba, list) and len(proba) and isinstance(proba[0], dict):
        d = proba[0]
        for key in ("1", 1):
            if key in d:
                return float(d[key])
        return float(list(d.values())[1 if proba_col == 1 and len(d) > 1 else 0])

    if isinstance(proba, np.ndarray):
        if proba.ndim == 2:
            c = max(0, min(proba.shape[1]-1, proba_col))
            return float(proba[0, c])
        if proba.ndim == 1:
            return float(proba[0])

    if isinstance(proba, list):
        v = proba[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            v = v[proba_col] if len(v) > proba_col else v[0]
        return float(v)

    raise ValueError(f"Format sortie ONNX non géré: {type(proba)} / shape={getattr(proba, 'shape', None)}")

@st.cache_resource
def infer_best_proba_col(_sess, df_app: pd.DataFrame, num_cols, cat_cols, template_cols, sample_n: int = 256) -> int:
    sess = _sess  # éviter UnhashableParamError
    if df_app.empty or "TARGET" not in df_app.columns:
        return 1
    sample = df_app.drop(columns=["TARGET"]).head(sample_n)
    y = df_app["TARGET"].head(len(sample)).astype(int).to_numpy()
    X_t = transform_with_template(sample, num_cols, cat_cols, template_cols)
    X_np = _align_to_expected_dim(sess, X_t)
    outs = run_onnx_raw(sess, X_np)
    # essaie col 1 puis col 0
    try:
        p1 = np.array([pick_proba([o], 1) for o in outs[0]])  # si ndarray Nx2 → pick_proba gérera
    except Exception:
        p1 = np.zeros(len(y))
    try:
        p0 = np.array([pick_proba([o], 0) for o in outs[0]])
    except Exception:
        p0 = np.zeros(len(y))
    acc1 = ((p1 >= 0.5).astype(int) == y).mean() if len(p1) else 0
    acc0 = ((p0 >= 0.5).astype(int) == y).mean() if len(p0) else 0
    return int(acc1 >= acc0)

# ======================================================
# 1) Config
# ======================================================
st.title("📊 Dashboard de Credit Scoring")
MODEL_FILE = "best_model.joblib"   # 👉 on GARDE cette extension
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
HISTORY_FILE = "history.csv"

# Charger session ONNX depuis le .joblib
try:
    src_kind, sess = load_session_from_joblib(MODEL_FILE)
    st.success(f"Modèle chargé depuis {src_kind}.")
except Exception as e:
    st.error(f"❌ Impossible d'utiliser {MODEL_FILE} sans sklearn : {e}")
    st.stop()

# Dataset global (sert de gabarit)
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("❌ application_train.csv introuvable.")
    df_app = pd.DataFrame()

# Gabarit get_dummies
if not df_app.empty:
    NUM_COLS, CAT_COLS, TEMPLATE_COLS = fit_dummy_template(df_app)
else:
    NUM_COLS, CAT_COLS, TEMPLATE_COLS = [], [], []

# Artefacts (affichage)
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

# Sidebar debug + colonne proba
with st.sidebar:
    debug_mode = st.checkbox("🔧 Mode debug ONNX", value=False)
    auto_col = infer_best_proba_col(sess, df_app, NUM_COLS, CAT_COLS, TEMPLATE_COLS) if not df_app.empty else 1
    st.caption(f"Colonne proba détectée: {auto_col} (0 ou 1)")
    invert = st.checkbox("Inverser colonne proba ?", value=False)
    PROBA_COL = auto_col ^ int(invert)
    st.caption(f"Colonne proba utilisée: {PROBA_COL}")

# ======================================================
# 2) Mode
# ======================================================
st.header("⚙️ Sélection du Mode de Test")
mode = st.radio("Choisissez :", ["Client existant", "Nouveau client"])

# ======================================================
# 3) Client existant
# ======================================================
if mode == "Client existant" and not df_app.empty:
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
        X_t = transform_with_template(X, NUM_COLS, CAT_COLS, TEMPLATE_COLS)
        X_np = _align_to_expected_dim(sess, X_t)
        outs = run_onnx_raw(sess, X_np)
        prob_default = pick_proba(outs, PROBA_COL, debug=debug_mode)

        rep_str, def_str = display_probability(prob_default)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        if true_target is not None:
            st.success("✅ Bonne prédiction") if bool(true_target) == decision else st.warning("⚠️ Mauvaise prédiction")
        record = {"Mode": "Client existant", "Index": idx,
                  "Vérité terrain": true_target,
                  "default_probability": prob_default,
                  "decision": decision}
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)

    # Univarié
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

    # Bivarié
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
# 4) Nouveau client
# ======================================================
elif mode == "Nouveau client":
    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0)
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0)

    if st.button("⚡ Prédire nouveau client"):
        X = pd.DataFrame([new_client])
        X_t = transform_with_template(X, NUM_COLS, CAT_COLS, TEMPLATE_COLS)
        X_np = _align_to_expected_dim(sess, X_t)
        outs = run_onnx_raw(sess, X_np)
        prob_default = pick_proba(outs, PROBA_COL, debug=debug_mode)

        rep_str, def_str = display_probability(prob_default)
        st.markdown(rep_str, unsafe_allow_html=True)
        st.markdown(def_str, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        record = {"Mode": "Nouveau client", "Données": new_client,
                  "default_probability": prob_default,
                  "decision": decision}
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

# ======================================================
# 5) Historique
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
# 6) Données générales
# ======================================================
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
