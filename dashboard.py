# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# on garde joblib seulement pour compat de nom, mais on unpickle nous-mêmes
import joblib  # noqa: F401
from matplotlib.patches import Patch

# onnxruntime optionnel (au cas où best_model.joblib contient {'onnx_bytes': ...})
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
        # Tolère des symboles cythonisés qui n'existent pas selon la version
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

def draw_client_marker(ax, x, color="yellow", label="Client sélectionné"):
    """
    Marque la position du client :
    - bande verticale semi-transparente centrée sur x
    - chevron en haut
    - étiquette au-dessus de la bande
    """
    # largeur de bin estimée si déjà dispo, sinon 1% de l'axe
    bin_w = None
    for p in ax.patches:
        if hasattr(p, "get_width"):
            w = p.get_width()
            if w and w > 0:
                bin_w = w
                break
    xlim = ax.get_xlim()
    if bin_w is None:
        bin_w = (xlim[1] - xlim[0]) * 0.01

    # Bande + chevron
    ax.axvspan(x - bin_w/2, x + bin_w/2, color=color, alpha=0.25, zorder=2.5)
    ylim = ax.get_ylim()
    y_tip = ylim[1] * 0.92
    ax.plot([x], [y_tip], marker="v", markersize=10, color=color, clip_on=False, zorder=3)

    # Étiquette compacte
    ax.annotate(f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else str(x),
                xy=(x, y_tip), xytext=(0, 14), textcoords="offset points",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.25),
                color="black", fontsize=9, zorder=3)

    # Légende (entrée discrète)
    handles, labels = ax.get_legend_handles_labels()
    if label not in labels:
        handles.append(Patch(facecolor=color, alpha=0.25, label=label))
        labels.append(label)
        ax.legend(handles=handles, labels=labels)

# ==============================
# Chargement reference_row
# ==============================
def load_reference_row(path="reference_row.csv"):
    if os.path.exists(path):
        try:
            ref_df = pd.read_csv(path, nrows=1)
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

    # Cas wrapper ONNX {'onnx_bytes': ..., 'feature_order': [...]}
    if isinstance(obj, dict) and "onnx_bytes" in obj:
        if not HAS_ORT:
            st.error("Le fichier contient un modèle ONNX mais onnxruntime n'est pas installé.")
            return {"kind": "error", "err": "onnxruntime_missing"}
        try:
            sess = ort.InferenceSession(obj["onnx_bytes"], providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            n_in = shape[-1] if isinstance(shape, (list, tuple)) else None
            if not isinstance(n_in, int):
                n_in = None
            st.success("✅ Modèle ONNX chargé (via best_model.joblib).")
            feat_order = obj.get("feature_order", None)
            feat_order = list(feat_order) if isinstance(feat_order, (list, tuple)) else None
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
    Repro de l'API Flask :
    - si expected_features connu: DF avec ces colonnes (complète via reference_row, sinon 0)
    - sinon: DF direct + complétion via reference_row
    - cast numérique si possible (sinon on laisse pour que le pipeline gère)
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
        if reference_row:
            for k, v in reference_row.items():
                if k not in X.columns:
                    X[k] = v

    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            pass
    return X.fillna(0)

def vectorize_for_onnx(row: dict, feature_order: list, reference_row: dict, n_features: int):
    """
    Vecteur float32 (1, n_features) dans l'ordre donné.
    """
    vals = []
    for feat in feature_order:
        v = row.get(feat, reference_row.get(feat, 0))
        try:
            v = float(v)
        except Exception:
            v = 0.0
        vals.append(v)
    arr = np.asarray(vals, dtype=np.float32)[None, :]

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
        feature_order = bundle.get("feature_order") or ref_feature_order
        if not feature_order:
            feature_order = [c for c in ref_df.select_dtypes(include=[np.number]).columns if c != "TARGET"]
            if not feature_order:
                st.error("Impossible de déterminer l'ordre des features pour ONNX.")
                return 0.0

        X = vectorize_for_onnx(row_dict, feature_order, reference_row, n_in if isinstance(n_in, int) else len(feature_order))
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: X})

        proba_arr = None
        for out in outputs:
            if hasattr(out, "ndim") and out.ndim >= 2:
                proba_arr = out
                break
        if proba_arr is None:
            proba_arr = outputs[0]

        proba_arr = np.asarray(proba_arr)
        if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
            return float(proba_arr[0, 1])
        return float(np.ravel(proba_arr)[0])

    st.error("Modèle non disponible.")
    return 0.0

# ==============================
# App
# ==============================
st.title("Dashboard de Credit Scoring")

MODEL_FILE = "best_model.joblib"              # on garde cette extension
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
REFERENCE_ROW_CSV = "reference_row.csv"
HISTORY_FILE = "history.csv"
NEW_CLIENTS_FILE = "new_clients.csv"

bundle = load_model_any(MODEL_FILE)
reference_row, ref_feature_order = load_reference_row(REFERENCE_ROW_CSV)

# Dataset global
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("application_train.csv introuvable.")
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

# ==============================
# Mode
# ==============================
st.header("Sélection du Mode de Test")
mode = st.radio("Choisissez un mode :", options=["Client existant", "Nouveau client"])

# Palette cible
target_palette = {0: "tab:blue", 1: "tab:red"}
target_names = {0: "Crédit remboursé (0)", 1: "Défaut (1)"}

# ==============================
# Client existant
# ==============================
if bundle.get("kind") != "error" and mode == "Client existant":
    if not df_app.empty:
        max_index = len(df_app) - 1
        row_index = st.number_input(f"Index du client (0 à {max_index})", min_value=0, max_value=max_index, value=0)

        client_row = df_app.iloc[row_index].copy()
        true_target = client_row.get("TARGET", None)
        st.write(f"**Vérité terrain** : {true_target}")
        if "TARGET" in client_row:
            client_row = client_row.drop("TARGET")
        client_row = client_row.replace([np.nan, np.inf, -np.inf], 0)
        client_dict = client_row.to_dict()

        st.subheader("Données du Client Sélectionné")
        st.write(client_dict)

        # ----- Édition des données du client -----
        if st.checkbox("Modifier les données du client"):
            st.write("Modifiez les champs ci-dessous :")
            edited_client = {}
            for key, val in client_dict.items():
                try:
                    val = float(val)
                    edited_client[key] = st.number_input(f"{key}", value=val, key=f"edit_{key}")
                except Exception:
                    edited_client[key] = st.text_input(f"{key}", value=str(val), key=f"edit_{key}")
            client_dict = edited_client
            st.write("Données modifiées :", client_dict)

        # ----- Prédiction locale -----
        if st.button("⚡ Prédire ce client"):
            prob_default = predict_proba_generic(bundle, client_dict, df_app, reference_row, ref_feature_order)
            rep_str, def_str = display_probability(prob_default)
            st.markdown(rep_str, unsafe_allow_html=True)
            st.markdown(def_str, unsafe_allow_html=True)
            decision = prob_default >= 0.5
            if true_target is not None:
                st.success("Le modèle a correctement prédit le résultat.") if bool(true_target) == decision else st.warning("Le modèle s'est trompé dans sa prédiction.")
            record = {
                "Mode": "Client existant",
                "Index": row_index,
                "Vérité terrain": true_target,
                "default_probability": prob_default,
                "decision": decision
            }
            st.session_state["history"].append(record)
            append_to_csv(record, HISTORY_FILE)

        # ----- Comparaison univariée -----
        st.subheader("Comparaison univariée avec la Population")
        if st.checkbox("Afficher histogramme de comparaison"):
            columns_list = df_app.columns.tolist()
            if "TARGET" in columns_list:
                columns_list.remove("TARGET")
            selected_feature = st.selectbox("Feature à comparer", columns_list, index=0)

            filtre_target = st.selectbox("Filtrer par TARGET", ["Tous", "0 (Crédit OK)", "1 (Défaut)"], index=0)
            df_plot = df_app.copy()
            if filtre_target.startswith("0"):
                df_plot = df_plot[df_plot["TARGET"] == 0]
            elif filtre_target.startswith("1"):
                df_plot = df_plot[df_plot["TARGET"] == 1]

            fig, ax = plt.subplots()
            if filtre_target == "Tous":
                sns.histplot(df_app, x=selected_feature, hue="TARGET",
                             palette=target_palette, kde=True, ax=ax, alpha=0.6, element="step")
            else:
                sns.histplot(df_plot, x=selected_feature, kde=True, ax=ax, color="tab:blue" if "0" in filtre_target else "tab:red", alpha=0.6, element="step")
            client_value = client_dict.get(selected_feature, 0)
            draw_client_marker(ax, client_value, color="yellow", label="Client sélectionné")
            ax.set_title(f"Distribution de {selected_feature}", fontsize=14)
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel("Fréquence", fontsize=12)
            st.pyplot(fig)

        # ----- Analyse bivariée -----
        st.subheader("Analyse bivariée")
        if st.checkbox("Afficher graphique bivarié"):
            cols = df_app.columns.tolist()
            if "TARGET" in cols:
                cols.remove("TARGET")
            feature_x = st.selectbox("Sélectionnez la feature X", cols, index=0, key="existing_x")
            feature_y = st.selectbox("Sélectionnez la feature Y", cols, index=1, key="existing_y")
            sample_size = 5000
            df_sample = df_app[[feature_x, feature_y, "TARGET"]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample) > sample_size:
                df_sample = df_sample.sample(sample_size, random_state=42)
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=df_sample, x=feature_x, y=feature_y, hue="TARGET",
                            palette=target_palette, alpha=0.5, ax=ax_scatter)
            client_x = client_dict.get(feature_x, np.nan)
            client_y = client_dict.get(feature_y, np.nan)
            ax_scatter.scatter(client_x, client_y, color="orange", s=120, label="Client")
            ax_scatter.set_title(f"Analyse bivariée: {feature_x} vs {feature_y}", fontsize=14)
            ax_scatter.set_xlabel(feature_x, fontsize=12)
            ax_scatter.set_ylabel(feature_y, fontsize=12)
            ax_scatter.legend()
            st.pyplot(fig_scatter)
    else:
        st.error("Dataset introuvable ou vide pour les clients existants.")

# ==============================
# Nouveau client
# ==============================
elif bundle.get("kind") != "error" and mode == "Nouveau client":
    st.subheader("Créer un Nouveau Client")

    new_client = {}
    new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0, key="new_AMT_INCOME_TOTAL")
    new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH", value=-15000.0, key="new_DAYS_BIRTH")
    new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED", value=-3000.0, key="new_DAYS_EMPLOYED")

    st.write("Ajouter d'autres champs (optionnel) :")
    if not df_app.empty:
        available_cols = df_app.columns.tolist()
        if "TARGET" in available_cols:
            available_cols.remove("TARGET")
        additional_fields = st.multiselect("Sélectionnez d'autres champs à ajouter", available_cols, default=[])
        for col in additional_fields:
            val = st.text_input(f"{col}", value="0", key=f"new_{col}")
            try:
                new_client[col] = float(val)
            except Exception:
                new_client[col] = val

    st.subheader("Données du Nouveau Client")
    st.write(new_client)

    if st.button("⚡ Prédire ce nouveau client"):
        prob_default = predict_proba_generic(bundle, new_client, df_app, reference_row, ref_feature_order)
        rep_str_new, def_str_new = display_probability(prob_default)
        st.markdown(rep_str_new, unsafe_allow_html=True)
        st.markdown(def_str_new, unsafe_allow_html=True)
        decision = prob_default >= 0.5
        record = {
            "Mode": "Nouveau client",
            "Données": new_client,
            "default_probability": prob_default,
            "decision": decision
        }
        st.session_state["history"].append(record)
        append_to_csv(record, HISTORY_FILE)
        if st.checkbox("Ajouter ce client aux données ?"):
            append_to_csv(new_client, NEW_CLIENTS_FILE)
            st.success("Client ajouté aux données permanentes.")

    # Comparaisons possibles si dataset dispo
    if not df_app.empty:
        st.subheader("Comparaison univariée avec la Population (Nouveau Client)")
        if st.checkbox("Afficher histogramme de comparaison (nouveau client)"):
            columns_list = df_app.columns.tolist()
            if "TARGET" in columns_list:
                columns_list.remove("TARGET")
            selected_feature_new = st.selectbox("Feature à comparer", columns_list, index=0, key="new_feature")
            fig_new, ax_new = plt.subplots()
            sns.histplot(df_app, x=selected_feature_new, hue="TARGET",
                         palette=target_palette, kde=True, ax=ax_new, alpha=0.6, element="step")
            client_value_new = new_client.get(selected_feature_new, 0)
            draw_client_marker(ax_new, client_value_new, color="orange", label="Nouveau client")
            ax_new.set_title(f"Distribution de {selected_feature_new} (Nouveau Client)", fontsize=14)
            ax_new.set_xlabel(selected_feature_new, fontsize=12)
            ax_new.set_ylabel("Fréquence", fontsize=12)
            st.pyplot(fig_new)

        st.subheader("Analyse bivariée (Nouveau Client)")
        if st.checkbox("Afficher graphique bivarié (nouveau client)"):
            cols = df_app.columns.tolist()
            if "TARGET" in cols:
                cols.remove("TARGET")
            feature_x_new = st.selectbox("Sélectionnez la feature X", cols, index=0, key="new_x")
            feature_y_new = st.selectbox("Sélectionnez la feature Y", cols, index=1, key="new_y")
            df_sample_new = df_app[[feature_x_new, feature_y_new, "TARGET"]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample_new) > 5000:
                df_sample_new = df_sample_new.sample(5000, random_state=42)
            fig_scatter_new, ax_scatter_new = plt.subplots()
            sns.scatterplot(data=df_sample_new, x=feature_x_new, y=feature_y_new, hue="TARGET",
                            palette=target_palette, alpha=0.5, ax=ax_scatter_new)
            ax_scatter_new.scatter(new_client.get(feature_x_new, np.nan),
                                   new_client.get(feature_y_new, np.nan),
                                   color="orange", s=120, label="Nouveau client")
            ax_scatter_new.set_title(f"Analyse bivariée: {feature_x_new} vs {feature_y_new}", fontsize=14)
            ax_scatter_new.set_xlabel(feature_x_new, fontsize=12)
            ax_scatter_new.set_ylabel(feature_y_new, fontsize=12)
            ax_scatter_new.legend()
            st.pyplot(fig_scatter_new)

# ==============================
# Historique des Tests (Permanent)
# ==============================
st.header("Historique des Tests (Permanent)")
if st.session_state["history"]:
    session_history = pd.DataFrame(st.session_state["history"])
    combined_history = pd.concat([persistent_history, session_history], ignore_index=True)
else:
    combined_history = persistent_history

with st.expander("Afficher l'historique complet des tests", expanded=True):
    st.dataframe(combined_history, height=500)
    if "default_probability" in combined_history:
        avg_repayment = (1 - combined_history["default_probability"]).mean()
        st.metric(label="Performance Moyenne (Probabilité de remboursement)", value=f"{avg_repayment*100:.1f}%")
    else:
        st.info("Pas d'indicateur de performance disponible.")

# ==============================
# Données Générales (Données d'Analyse)
# ==============================
with st.expander("Afficher les Données Générales", expanded=False):
    st.subheader("Feature Importance Complète et Signification")
    if not fi_df.empty:
        st.dataframe(fi_df)
        top_features = fi_df.sort_values(by="Importance", ascending=False).head(10)
        explanations = {
            "AMT_INCOME_TOTAL": "Revenu total du client.",
            "DAYS_BIRTH": "Âge du client (en jours, négatif).",
            "DAYS_EMPLOYED": "Nombre de jours d'emploi (négatif si en cours).",
        }
        st.write("**Signification des Top Features :**")
        for _, row in top_features.iterrows():
            feat = row["Feature"]
            imp = row["Importance"]
            exp = explanations.get(feat, "Pas d'explication fournie.")
            st.markdown(f"- **{feat}** (Importance: {imp:.3f}) : {exp}")
    else:
        st.info("Aucune feature importance à afficher.")

    st.subheader("Thresholds Complète")
    if not th_df.empty:
        st.dataframe(th_df)
    else:
        st.info("Aucun thresholds à afficher.")

    st.subheader("Rapport de Data Drift")
    if data_drift_available:
        with open(DATA_DRIFT_REPORT_HTML, 'r', encoding='utf-8') as f:
            data_drift_html = f.read()
        st.components.v1.html(data_drift_html, height=600, scrolling=True)
    else:
        st.info("Aucun rapport de data drift disponible.")
