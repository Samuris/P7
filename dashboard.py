# app.py
# =========================================================
# Dashboard Credit Scoring — Streamlit
# - Requête API (client existant / nouveau client)
# - Affichage FI, Thresholds (avec contrôle FN/FP) & Drift report
# - Graphiques univariés / bivariés
# - Historique persistant des tests
# Lancement : streamlit run app.py
# =========================================================

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# ------------------------------
# Config générale
# ------------------------------
st.set_page_config(page_title="Credit Scoring", layout="wide")

# API Flask locale (adapte si besoin)
API_URL = st.sidebar.text_input("API URL", value="http://127.0.0.1:5000/predict")

# Dossiers/fichiers par défaut
DATA_DIR = st.sidebar.text_input("Dossier données", value="./donnéecoupé")
APP_TRAIN_CSV = os.path.join(DATA_DIR, "application_train.csv")

# Artefacts (on essaie plusieurs noms: avec espace ou underscore)
def _cands(name_core, suffix):
    return [
        f"{name_core}{suffix}",
        f"{name_core.replace(' ', '_')}{suffix}",
        f"{name_core.replace(' ', '-')}{suffix}",
    ]

MODEL_BASENAME = st.sidebar.text_input("Nom modèle (pour lire les artefacts)", value="Gradient Boosting")

def find_first_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FI_FILE = find_first_file(_cands(MODEL_BASENAME, "_feature_importance.csv")) or st.sidebar.text_input(
    "FI CSV (optionnel)", value=f"{MODEL_BASENAME}_feature_importance.csv"
)
TH_FILE = find_first_file(_cands(MODEL_BASENAME, "_thresholds.csv")) or st.sidebar.text_input(
    "Thresholds CSV (optionnel)", value=f"{MODEL_BASENAME}_thresholds.csv"
)
DRIFT_HTML = st.sidebar.text_input("Drift HTML", value="artifacts/data_drift_report.html")

HISTORY_FILE = "history.csv"
NEW_CLIENTS_FILE = "new_clients.csv"

# Coûts métier (cohérents avec l’entraînement)
COST_FN, COST_FP, BENEFIT_TN, BENEFIT_TP = 10.0, 1.0, 1.0, 0.0

# ------------------------------
# Utilitaires
# ------------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def sample_feature(df: pd.DataFrame, feature: str, sample_size: int = 10_000) -> pd.Series:
    data = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > sample_size:
        return data.sample(sample_size, random_state=42)
    return data

def append_to_csv(record: dict, filename: str):
    df = pd.DataFrame([record])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, mode='w', index=False)

def display_probability(prob_default: float):
    prob_default_pct = 100 * prob_default
    prob_repay_pct = 100 * (1 - prob_default)
    if (1 - prob_default) >= 0.8:
        color_rep = "green"
    elif (1 - prob_default) >= 0.4:
        color_rep = "orange"
    else:
        color_rep = "red"
    rep = f"<span style='color:{color_rep}; font-size:20px;'>Probabilité de remboursement : {prob_repay_pct:.1f}%</span>"
    deff = f"<span style='color:{'red' if (1 - prob_default) < 0.4 else 'black'}; font-size:20px;'>Probabilité de défaut : {prob_default_pct:.1f}%</span>"
    return rep, deff

def read_feature_importance(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Tentatives de normalisation colonnes
    cols = [c.lower() for c in df.columns]
    if "feature" in cols and "importance" in cols:
        # déjà OK
        return df.rename(columns={df.columns[cols.index("feature")]: "Feature",
                                  df.columns[cols.index("importance")]: "Importance"})
    # Cas Series.to_csv: "Unnamed: 0", "0"
    if "unnamed: 0" in cols and ("0" in cols or "importance" in cols):
        fcol = df.columns[cols.index("unnamed: 0")]
        icol = df.columns[cols.index("0")] if "0" in cols else df.columns[cols.index("importance")]
        return df.rename(columns={fcol: "Feature", icol: "Importance"})[["Feature", "Importance"]]
    # Si deux colonnes quelconques, on les renomme
    if df.shape[1] == 2:
        df.columns = ["Feature", "Importance"]
        return df
    # Dernier recours: index = Feature
    if df.shape[1] == 1:
        return df.reset_index().rename(columns={"index": "Feature", df.columns[0]: "Importance"})
    return df

def read_thresholds(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

def gain_from_counts(tn, fp, fn, tp):
    return tn*BENEFIT_TN + tp*BENEFIT_TP - fp*COST_FP - fn*COST_FN

def call_api(payload: dict, url: str, timeout=15):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.json() if r.headers.get("Content-Type","").startswith("application/json") else r.text
    except Exception as e:
        return 599, str(e)

# ------------------------------
# Titre
# ------------------------------
st.title("Dashboard de Credit Scoring")

# ------------------------------
# 1) Chargements
# ------------------------------
colL, colR = st.columns([2,1])
with colL:
    if os.path.exists(APP_TRAIN_CSV):
        df_app = load_csv(APP_TRAIN_CSV)
        st.success(f"Dataset chargé : {APP_TRAIN_CSV} ({df_app.shape[0]} x {df_app.shape[1]})")
    else:
        st.error("application_train.csv introuvable.")
        df_app = pd.DataFrame()

with colR:
    st.markdown("**Artefacts détectés**")
    st.write("FI CSV :", FI_FILE if FI_FILE and os.path.exists(FI_FILE) else "—")
    st.write("TH CSV :", TH_FILE if TH_FILE and os.path.exists(TH_FILE) else "—")
    st.write("Drift :", DRIFT_HTML if os.path.exists(DRIFT_HTML) else "—")

fi_df = read_feature_importance(FI_FILE) if FI_FILE and os.path.exists(FI_FILE) else pd.DataFrame()
th_df_raw = read_thresholds(TH_FILE) if TH_FILE and os.path.exists(TH_FILE) else pd.DataFrame()

# Historique
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------
# 2) Mode de test
# ------------------------------
st.header("Sélection du Mode de Test")
mode = st.radio("Choisissez un mode :", options=["Client existant", "Nouveau client"], horizontal=True)

# ------------------------------
# 3) Client existant
# ------------------------------
if mode == "Client existant":
    st.subheader("Requêter un client du dataset (option: éditer)")

    if df_app.empty:
        st.warning("Dataset introuvable ou vide.")
    else:
        max_index = len(df_app) - 1
        row_index = st.number_input(f"Index (0..{max_index})", min_value=0, max_value=max_index, value=0)
        row = df_app.iloc[row_index].copy()
        true_target = row.get("TARGET", None)
        st.write(f"**Vérité terrain (TARGET)** : {true_target}")
        if "TARGET" in row:
            row = row.drop("TARGET")

        # Nettoyage pour l’envoi
        row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
        client_dict = row.to_dict()

        if st.checkbox("Modifier les données du client"):
            st.info("Saisissez les valeurs à écraser (vides = inchangé)")
            edited = {}
            for k, v in client_dict.items():
                if isinstance(v, (int, float, np.number)):
                    edited_val = st.text_input(k, value=str(v))
                    try:
                        edited[k] = float(edited_val)
                    except:
                        edited[k] = v
                else:
                    edited[k] = st.text_input(k, value=str(v))
            client_dict = edited

        if st.button("Requêter l’API"):
            code, out = call_api(client_dict, API_URL)
            if code == 200 and isinstance(out, dict):
                st.success("Réponse API OK")
                st.json(out)
                pdef = float(out.get("default_probability", 0))
                rep_html, def_html = display_probability(pdef)
                st.markdown(rep_html, unsafe_allow_html=True)
                st.markdown(def_html, unsafe_allow_html=True)

                # Feedback vs vérité terrain
                decision = out.get("decision", None)
                if decision is not None and true_target is not None:
                    if bool(true_target) == bool(decision):
                        st.success("Prédiction correcte vs vérité terrain.")
                    else:
                        st.warning("Prédiction incorrecte vs vérité terrain.")

                # Historique
                record = {"mode": "exist", "index": row_index,
                          "true_target": true_target, "prediction": json.dumps(out)}
                st.session_state["history"].append(record)
                append_to_csv(record, HISTORY_FILE)
            else:
                st.error(f"Appel API KO ({code})")
                st.code(str(out))

        # Univarié
        st.markdown("### Comparaison univariée")
        with st.expander("Afficher histogramme de comparaison", expanded=False):
            cols = [c for c in df_app.columns if c != "TARGET"]
            if cols:
                feat = st.selectbox("Feature", cols, index=0, key="uni_exist")
                pop = sample_feature(df_app, feat, 10_000)
                fig, ax = plt.subplots()
                sns.histplot(pop, kde=True, ax=ax, color="navy", label="Population")
                ax.axvline(client_dict.get(feat, 0), color="yellow", lw=2, label="Client")
                ax.set_title(f"Distribution de {feat}"); ax.legend()
                st.pyplot(fig)

        # Bivarié
        st.markdown("### Analyse bivariée")
        with st.expander("Afficher graphique bivarié", expanded=False):
            cols = [c for c in df_app.columns if c != "TARGET"]
            if len(cols) >= 2:
                x = st.selectbox("X", cols, index=0, key="bi_x_exist")
                y_ = st.selectbox("Y", cols, index=1, key="bi_y_exist")
                df_s = df_app[[x, y_]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(df_s) > 5000:
                    df_s = df_s.sample(5000, random_state=42)
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df_s, x=x, y=y_, ax=ax2, color="steelblue", alpha=0.5, label="Population")
                ax2.scatter(client_dict.get(x, np.nan), client_dict.get(y_, np.nan), color="orange", s=80, label="Client")
                ax2.set_title(f"{x} vs {y_}"); ax2.legend()
                st.pyplot(fig2)

# ------------------------------
# 4) Nouveau client
# ------------------------------
else:
    st.subheader("Créer et requêter un nouveau client")

    new_client = {}
    # Petit starter pack de champs utiles
    c1, c2, c3 = st.columns(3)
    with c1:
        new_client["AMT_INCOME_TOTAL"] = st.number_input("AMT_INCOME_TOTAL", value=200000.0)
    with c2:
        new_client["DAYS_BIRTH"] = st.number_input("DAYS_BIRTH (négatif)", value=-15000.0)
    with c3:
        new_client["DAYS_EMPLOYED"] = st.number_input("DAYS_EMPLOYED (négatif si en poste)", value=-3000.0)

    # Champs additionnels
    if not df_app.empty:
        with st.expander("Ajouter d’autres champs du dataset", expanded=False):
            cols = [c for c in df_app.columns if c != "TARGET"]
            add = st.multiselect("Sélectionnez des colonnes", cols, default=[])
            for c in add:
                val = st.text_input(c, value="0")
                try:
                    new_client[c] = float(val)
                except:
                    new_client[c] = val

    st.write("**Payload envoyé à l’API :**")
    st.json(new_client)

    if st.button("Requêter l’API (nouveau client)"):
        code, out = call_api(new_client, API_URL)
        if code == 200 and isinstance(out, dict):
            st.success("Réponse API OK")
            st.json(out)
            pdef = float(out.get("default_probability", 0))
            rep_html, def_html = display_probability(pdef)
            st.markdown(rep_html, unsafe_allow_html=True)
            st.markdown(def_html, unsafe_allow_html=True)

            rec = {"mode":"new", "payload": json.dumps(new_client), "prediction": json.dumps(out)}
            st.session_state["history"].append(rec)
            append_to_csv(rec, HISTORY_FILE)

            if st.checkbox("Ajouter ce client à un CSV permanent"):
                append_to_csv(new_client, NEW_CLIENTS_FILE)
                st.success(f"Ajouté dans {NEW_CLIENTS_FILE}")
        else:
            st.error(f"Appel API KO ({code})")
            st.code(str(out))

# ------------------------------
# 5) Historique (session + permanent)
# ------------------------------
st.header("Historique des Tests")
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()

session_df = pd.DataFrame(st.session_state["history"])
hist = pd.concat([persistent_history, session_df], ignore_index=True) if not session_df.empty else persistent_history
with st.expander("Afficher l'historique complet", expanded=False):
    st.dataframe(hist, height=400)

# ------------------------------
# 6) Données générales (FI, thresholds, drift)
# ------------------------------
st.header("Données Générales & Modèle")

# 6.1 Feature Importances
st.subheader("Feature Importances")
if not fi_df.empty:
    st.dataframe(fi_df, height=420)
    # Top 20 chart
    with st.expander("Top 20 (bar chart)", expanded=False):
        top = fi_df.sort_values("Importance", ascending=False).head(20)
        fig = px.bar(top, x="Importance", y="Feature", orientation="h")
        fig.update_layout(height=600, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Aucune FI chargée.")

# 6.2 Thresholds + contrôle FN/FP + graphe
st.subheader("Thresholds, Confusions & Gain (contrôle FP/FN)")
if not th_df_raw.empty:
    disp = th_df_raw.copy()

    # Renommer pour clarifier (si présent)
    ren = {}
    for k in ["tp","fp","fn","tn"]:
        if k in disp.columns:
            if k=="tp": ren[k] = "rejetes_mauvais(TP)"
            if k=="fp": ren[k] = "rejetes_bons(FP)"
            if k=="fn": ren[k] = "acceptes_mauvais(FN)"
            if k=="tn": ren[k] = "acceptes_bons(TN)"
    if ren:
        disp = disp.rename(columns=ren)

    # Recalcul du gain pour “auto-check”
    cn = disp.columns
    need = {"acceptes_bons(TN)","rejetes_bons(FP)","acceptes_mauvais(FN)","rejetes_mauvais(TP)"}
    if need.issubset(set(cn)):
        disp["gain_check"] = (
            disp["acceptes_bons(TN)"] * BENEFIT_TN
            + disp["rejetes_mauvais(TP)"] * BENEFIT_TP
            - disp["rejetes_bons(FP)"] * COST_FP
            - disp["acceptes_mauvais(FN)"] * COST_FN
        )
        if "gain_total" in disp.columns:
            disp["delta_gain"] = disp["gain_total"] - disp["gain_check"]

    st.dataframe(disp, height=420)

    # Courbe Gain vs Seuil si colonnes présentes
    if "threshold" in disp.columns and ("gain_total" in disp.columns or "gain_check" in disp.columns):
        st.markdown("**Gain vs Seuil**")
        gain_col = "gain_total" if "gain_total" in disp.columns else "gain_check"
        fig = px.line(disp.sort_values("threshold"), x="threshold", y=gain_col, markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Rappel sémantique :**  
- *rejetes_bons(FP)* = bon payeur refusé → **coût 1**  
- *acceptes_mauvais(FN)* = mauvais payeur accepté → **coût 10**  
- *acceptes_bons(TN)* = bon payeur accepté → **bénéfice 1**  
- *rejetes_mauvais(TP)* = mauvais payeur rejeté → **bénéfice 0** (perte évitée)  
""")
else:
    st.info("Aucun thresholds CSV chargé.")

# 6.3 Drift report
st.subheader("Rapport de Data Drift (Evidently)")
if DRIFT_HTML and os.path.exists(DRIFT_HTML):
    with open(DRIFT_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=600, scrolling=True)
else:
    st.info("Aucun rapport drift trouvé.")

# ------------------------------
# Footer
# ------------------------------
st.caption("© Projet P7 — Tableau de bord Streamlit")

