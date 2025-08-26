import streamlit as st
import joblib
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# ------------------------------
# Fonctions utilitaires et de cache
# ------------------------------
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
    """
    On considère ici que l'API renvoie "default_probability" qui correspond à la probabilité de défaut.
    On calcule donc la probabilité de remboursement = 1 - default_probability.
    Les couleurs sont choisies avec un fort contraste pour une meilleure accessibilité.
    """
    prob_default_pct = prob_default * 100
    prob_repayment_pct = (1 - prob_default) * 100
    # Pour une meilleure lisibilité (WCAG), on utilise :
    # - Vert pour une haute probabilité de remboursement (>=80%)
    # - Orange pour une probabilité modérée (40%-80%)
    # - Rouge pour une faible probabilité de remboursement (<40%)
    if (1 - prob_default) >= 0.8:
        color_rep = "green"
    elif (1 - prob_default) >= 0.4:
        color_rep = "orange"
    else:
        color_rep = "red"
    rep_str = f"<span style='color: {color_rep}; font-size: 20px;'>Probabilité de remboursement : {prob_repayment_pct:.1f} %</span>"
    def_str = f"<span style='color: {'red' if (1 - prob_default) < 0.4 else 'black'}; font-size: 20px;'>Probabilité de défaut : {prob_default_pct:.1f} %</span>"
    return rep_str, def_str

# Fichiers pour historique permanent
HISTORY_FILE = "history.csv"
NEW_CLIENTS_FILE = "new_clients.csv"

# ------------------------------
# 1. Configuration et Chargement des Artefacts
# ------------------------------

st.title("Dashboard de Credit Scoring")

# Chemins à adapter
GRADIENT_BOOSTING_MODEL = "best_model.joblib"
FEATURE_IMPORTANCE_CSV = "Gradient Boosting_feature_importance.csv"
THRESHOLDS_CSV = "Gradient Boosting_thresholds.csv"
DATA_DRIFT_REPORT_HTML = "data_drift_report.html"
APPLICATION_TRAIN_CSV = "./donnéecoupé/application_train.csv"
API_URL = "http://127.0.0.1:5000/predict"  # URL de l'API Flask

# Pour une interface épurée, nous ne montrons pas de détails superflus sur le modèle

# Chargement du dataset global
if os.path.exists(APPLICATION_TRAIN_CSV):
    df_app = load_csv(APPLICATION_TRAIN_CSV)
    st.success("Dataset client chargé.")
else:
    st.error("application_train.csv introuvable.")
    df_app = pd.DataFrame()

# Chargement des artefacts pour le menu "Données Générales"
fi_df = pd.read_csv(FEATURE_IMPORTANCE_CSV) if os.path.exists(FEATURE_IMPORTANCE_CSV) else pd.DataFrame()
th_df = pd.read_csv(THRESHOLDS_CSV) if os.path.exists(THRESHOLDS_CSV) else pd.DataFrame()
data_drift_available = os.path.exists(DATA_DRIFT_REPORT_HTML)

# Historique permanent et de session
if os.path.exists(HISTORY_FILE):
    persistent_history = pd.read_csv(HISTORY_FILE)
else:
    persistent_history = pd.DataFrame()

if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------
# 2. Sélection du Mode de Test
# ------------------------------

st.header("Sélection du Mode de Test")
mode = st.radio("Choisissez un mode :", options=["Client existant", "Nouveau client"])

# ------------------------------
# 3. Mode : Client existant
# ------------------------------
if mode == "Client existant":
    st.subheader("Rechercher et Modifier un Client Existant")
    if not df_app.empty:
        max_index = len(df_app) - 1
        row_index = st.number_input(f"Index du client (0 à {max_index})", min_value=0, max_value=max_index, value=0)
        
        # Extraire la ligne du client
        client_row = df_app.iloc[row_index].copy()
        true_target = client_row.get("TARGET", None)
        st.write(f"**Vérité terrain** : {true_target}")
        if "TARGET" in client_row:
            client_row = client_row.drop("TARGET")
        client_row = client_row.replace([np.nan, np.inf, -np.inf], 0)
        client_dict = client_row.to_dict()
        st.subheader("Données du Client Sélectionné")
        st.write(client_dict)
        
        # Modifier les données si besoin (affichage des inputs uniquement après avoir coché)
        if st.checkbox("Modifier les données du client"):
            st.write("Modifiez les champs ci-dessous :")
            edited_client = {}
            for key, val in client_dict.items():
                try:
                    val = float(val)
                    edited_client[key] = st.number_input(f"{key}", value=val, key=f"edit_{key}")
                except:
                    edited_client[key] = st.text_input(f"{key}", value=str(val), key=f"edit_{key}")
            client_dict = edited_client
            st.write("Données modifiées :", client_dict)
        
        if st.button("Requêter ce client via l'API"):
            response = requests.post(API_URL, json=client_dict)
            if response.status_code == 200:
                prediction = response.json()
                st.write("**Réponse de l'API** :", prediction)
                default_prob = prediction.get("default_probability", 0)
                rep_str, def_str = display_probability(default_prob)
                st.markdown(rep_str, unsafe_allow_html=True)
                st.markdown(def_str, unsafe_allow_html=True)
                predicted_decision = prediction.get("decision", None)
                if predicted_decision is not None and true_target is not None:
                    if bool(true_target) == bool(predicted_decision):
                        st.success("Le modèle a correctement prédit le résultat.")
                    else:
                        st.warning("Le modèle s'est trompé dans sa prédiction.")
                record = {
                    "Mode": "Client existant",
                    "Index": row_index,
                    "Vérité terrain": true_target,
                    "Prediction": prediction
                }
                st.session_state["history"].append(record)
                append_to_csv(record, HISTORY_FILE)
            else:
                st.error(f"Erreur requête : {response.status_code}")
                st.error(f"Message : {response.text}")
        
        # Comparaison univariée
        st.subheader("Comparaison univariée avec la Population")
        if st.checkbox("Afficher histogramme de comparaison"):
            columns_list = df_app.columns.tolist()
            if "TARGET" in columns_list:
                columns_list.remove("TARGET")
            selected_feature = st.selectbox("Feature à comparer", columns_list, index=0)
            data_pop = sample_feature(df_app, selected_feature, sample_size=10000)
            fig, ax = plt.subplots()
            # Couleurs accessibles : fond sombre pour la population, jaune pour le client
            sns.histplot(data_pop, kde=True, ax=ax, color='navy', label='Population')
            client_value = client_dict.get(selected_feature, 0)
            ax.axvline(client_value, color='yellow', linewidth=2, label='Client')
            ax.set_title(f"Distribution de {selected_feature}", fontsize=14)
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel("Fréquence", fontsize=12)
            ax.legend()
            st.pyplot(fig)
        
        # Analyse bivariée
        st.subheader("Analyse bivariée")
        if st.checkbox("Afficher graphique bivarié"):
            cols = df_app.columns.tolist()
            if "TARGET" in cols:
                cols.remove("TARGET")
            feature_x = st.selectbox("Sélectionnez la feature X", cols, index=0, key="existing_x")
            feature_y = st.selectbox("Sélectionnez la feature Y", cols, index=1, key="existing_y")
            sample_size = 5000
            df_sample = df_app[[feature_x, feature_y]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample) > sample_size:
                df_sample = df_sample.sample(sample_size, random_state=42)
            fig_scatter, ax_scatter = plt.subplots()
            sns.scatterplot(data=df_sample, x=feature_x, y=feature_y, ax=ax_scatter, label="Population", color="blue", alpha=0.5)
            client_x = client_dict.get(feature_x, np.nan)
            client_y = client_dict.get(feature_y, np.nan)
            ax_scatter.scatter(client_x, client_y, color="orange", s=100, label="Client")
            ax_scatter.set_title(f"Analyse bivariée: {feature_x} vs {feature_y}", fontsize=14)
            ax_scatter.set_xlabel(feature_x, fontsize=12)
            ax_scatter.set_ylabel(feature_y, fontsize=12)
            ax_scatter.legend()
            st.pyplot(fig_scatter)
    else:
        st.error("Dataset introuvable ou vide pour les clients existants.")

# ------------------------------
# 4. Mode : Nouveau client
# ------------------------------
elif mode == "Nouveau client":
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
            except:
                new_client[col] = val
    
    st.subheader("Données du Nouveau Client")
    st.write(new_client)
    
    if st.button("Requêter ce nouveau client via l'API"):
        response = requests.post(API_URL, json=new_client)
        if response.status_code == 200:
            prediction_new = response.json()
            st.write("**Réponse de l'API** :", prediction_new)
            default_prob_new = prediction_new.get("default_probability", 0)
            rep_str_new, def_str_new = display_probability(default_prob_new)
            st.markdown(rep_str_new, unsafe_allow_html=True)
            st.markdown(def_str_new, unsafe_allow_html=True)
            record = {
                "Mode": "Nouveau client",
                "Données": new_client,
                "Prediction": prediction_new
            }
            st.session_state["history"].append(record)
            append_to_csv(record, HISTORY_FILE)
            if st.checkbox("Ajouter ce client aux données ?"):
                append_to_csv(new_client, NEW_CLIENTS_FILE)
                st.success("Client ajouté aux données permanentes.")
        else:
            st.error(f"Erreur requête : {response.status_code}")
            st.error(f"Message : {response.text}")
    
    st.subheader("Comparaison univariée avec la Population (Nouveau Client)")
    if st.checkbox("Afficher histogramme de comparaison (nouveau client)"):
        if df_app.empty:
            st.info("Dataset introuvable ou vide, comparaison impossible.")
        else:
            columns_list = df_app.columns.tolist()
            if "TARGET" in columns_list:
                columns_list.remove("TARGET")
            selected_feature_new = st.selectbox("Feature à comparer", columns_list, index=0, key="new_feature")
            data_pop_new = sample_feature(df_app, selected_feature_new, sample_size=10000)
            fig_new, ax_new = plt.subplots()
            sns.histplot(data_pop_new, kde=True, ax=ax_new, color='navy', label='Population')
            client_value_new = new_client.get(selected_feature_new, 0)
            ax_new.axvline(client_value_new, color='yellow', linewidth=2, label='Nouveau Client')
            ax_new.set_title(f"Distribution de {selected_feature_new} (Nouveau Client)", fontsize=14)
            ax_new.set_xlabel(selected_feature_new, fontsize=12)
            ax_new.set_ylabel("Fréquence", fontsize=12)
            ax_new.legend()
            st.pyplot(fig_new)
    
    st.subheader("Analyse bivariée (Nouveau Client)")
    if st.checkbox("Afficher graphique bivarié (nouveau client)"):
        if df_app.empty:
            st.info("Dataset introuvable ou vide, comparaison impossible.")
        else:
            cols = df_app.columns.tolist()
            if "TARGET" in cols:
                cols.remove("TARGET")
            feature_x_new = st.selectbox("Sélectionnez la feature X", cols, index=0, key="new_x")
            feature_y_new = st.selectbox("Sélectionnez la feature Y", cols, index=1, key="new_y")
            sample_size = 5000
            df_sample_new = df_app[[feature_x_new, feature_y_new]].replace([np.nan, np.inf, -np.inf], 0)
            if len(df_sample_new) > sample_size:
                df_sample_new = df_sample_new.sample(sample_size, random_state=42)
            fig_scatter_new, ax_scatter_new = plt.subplots()
            sns.scatterplot(data=df_sample_new, x=feature_x_new, y=feature_y_new, ax=ax_scatter_new, label="Population", color="blue", alpha=0.5)
            client_x_new = new_client.get(feature_x_new, np.nan)
            client_y_new = new_client.get(feature_y_new, np.nan)
            ax_scatter_new.scatter(client_x_new, client_y_new, color="orange", s=100, label="Nouveau Client")
            ax_scatter_new.set_title(f"Analyse bivariée: {feature_x_new} vs {feature_y_new}", fontsize=14)
            ax_scatter_new.set_xlabel(feature_x_new, fontsize=12)
            ax_scatter_new.set_ylabel(feature_y_new, fontsize=12)
            ax_scatter_new.legend()
            st.pyplot(fig_scatter_new)

# ------------------------------
# 5. Historique des Tests (Permanent)
# ------------------------------
st.header("Historique des Tests (Permanent)")
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
    # Indicateur de performance : moyenne de la probabilité de remboursement
    probs = []
    # On tente d'extraire "default_probability" de chaque enregistrement dans l'historique de session
    for rec in st.session_state["history"]:
        pred = rec.get("Prediction", {})
        if isinstance(pred, dict) and "default_probability" in pred:
            probs.append(1 - pred["default_probability"])  # Probabilité de remboursement
    if probs:
        avg_repayment = sum(probs)/len(probs)
        st.metric(label="Performance Moyenne (Probabilité de remboursement)", value=f"{avg_repayment*100:.1f}%")
    else:
        st.info("Pas d'indicateur de performance disponible.")

# ------------------------------
# 6. Données Générales (Données d'Analyse)
# ------------------------------
with st.expander("Afficher les Données Générales", expanded=False):
    st.subheader("Feature Importance Complète et Signification")
    if not fi_df.empty:
        st.dataframe(fi_df)
        top_features = fi_df.sort_values(by="Importance", ascending=False).head(10)
        explanations = {
            "AMT_INCOME_TOTAL": "Revenu total du client.",
            "DAYS_BIRTH": "Âge du client (en jours, négatif).",
            "DAYS_EMPLOYED": "Nombre de jours d'emploi (négatif si en cours).",
            # Ajoutez d'autres explications selon vos données
        }
        st.write("**Signification des Top Features :**")
        for idx, row in top_features.iterrows():
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
