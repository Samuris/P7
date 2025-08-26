from flask import Flask, request, jsonify
import os
import pandas as pd

# ==== Chargement modèle: skops -> joblib (fallback) ====
model = None
expected_features = []
load_error = None

def load_model_any(model_path="best_model.joblib"):
    global model, expected_features, load_error
    try:
        # 1) Si un .skops est là, on le privilégie (portable inter-versions)
        skops_path = os.path.splitext(model_path)[0] + ".skops"
        if os.path.exists(skops_path):
            import skops.io as sio
            model = sio.load(skops_path)
        else:
            import joblib
            model = joblib.load(model_path)

        # Récupération des features attendues si dispo
        # (tous les estimateurs ne l'exposent pas)
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is None:
            # Option: si tu as sauvegardé une liste ailleurs (ex: CSV/JSON), charge ici.
            expected_features = []
        else:
            expected_features = list(expected_features)

        print("Modèle chargé avec succès.")
        if expected_features:
            print("Colonnes attendues par le modèle :", expected_features)

    except Exception as e:
        load_error = str(e)
        model = None
        expected_features = []
        print(f"Erreur lors du chargement du modèle : {e}")

# ==== Charger la ligne de référence ====
def load_reference_row(path="reference_row.csv"):
    try:
        row = pd.read_csv(path).iloc[0].to_dict()
        print("Ligne de référence chargée avec succès.")
        return row
    except Exception as e:
        print(f"Erreur lors du chargement de la ligne de référence : {e}")
        return {}

reference_row = load_reference_row()
load_model_any("best_model.joblib")

# ==== Flask app ====
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None else "error"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "expected_features_count": len(expected_features),
        "load_error": load_error
    }), (200 if model is not None else 500)

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "expected_features": expected_features,
        "has_reference_row": bool(reference_row),
    })

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Si on ne connaît pas les features attendues (None/[]), on passe le DF tel quel.
    if not expected_features:
        return df

    # Complète les colonnes manquantes avec la référence (ou 0 par défaut)
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = reference_row.get(feat, 0)

    # Supprime les colonnes en trop (non attendues)
    df = df[[f for f in expected_features]]

    return df

def predict_proba_safely(mdl, X: pd.DataFrame):
    # Certains modèles n'ont pas predict_proba (ex: SVM linéaire sans probas)
    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X)
        return float(proba[0, 1])  # classe positive
    elif hasattr(mdl, "decision_function"):
        import numpy as np
        # Sigmoïde sur la distance (approx): 1 / (1 + exp(-score))
        scores = mdl.decision_function(X)
        return float(1.0 / (1.0 + np.exp(-scores[0])))
    else:
        # Dernier recours: prédiction binaire -> proba “0 ou 1”
        y = mdl.predict(X)
        return float(y[0])

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Le modèle n'a pas pu être chargé.", "detail": load_error}), 500

    try:
        input_data = request.get_json(force=True, silent=False)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Le corps de la requête doit être un JSON objet."}), 400

        input_df = pd.DataFrame([input_data])
        input_df = ensure_features(input_df)

        # Tentative de cast en numérique là où c'est possible
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                pass  # garde tel quel (categorical/text si le pipeline gère)

        probability = predict_proba_safely(model, input_df)
        decision = bool(probability >= 0.5)

        return jsonify({
            "default_probability": probability,
            "decision": decision
        })
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": f"Erreur lors de la prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    # Définis l’hôte/port via variables d’env si besoin
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "80"))
    app.run(debug=True, host=host, port=port)
