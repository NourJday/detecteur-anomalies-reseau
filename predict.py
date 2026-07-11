import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("outputs/model/isolation_forest.joblib")
scaler = joblib.load("outputs/model/scaler.joblib")

# Tes données
ma_data = pd.DataFrame({
    "nb_connexions_jour": [50],
    "duree_moy_session_min": [10],
    "volume_data_mo": [1500],
    "frequence_hors_horaires": [0.5],
    "nb_echecs_auth": [3],
    "diversite_destinations": [15],
    "ecart_profil_moyen": [0.6],
    "frequence_changement_session": [5]
})

# Prédiction
X = scaler.transform(ma_data)
pred = model.predict(X)
score = -model.decision_function(X)

if pred[0] == -1:
    print(f"🔴 ANOMALIE! Score: {score[0]:.3f}")
else:
    print(f"🟢 NORMAL. Score: {score[0]:.3f}")
