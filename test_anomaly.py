import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("outputs/model/isolation_forest.joblib")
scaler = joblib.load("outputs/model/scaler.joblib")

# EXEMPLE 1: Utilisateur NORMAL
normal = pd.DataFrame({
    "nb_connexions_jour": [25],
    "duree_moy_session_min": [18],
    "volume_data_mo": [700],
    "frequence_hors_horaires": [0.1],
    "nb_echecs_auth": [0],
    "diversite_destinations": [6],
    "ecart_profil_moyen": [0.2],
    "frequence_changement_session": [1.5]
})

# EXEMPLE 2: Utilisateur SUSPECT
suspect = pd.DataFrame({
    "nb_connexions_jour": [100],
    "duree_moy_session_min": [2],
    "volume_data_mo": [3000],
    "frequence_hors_horaires": [0.9],
    "nb_echecs_auth": [15],
    "diversite_destinations": [30],
    "ecart_profil_moyen": [0.95],
    "frequence_changement_session": [20]
})

def test(data, label):
    X = scaler.transform(data)
    pred = model.predict(X)
    score = -model.decision_function(X)
    
    if pred[0] == -1:
        print(f"🔴 {label}: ANOMALIE DÉTECTÉE! Score: {score[0]:.3f}")
    else:
        print(f"🟢 {label}: Comportement NORMAL. Score: {score[0]:.3f}")

test(normal, "Utilisateur Normal")
test(suspect, "Utilisateur Suspect")
