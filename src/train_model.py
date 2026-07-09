import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = [
    "nb_connexions_jour",
    "duree_moy_session_min",
    "volume_data_mo",
    "frequence_hors_horaires",
    "nb_echecs_auth",
    "diversite_destinations",
    "ecart_profil_moyen",
    "frequence_changement_session",
]

def main():
    df = pd.read_csv("data/raw/network_usage_simulated.csv")
    X = df[FEATURES].copy()
    y_true = df["true_label"].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)
    model.fit(X_scaled)
    pred_raw = model.predict(X_scaled)
    df["anomaly_pred"] = (pred_raw == -1).astype(int)
    df["anomaly_score"] = -model.decision_function(X_scaled)
    print("\n=== Classification report (sur données simulées) ===")
    print(classification_report(y_true, df["anomaly_pred"], digits=3))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_true, df["anomaly_pred"]))
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/model", exist_ok=True)
    df.to_csv("data/processed/network_usage_scored.csv", index=False)
    joblib.dump(model, "outputs/model/isolation_forest.joblib")
    joblib.dump(scaler, "outputs/model/scaler.joblib")
    print("\n✅ Fichier enrichi : data/processed/network_usage_scored.csv")
    print("✅ Modèle sauvegardé : outputs/model/isolation_forest.joblib")
    print("✅ Scaler sauvegardé : outputs/model/scaler.joblib")

if __name__ == "__main__":
    main()
