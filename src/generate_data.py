import os
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_dataset(n_normal=450, n_anomalies=50):
    normal = pd.DataFrame({
        "nb_connexions_jour": np.random.normal(25, 8, n_normal).clip(1),
        "duree_moy_session_min": np.random.normal(18, 6, n_normal).clip(1),
        "volume_data_mo": np.random.normal(700, 220, n_normal).clip(50),
        "frequence_hors_horaires": np.random.beta(2, 10, n_normal),
        "nb_echecs_auth": np.random.poisson(1, n_normal),
        "diversite_destinations": np.random.normal(7, 2, n_normal).clip(1),
        "ecart_profil_moyen": np.random.normal(0.25, 0.10, n_normal).clip(0),
        "frequence_changement_session": np.random.normal(2, 1, n_normal).clip(0),
    })
    normal["true_label"] = 0
    anomalies = pd.DataFrame({
        "nb_connexions_jour": np.random.normal(90, 20, n_anomalies).clip(5),
        "duree_moy_session_min": np.random.normal(5, 2, n_anomalies).clip(0.5),
        "volume_data_mo": np.random.normal(2200, 500, n_anomalies).clip(300),
        "frequence_hors_horaires": np.random.uniform(0.6, 1.0, n_anomalies),
        "nb_echecs_auth": np.random.poisson(8, n_anomalies),
        "diversite_destinations": np.random.normal(20, 5, n_anomalies).clip(3),
        "ecart_profil_moyen": np.random.normal(0.9, 0.15, n_anomalies).clip(0.2),
        "frequence_changement_session": np.random.normal(10, 3, n_anomalies).clip(1),
    })
    anomalies["true_label"] = 1
    df = pd.concat([normal, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset()
    output_path = "data/raw/network_usage_simulated.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset simulé généré : {output_path}")
    print(df.head())
