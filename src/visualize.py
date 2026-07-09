import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

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
    os.makedirs("outputs/figures", exist_ok=True)
    df = pd.read_csv("data/processed/network_usage_scored.csv")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="anomaly_score", hue="anomaly_pred", bins=30, kde=True, palette="Set1")
    plt.title("Distribution du score d'anomalie")
    plt.xlabel("Anomaly score (plus élevé = plus anormal)")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.savefig("outputs/figures/01_score_distribution.png", dpi=150)
    plt.close()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="nb_connexions_jour", y="volume_data_mo", hue="anomaly_pred", palette={0: "blue", 1: "red"}, alpha=0.7)
    plt.title("Connexions/jour vs Volume data")
    plt.xlabel("Nb connexions / jour")
    plt.ylabel("Volume data (Mo)")
    plt.legend(title="Anomalie prédite", labels=["Normal", "Anomalie"])
    plt.tight_layout()
    plt.savefig("outputs/figures/02_connexions_volume.png", dpi=150)
    plt.close()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(df[FEATURES])
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="pca1", y="pca2", hue="anomaly_pred", palette={0: "green", 1: "red"}, alpha=0.7)
    plt.title("Projection PCA des comportements")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Anomalie prédite", labels=["Normal", "Anomalie"])
    plt.tight_layout()
    plt.savefig("outputs/figures/03_pca_projection.png", dpi=150)
    plt.close()
    print("✅ Visualisations générées dans outputs/figures/")

if __name__ == "__main__":
    main()
