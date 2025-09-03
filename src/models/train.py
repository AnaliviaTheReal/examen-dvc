import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def main(in_dir: str, out_dir: str):
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Charger les données
    X_train = pd.read_csv(in_path / "X_train_scaled.csv")
    y_train = pd.read_csv(in_path / "y_train.csv").squeeze()  # Série

    # Garder uniquement les colonnes numériques
    X_train = X_train.select_dtypes(include=["number"])

    # Modèle simple
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    joblib.dump(model, out_path / "model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
