import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def main(in_dir: str, model_dir: str, out_dir: str):
    in_path = Path(in_dir)
    model_path = Path(model_dir) / "model.pkl"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Charger les données
    X_test = pd.read_csv(in_path / "X_test_scaled.csv")
    y_test = pd.read_csv(in_path / "y_test.csv").squeeze()

    # Charger le modèle
    model = joblib.load(model_path)

    # Garder uniquement les colonnes numériques
    X_test = X_test.select_dtypes(include=["number"])

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    # Sauvegarde en JSON
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--out_dir", type=str, default="metrics")
    args = parser.parse_args()
    main(args.in_dir, args.model_dir, args.out_dir)
