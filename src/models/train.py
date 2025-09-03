import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def main(in_dir: str, out_dir: str, best_params_path: str = "models/best_params.pkl"):
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(in_path / "X_train_scaled.csv").select_dtypes(include=["number"])
    y_train = pd.read_csv(in_path / "y_train.csv").squeeze()

    params = {"n_estimators": 100, "random_state": 42}
    bp = Path(best_params_path)
    if bp.exists():
        try:
            best = joblib.load(bp)
            if isinstance(best, dict):
                best = {k: v for k, v in best.items() if k != "random_state"}
                params.update(best)
        except Exception:
            pass

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, out_path / "model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--best_params", type=str, default="models/best_params.pkl")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.best_params)
