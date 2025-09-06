import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def main(in_dir: str, out_path: str):
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(in_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(in_dir / "y_train.csv").squeeze()

    # Grille simple (tu peux l'élargir si tu veux)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train.select_dtypes(include=["number"]), y_train)

    joblib.dump(grid.best_params_, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/processed")
    parser.add_argument("--out_path", type=str, default="models/best_params.pkl")
    args = parser.parse_args()
    main(args.in_dir, args.out_path)
