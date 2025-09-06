import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(in_dir: str, out_dir: str):
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(in_path / "X_train.csv")
    X_test  = pd.read_csv(in_path / "X_test.csv")

    # Séparer numériques et non-numériques
    num_cols = X_train.select_dtypes(include=["number"]).columns
    other_cols = X_train.columns.difference(num_cols)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

    # Sauvegarde
    X_train_scaled.to_csv(out_path / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(out_path / "X_test_scaled.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)
