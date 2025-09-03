import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv: str, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Chargement
    df = pd.read_csv(input_csv)

    # La cible est la dernière colonne (silica_concentrate)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Sauvegarde
    X_train.to_csv(out_path / "X_train.csv", index=False)
    X_test.to_csv(out_path / "X_test.csv", index=False)
    y_train.to_csv(out_path / "y_train.csv", index=False)
    y_test.to_csv(out_path / "y_test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="data/raw/raw.csv")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    main(args.in_csv, args.out_dir, args.test_size, args.random_state)
