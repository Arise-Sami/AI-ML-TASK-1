"""
Project 1: Student Performance Predictor
Task 1: Load and Explore the Data

Load a dataset of study hours and exam scores, then inspect structure,
missing values, and basic statistics before building a model.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "student_scores.csv"


def load_data() -> pd.DataFrame:
    """Load student performance data from CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Ensure student_scores.csv is in the data folder."
        )
    return pd.read_csv(DATA_FILE)


def explore_data(df: pd.DataFrame) -> None:
    """Print an overview of the dataset."""
    print("=" * 60)
    print("STUDENT PERFORMANCE DATASET")
    print("=" * 60)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print()

    print("First 5 rows:")
    print(df.head())
    print()

    print("Data types:")
    print(df.dtypes)
    print()

    print("Missing values per column:")
    print(df.isna().sum())
    print()

    print("Data statistics:")
    print(df.describe())


def main() -> None:
    df = load_data()
    explore_data(df)


if __name__ == "__main__":
    main()
