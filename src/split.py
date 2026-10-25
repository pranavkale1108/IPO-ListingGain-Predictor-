"""
split.py
--------
Strict chronological (time-based) train/test split module for IPO Listing-Gain Predictor.

Core Project Rules Enforced:
- Use a TIME-BASED train/test split (oldest 75% train, newest 25% test). Never a random split.
- Preserves temporal order so that models are trained on past market data and evaluated
  on future market conditions, mirroring real-world deployment.
- Proves no temporal overlap between train and test sets.
"""

from typing import Tuple, Optional
import os
import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    date_col: str = "listing_date",
    save_splits: bool = True,
    output_dir: str = os.path.join("data", "processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned IPO dataset containing `date_col`.
    train_frac : float, default=0.75
        Fraction of oldest samples to allocate to the training set (0 < train_frac < 1).
    date_col : str, default='listing_date'
        Name of the date column used for temporal ordering.
    save_splits : bool, default=True
        Whether to save train.csv and test.csv into `output_dir`.
    output_dir : str, default='data/processed'
        Destination folder for split CSV files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) partitioned chronologically without data leakage.

    Raises
    ------
    ValueError
        If train_frac is not between 0 and 1, or date_col is missing.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be strictly between 0 and 1, got {train_frac}")

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe columns: {list(df.columns)}")

    # Ensure sorting by date column ascending
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
    df_sorted = df_sorted.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    n_total = len(df_sorted)
    n_train = int(n_total * train_frac)

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train:].copy().reset_index(drop=True)

    # Date range calculations
    train_min = train_df[date_col].min().strftime("%Y-%m-%d")
    train_max = train_df[date_col].max().strftime("%Y-%m-%d")
    test_min = test_df[date_col].min().strftime("%Y-%m-%d")
    test_max = test_df[date_col].max().strftime("%Y-%m-%d")

    # Temporal verification
    train_max_ts = train_df[date_col].max()
    test_min_ts = test_df[date_col].min()
    no_overlap = train_max_ts <= test_min_ts

    print("=" * 70)
    print("CHRONOLOGICAL TIME-BASED TRAIN/TEST SPLIT VERIFICATION")
    print("=" * 70)
    print(f"Total IPO records  : {n_total}")
    print(f"Train split ({train_frac:.0%})   : {len(train_df)} records | Date Range: [{train_min}] to [{train_max}]")
    print(f"Test split ({1 - train_frac:.0%})    : {len(test_df)} records | Date Range: [{test_min}] to [{test_max}]")
    print(f"Temporal overlap   : {'NONE (Valid chronological boundary)' if no_overlap else 'WARNING: Overlap detected!'}")
    print(f"Boundary check     : Last Train Date ({train_max}) <= First Test Date ({test_min}) -> {no_overlap}")
    print("=" * 70)

    # Save to disk if requested
    if save_splits:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"[SAVE] Saved train set to: {train_path}")
        print(f"[SAVE] Saved test set to : {test_path}")

    return train_df, test_df


def load_and_split(
    clean_filepath: str = os.path.join("data", "processed", "ipo_clean.csv"),
    train_frac: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper function to load ipo_clean.csv and perform time_split."""
    if not os.path.exists(clean_filepath):
        # If clean file does not exist, run the data_loader first
        from src.data_loader import load_and_clean_ipo_data
        print(f"[INFO] '{clean_filepath}' not found. Triggering data_loader...")
        df_clean = load_and_clean_ipo_data()
    else:
        df_clean = pd.read_csv(clean_filepath)

    return time_split(df_clean, train_frac=train_frac)


if __name__ == "__main__":
    load_and_split()
