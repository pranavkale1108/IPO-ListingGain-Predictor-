"""
data_loader.py
--------------
Data loading, validation, dtype coercion, sector standardization,
and target feature engineering for the Indian IPO Listing-Gain Predictor.

Strict Project Rules Enforced:
- Clean, documented code with docstrings.
- Ascending sort by listing_date for subsequent chronological train/test splitting.
- Fit all transformations only on training data (no scaling/imputation happens here;
  only raw cleaning and deterministic target variable calculations).
"""

from typing import List, Optional
import os
import re
import numpy as np
import pandas as pd

# Expected columns in raw IPO dataset
REQUIRED_RAW_COLUMNS: List[str] = [
    "ipo_name",
    "listing_date",
    "sector",
    "issue_size_cr",
    "price_band_upper",
    "issue_price",
    "listing_price",
    "qib_sub",
    "nii_sub",
    "retail_sub",
    "total_sub",
    "gmp_last_day",
    "fresh_issue_pct",
    "ofs_pct",
    "revenue_growth_pct",
    "profit_growth_pct",
    "pe_vs_peers",
    "nifty_7d_trend_pct",
]

NUMERIC_COLUMNS: List[str] = [
    "issue_size_cr",
    "price_band_upper",
    "issue_price",
    "listing_price",
    "qib_sub",
    "nii_sub",
    "retail_sub",
    "total_sub",
    "gmp_last_day",
    "fresh_issue_pct",
    "ofs_pct",
    "revenue_growth_pct",
    "profit_growth_pct",
    "pe_vs_peers",
    "nifty_7d_trend_pct",
]

# Canonical sector mapping dictionary to normalize variations
SECTOR_CANONICAL_MAP = {
    "it": "Technology",
    "tech": "Technology",
    "information technology": "Technology",
    "technology": "Technology",
    "software": "Technology",
    "fintech": "Financial Services",
    "finance": "Financial Services",
    "financial services": "Financial Services",
    "banking": "Financial Services",
    "nbfc": "Financial Services",
    "pharma": "Healthcare",
    "pharmaceuticals": "Healthcare",
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "hospitals": "Healthcare",
    "fmcg": "Consumer Goods",
    "consumer": "Consumer Goods",
    "consumer goods": "Consumer Goods",
    "consumer durables": "Consumer Goods",
    "manufacturing": "Manufacturing",
    "industrial": "Manufacturing",
    "capital goods": "Manufacturing",
    "auto": "Automobile",
    "automobile": "Automobile",
    "auto components": "Automobile",
    "infra": "Infrastructure",
    "infrastructure": "Infrastructure",
    "construction": "Infrastructure",
    "chemicals": "Chemicals",
    "specialty chemicals": "Chemicals",
    "energy": "Energy",
    "oil & gas": "Energy",
    "power": "Energy",
    "renewables": "Energy",
    "retail": "Retail",
    "e-commerce": "Retail",
}


def validate_raw_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns exist in the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataset is missing required column(s): {missing}\n"
            f"Expected schema: {REQUIRED_RAW_COLUMNS}"
        )


def clean_numeric_value(val):
    """Strip common formatting artifacts (commas, currency symbols, whitespace)

    and convert to float or NaN.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    # Remove commas, rupee signs, currency strings
    cleaned = re.sub(r"[₹$,\s]", "", val_str)
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def standardize_sector(sector_val: Optional[str]) -> str:
    """Standardize sector names into canonical industry groups.

    Parameters
    ----------
    sector_val : Optional[str]
        Raw sector name string.

    Returns
    -------
    str
        Canonical sector name or 'Other' if unrecognized or blank.
    """
    if pd.isna(sector_val):
        return "Other"
    raw_clean = str(sector_val).strip().lower()
    return SECTOR_CANONICAL_MAP.get(raw_clean, str(sector_val).strip().title())


def compute_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic target variables for regression, classification,

    and business verdicts.

    Derived Targets:
    - listing_gain_pct: ((listing_price - issue_price) / issue_price) * 100
    - positive_listing: 1 if listing_gain_pct > 0 else 0
    - verdict: Avoid if gain < 0, Neutral if 0 <= gain <= 15, Apply if gain > 15

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with issue_price and listing_price.

    Returns
    -------
    pd.DataFrame
        Dataframe with added target columns.
    """
    df = df.copy()

    # 1. Listing Gain Percentage
    # listing_gain_pct = (listing_price - issue_price) / issue_price * 100
    issue_price = df["issue_price"].replace(0, np.nan)
    df["listing_gain_pct"] = np.round(((df["listing_price"] - issue_price) / issue_price) * 100.0, 2)

    # 2. Binary classification target: Positive Listing (1 = positive gain, 0 = zero or negative)
    df["positive_listing"] = (df["listing_gain_pct"] > 0.0).astype(int)

    # 3. Decision Verdict: Avoid (<0%), Neutral (0-15%), Apply (>15%)
    conditions = [
        df["listing_gain_pct"] < 0.0,
        (df["listing_gain_pct"] >= 0.0) & (df["listing_gain_pct"] <= 15.0),
        df["listing_gain_pct"] > 15.0,
    ]
    choices = ["Avoid", "Neutral", "Apply"]
    df["verdict"] = np.select(conditions, choices, default="Neutral")

    return df


def clean_ipo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply parsing, type coercions, deduplication, sector normalization,

    and chronological sorting to the raw IPO dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw IPO dataset.

    Returns
    -------
    pd.DataFrame
        Fully sanitized and chronologically ordered DataFrame with target variables.
    """
    validate_raw_columns(df)
    df_clean = df.copy()

    # 1. Standardize string fields
    df_clean["ipo_name"] = df_clean["ipo_name"].astype(str).str.strip()
    df_clean["sector"] = df_clean["sector"].apply(standardize_sector)

    # 2. Parse listing_date
    df_clean["listing_date"] = pd.to_datetime(df_clean["listing_date"], errors="coerce")
    invalid_dates = df_clean["listing_date"].isna().sum()
    if invalid_dates > 0:
        print(f"[WARNING] Dropped {invalid_dates} record(s) due to unparseable listing_date.")
        df_clean = df_clean.dropna(subset=["listing_date"])

    # 3. Clean numeric fields
    for col in NUMERIC_COLUMNS:
        df_clean[col] = df_clean[col].apply(clean_numeric_value)

    # Validate issue_price and listing_price sanity
    invalid_prices = (df_clean["issue_price"] <= 0) | (df_clean["listing_price"] <= 0)
    if invalid_prices.sum() > 0:
        print(f"[WARNING] Dropped {invalid_prices.sum()} record(s) with non-positive issue or listing price.")
        df_clean = df_clean[~invalid_prices]

    # 4. Drop duplicates (preferring unique IPO name)
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["ipo_name"], keep="last")
    dropped_dupes = initial_count - len(df_clean)
    if dropped_dupes > 0:
        print(f"[INFO] Removed {dropped_dupes} duplicate IPO record(s).")

    # 5. Chronological sorting by listing_date (MANDATORY for time-based split rule)
    df_clean = df_clean.sort_values(by="listing_date", ascending=True).reset_index(drop=True)

    # 6. Compute Target Variables
    df_clean = compute_target_variables(df_clean)

    return df_clean


def load_and_clean_ipo_data(
    raw_filepath: str = os.path.join("data", "raw", "ipo_data.csv"),
    processed_filepath: str = os.path.join("data", "processed", "ipo_clean.csv"),
) -> pd.DataFrame:
    """Load raw IPO data, execute full cleaning and target creation,

    and save the cleaned result to processed storage.

    Parameters
    ----------
    raw_filepath : str, default='data/raw/ipo_data.csv'
        Path to raw CSV file.
    processed_filepath : str, default='data/processed/ipo_clean.csv'
        Destination path for cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned, chronological DataFrame ready for imputation, feature modeling,
        and time-based train/test splitting.
    """
    if not os.path.exists(raw_filepath):
        # Check if synthetic dataset exists as a helpful fallback
        fallback_synthetic = os.path.join("data", "raw", "synthetic_ipo_data_WARNING_SYNTHETIC.csv")
        if os.path.exists(fallback_synthetic):
            print(f"[INFO] Primary '{raw_filepath}' not found. Loading synthetic fallback: '{fallback_synthetic}'")
            raw_filepath = fallback_synthetic
        else:
            raise FileNotFoundError(
                f"Raw IPO data file not found at: '{raw_filepath}'.\n"
                f"Please place your real IPO dataset at '{raw_filepath}' or run "
                f"'python scripts/make_sample_data.py' to generate synthetic test data."
            )

    print(f"[LOAD] Reading raw IPO data from: {raw_filepath}")
    df_raw = pd.read_csv(raw_filepath)
    print(f"[LOAD] Loaded {len(df_raw)} raw rows across {len(df_raw.columns)} columns.")

    # Clean and create targets
    df_clean = clean_ipo_dataframe(df_raw)

    # Save to processed directory
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df_clean.to_csv(processed_filepath, index=False)
    print(f"[SAVE] Cleaned dataset ({len(df_clean)} rows) saved to: {processed_filepath}")

    # Summary statistics
    verdict_counts = df_clean["verdict"].value_counts().to_dict()
    pos_count = (df_clean["positive_listing"] == 1).sum()
    print(f"[SUMMARY] Target Distribution:")
    print(f"   - Positive Listings: {pos_count}/{len(df_clean)} ({pos_count / len(df_clean):.1%})")
    print(f"   - Verdict Breakdown: {verdict_counts}")
    print(f"   - Date Span: {df_clean['listing_date'].min().date()} to {df_clean['listing_date'].max().date()}")

    return df_clean


if __name__ == "__main__":
    load_and_clean_ipo_data()
