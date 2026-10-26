"""
data_loader.py
--------------
Data loading, normalization, target calculation, and validation for the Indian IPO Listing-Gain Predictor.

Supports real-world market datasets tracking:
- Company name, listing date, symbol, board
- Issue price (INR), Grey Market Premium (GMP INR)
- GMP-implied gain percentage
- Target actual listing gain percentage, positive listing indicator, and verdict
"""

from typing import List, Optional
import os
import re
import numpy as np
import pandas as pd

SECTOR_CANONICAL_MAP = {
    "technology": "Technology",
    "software": "Technology",
    "infotech": "Technology",
    "tech": "Technology",
    "digital": "Technology",
    "srit": "Technology",
    "acevector": "Technology",
    "financial services": "Financial Services",
    "finance": "Financial Services",
    "banking": "Financial Services",
    "payment": "Financial Services",
    "investor": "Financial Services",
    "reconstruction": "Financial Services",
    "moneyview": "Financial Services",
    "consumer goods": "Consumer Goods",
    "consumer": "Consumer Goods",
    "jewel": "Consumer Goods",
    "gems": "Consumer Goods",
    "style": "Consumer Goods",
    "rentomojo": "Consumer Goods",
    "chemicals": "Chemicals",
    "specialty chemicals": "Chemicals",
    "automobile": "Automobile",
    "motor": "Automobile",
    "infrastructure": "Infrastructure",
    "construction": "Infrastructure",
    "developers": "Infrastructure",
    "granito": "Infrastructure",
    "campuses": "Infrastructure",
    "manufacturing": "Manufacturing",
    "steel": "Manufacturing",
    "electrical": "Manufacturing",
    "engineering": "Manufacturing",
    "cables": "Manufacturing",
    "power": "Manufacturing",
    "industries": "Manufacturing",
    "retail": "Retail",
    "healthcare": "Healthcare",
    "pharma": "Healthcare",
}


def infer_sector_from_name(company_name: str) -> str:
    """Infer industry sector from company title and keywords."""
    name_l = str(company_name).lower()
    for keyword, sector in SECTOR_CANONICAL_MAP.items():
        if keyword in name_l:
            return sector
    return "Manufacturing" if any(w in name_l for w in ["ltd", "limited", "india", "indus"]) else "Other"


def compute_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic debut targets: listing_gain_pct, positive_listing, and verdict."""
    df = df.copy()

    # If listing_gain_pct is already present from actual_listing_gain_pct
    if "listing_gain_pct" not in df.columns and "actual_listing_gain_pct" in df.columns:
        df["listing_gain_pct"] = df["actual_listing_gain_pct"].astype(float)
    elif "listing_gain_pct" not in df.columns:
        issue_price = df["issue_price"].replace(0, np.nan)
        df["listing_gain_pct"] = np.round(((df["listing_price"] - issue_price) / issue_price) * 100.0, 2)

    df["listing_gain_pct"] = df["listing_gain_pct"].round(2)

    # If listing_price is missing, calculate from issue_price and listing_gain_pct
    if "listing_price" not in df.columns or df["listing_price"].isna().any():
        df["listing_price"] = np.round(df["issue_price"] * (1.0 + df["listing_gain_pct"] / 100.0), 2)

    # Binary positive debut (1 = positive gain, 0 = zero or discount)
    df["positive_listing"] = (df["listing_gain_pct"] > 0.0).astype(int)

    # 3-class Investment Verdict: Avoid (<0%), Neutral (0-15%), Apply (>15%)
    conditions = [
        df["listing_gain_pct"] < 0.0,
        (df["listing_gain_pct"] >= 0.0) & (df["listing_gain_pct"] <= 15.0),
        df["listing_gain_pct"] > 15.0,
    ]
    choices = ["Avoid", "Neutral", "Apply"]
    df["verdict"] = np.select(conditions, choices, default="Neutral")
    return df


def load_and_clean_ipo_data(
    raw_path: str = os.path.join("data", "raw", "ipo_data.csv"),
    processed_path: str = os.path.join("data", "processed", "ipo_clean.csv"),
) -> pd.DataFrame:
    """Load raw dataset, standardize schema, coerce data types, sort chronologically,

    and export cleaned dataset.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    df = raw_df.copy()

    # Column mapping for real market dataset
    rename_dict = {
        "company": "ipo_name",
        "issue_price_inr": "issue_price",
        "gmp_inr": "gmp_last_day",
        "actual_listing_gain_pct": "listing_gain_pct",
    }
    for old_col, new_col in rename_dict.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Handle Sector
    if "sector" not in df.columns:
        df["sector"] = df["ipo_name"].apply(infer_sector_from_name)
    else:
        df["sector"] = df["sector"].fillna(df["ipo_name"].apply(infer_sector_from_name))

    # Parse and sort by listing_date
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df = df.dropna(subset=["listing_date"]).sort_values("listing_date").reset_index(drop=True)

    # Standardize numerics
    df["issue_price"] = pd.to_numeric(df["issue_price"], errors="coerce").astype(float)
    df["gmp_last_day"] = pd.to_numeric(df["gmp_last_day"], errors="coerce").fillna(0.0).astype(float)

    if "gmp_implied_gain_pct" in df.columns:
        df["gmp_implied_gain_pct"] = pd.to_numeric(df["gmp_implied_gain_pct"], errors="coerce")
    else:
        df["gmp_implied_gain_pct"] = np.round((df["gmp_last_day"] / df["issue_price"]) * 100.0, 2)

    df["price_band_upper"] = df["issue_price"]

    # Target calculation
    df = compute_target_variables(df)

    # Deduplicate
    df = df.drop_duplicates(subset=["ipo_name", "listing_date"]).reset_index(drop=True)

    # Save processed clean CSV
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[SUCCESS] Cleaned {len(df)} records -> {processed_path}")
    return df


if __name__ == "__main__":
    cleaned = load_and_clean_ipo_data()
    print("\nCleaned Sample:")
    print(cleaned[["listing_date", "ipo_name", "sector", "issue_price", "gmp_last_day", "listing_gain_pct", "verdict"]].head(5))
