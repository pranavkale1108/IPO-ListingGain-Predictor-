"""
preprocessor.py
---------------
Preprocessing and feature engineering pipeline for the IPO Listing-Gain Predictor.

Strict Project Rules Enforced:
- All scalers and encoders are fitted strictly on the TRAINING set.
- Test set transformations are strictly out-of-sample (zero data leakage).
- Handles:
  1. Standard scaling for numeric indicators (issue price, GMP, implied gain, etc.).
  2. One-hot encoding for categorical sector names.
  3. Dynamic feature discovery for real-world or full-attribute datasets.
"""

from typing import List, Optional, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METADATA_COLS: List[str] = ["ipo_name", "listing_date", "company", "symbol", "board"]

TARGET_COLS: List[str] = [
    "listing_price",
    "listing_gain_pct",
    "positive_listing",
    "verdict",
    "actual_listing_gain_pct",
]

CATEGORICAL_COLS: List[str] = ["sector"]

PREFERRED_NUMERIC_COLS: List[str] = [
    "issue_price",
    "gmp_last_day",
    "gmp_implied_gain_pct",
    "qib_sub",
    "nii_sub",
    "retail_sub",
    "total_sub",
    "issue_size_cr",
    "price_band_upper",
    "fresh_issue_pct",
    "ofs_pct",
    "revenue_growth_pct",
    "profit_growth_pct",
    "pe_vs_peers",
    "nifty_7d_trend_pct",
]


class IPOFeaturePipeline:
    """Full Feature Engineering and Preprocessing Pipeline.

    Fitted strictly on training data to prevent any data leakage.
    """

    def __init__(self):
        self.sector_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_feature_names: List[str] = []
        self.encoded_sector_names: List[str] = []
        self.final_feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "IPOFeaturePipeline":
        """Fit OneHotEncoder and StandardScaler on training data only."""
        df = df_train.copy()

        # Compute gmp_implied_gain_pct if missing
        if "gmp_implied_gain_pct" not in df.columns and "gmp_last_day" in df.columns and "issue_price" in df.columns:
            df["gmp_implied_gain_pct"] = (df["gmp_last_day"] / df["issue_price"].replace(0, np.nan)) * 100.0

        # Identify numeric features present in df_train
        self.numeric_feature_names = [
            c for c in PREFERRED_NUMERIC_COLS
            if c in df.columns and not df[c].isna().all()
        ]
        if not self.numeric_feature_names:
            # Fallback to any numeric column not in metadata or targets
            self.numeric_feature_names = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in TARGET_COLS + METADATA_COLS
            ]

        X_num = df[self.numeric_feature_names].fillna(0.0).to_numpy(dtype=np.float64)
        self.scaler.fit(X_num)

        # Fit sector OneHotEncoder
        if "sector" in df.columns:
            sector_raw = df[CATEGORICAL_COLS].fillna("Other")
        else:
            sector_raw = pd.DataFrame({"sector": ["Other"] * len(df)})
        self.sector_encoder.fit(sector_raw)
        self.encoded_sector_names = list(self.sector_encoder.get_feature_names_out(CATEGORICAL_COLS))

        self.final_feature_names = self.numeric_feature_names + self.encoded_sector_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform any dataset (train, test, or live query) using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform().")

        df_copy = df.copy()

        # Compute gmp_implied_gain_pct if missing
        if "gmp_implied_gain_pct" not in df_copy.columns and "gmp_last_day" in df_copy.columns and "issue_price" in df_copy.columns:
            df_copy["gmp_implied_gain_pct"] = (df_copy["gmp_last_day"] / df_copy["issue_price"].replace(0, np.nan)) * 100.0

        # Fill missing numeric columns with zeros
        for col in self.numeric_feature_names:
            if col not in df_copy.columns:
                df_copy[col] = 0.0

        X_num = df_copy[self.numeric_feature_names].fillna(0.0).to_numpy(dtype=np.float64)
        X_num_scaled = self.scaler.transform(X_num)
        df_num_scaled = pd.DataFrame(X_num_scaled, columns=self.numeric_feature_names, index=df_copy.index)

        # Sector encoding
        if "sector" in df_copy.columns:
            sector_raw = df_copy[CATEGORICAL_COLS].fillna("Other")
        else:
            sector_raw = pd.DataFrame({"sector": ["Other"] * len(df_copy)})
        sector_encoded = self.sector_encoder.transform(sector_raw)
        df_sector_encoded = pd.DataFrame(sector_encoded, columns=self.encoded_sector_names, index=df_copy.index)

        df_transformed = pd.concat([df_num_scaled, df_sector_encoded], axis=1)

        # Attach metadata and targets if present
        for col in METADATA_COLS + TARGET_COLS:
            if col in df_copy.columns:
                df_transformed[col] = df_copy[col].values

        return df_transformed

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed training set."""
        return self.fit(df_train).transform(df_train)

    def save(self, filepath: str = os.path.join("models", "preprocessor.joblib")) -> None:
        """Persist fitted pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SAVE] Preprocessor pipeline saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = os.path.join("models", "preprocessor.joblib")) -> "IPOFeaturePipeline":
        """Load fitted pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fitted pipeline not found at '{filepath}'")
        return joblib.load(filepath)
