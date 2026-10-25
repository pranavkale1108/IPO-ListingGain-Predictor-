"""
preprocessor.py
---------------
Preprocessing and feature engineering pipeline for the IPO Listing-Gain Predictor.

Strict Project Rules Enforced:
- All scalers, imputers, and encoders are fitted strictly on the TRAINING set.
- Test set transformations are strictly out-of-sample (zero data leakage).
- Handles:
  1. Log-transformation on heavily-skewed subscription multiples.
  2. Expectation-Maximization (EM) imputation for missing valuation/growth features.
  3. One-hot encoding for categorical sector names.
  4. Standard scaling (mean=0, variance=1) for numeric features.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.em_imputer import EMGaussianImputer

# Metadata columns (identifiers & dates)
METADATA_COLS: List[str] = ["ipo_name", "listing_date"]

# Target columns (debut outcomes - NEVER used as input features)
TARGET_COLS: List[str] = [
    "listing_price",
    "listing_gain_pct",
    "positive_listing",
    "verdict",
]

# Subscription columns with heavy right skew -> log1p transformed
SUBSCRIPTION_COLS: List[str] = ["qib_sub", "nii_sub", "retail_sub", "total_sub"]

# Other numerical predictive features
BASE_NUMERIC_COLS: List[str] = [
    "issue_size_cr",
    "price_band_upper",
    "issue_price",
    "gmp_last_day",
    "fresh_issue_pct",
    "ofs_pct",
    "revenue_growth_pct",
    "profit_growth_pct",
    "pe_vs_peers",
    "nifty_7d_trend_pct",
]

# All numeric features subject to EM imputation
ALL_NUMERIC_FEATURES: List[str] = SUBSCRIPTION_COLS + BASE_NUMERIC_COLS

CATEGORICAL_COLS: List[str] = ["sector"]


class IPOFeaturePipeline:
    """Full Feature Engineering and Preprocessing Pipeline.

    Fitted strictly on training data to prevent any data leakage.
    """

    def __init__(self, em_max_iter: int = 100, em_tol: float = 1e-4, em_ridge: float = 1e-4):
        self.em_imputer = EMGaussianImputer(max_iter=em_max_iter, tol=em_tol, ridge=em_ridge)
        self.sector_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()

        self.numeric_feature_names: List[str] = []
        self.encoded_sector_names: List[str] = []
        self.final_feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "IPOFeaturePipeline":
        """Fit EM imputer, OneHotEncoder, and StandardScaler on training data only.

        Parameters
        ----------
        df_train : pd.DataFrame
            Chronological training dataset.

        Returns
        -------
        self : IPOFeaturePipeline
        """
        df = df_train.copy()

        # 1. Log-transform subscription features before imputation to improve Gaussianity
        X_num = df[ALL_NUMERIC_FEATURES].copy()
        for col in SUBSCRIPTION_COLS:
            X_num[col] = np.log1p(np.maximum(X_num[col], 0.0))

        # 2. Fit EM Imputer on numeric features
        self.em_imputer.fit(X_num)
        X_num_imputed = self.em_imputer.transform(X_num)

        # 3. Fit OneHotEncoder on sector
        sector_raw = df[CATEGORICAL_COLS].fillna("Other")
        self.sector_encoder.fit(sector_raw)
        self.encoded_sector_names = list(self.sector_encoder.get_feature_names_out(CATEGORICAL_COLS))

        # 4. Fit StandardScaler on imputed numeric features
        self.scaler.fit(X_num_imputed)
        self.numeric_feature_names = ALL_NUMERIC_FEATURES

        self.final_feature_names = self.numeric_feature_names + self.encoded_sector_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform any dataset (train or test) using the fitted parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to transform.

        Returns
        -------
        pd.DataFrame
            Clean, scaled, and one-hot encoded feature matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform().")

        df_copy = df.copy()

        # 1. Log-transform subscription features
        X_num = df_copy[ALL_NUMERIC_FEATURES].copy()
        for col in SUBSCRIPTION_COLS:
            X_num[col] = np.log1p(np.maximum(X_num[col], 0.0))

        # 2. EM Imputation using fitted mu_ and sigma_
        X_num_imputed = self.em_imputer.transform(X_num)

        # 3. Standard scaling
        X_num_scaled = self.scaler.transform(X_num_imputed)
        df_num_scaled = pd.DataFrame(X_num_scaled, columns=self.numeric_feature_names, index=df_copy.index)

        # 4. One-hot encode sector
        sector_raw = df_copy[CATEGORICAL_COLS].fillna("Other")
        sector_encoded = self.sector_encoder.transform(sector_raw)
        df_sector_encoded = pd.DataFrame(sector_encoded, columns=self.encoded_sector_names, index=df_copy.index)

        # 5. Concatenate processed features
        df_transformed = pd.concat([df_num_scaled, df_sector_encoded], axis=1)

        # Attach metadata if present
        for col in METADATA_COLS:
            if col in df_copy.columns:
                df_transformed[col] = df_copy[col].values

        # Attach targets if present
        for col in TARGET_COLS:
            if col in df_copy.columns:
                df_transformed[col] = df_copy[col].values

        return df_transformed

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit pipeline on training data and return transformed training set."""
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
