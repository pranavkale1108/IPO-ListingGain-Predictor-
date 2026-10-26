"""
predict.py
----------
Inference engine for the IPO Listing-Gain Predictor.

Loads trained and serialized models from disk:
- Preprocessor Pipeline (log1p + EM imputer + Scaler + OneHotEncoder)
- HMM Market Mood Engine
- K-Means Cluster Archetype Classifier
- Gradient Boosting Regressor (Expected listing gain %)
- Gradient Boosting Classifier (Probability of positive listing)
- Support Vector Machine (Investment verdict: Apply / Neutral / Avoid)
- k-NN Similarity Finder (Top 5 similar historical peer IPOs)

Enforces strict zero-leakage causal evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import IPOFeaturePipeline, ALL_NUMERIC_FEATURES, SUBSCRIPTION_COLS
from src.similar_ipos import IPOSimilarityFinder

# Global artifact cache to ensure fast Streamlit re-renders
_CACHE: Dict[str, Any] = {}


def load_artifacts(models_dir: str = "models", data_dir: str = os.path.join("data", "processed")) -> Dict[str, Any]:
    """Load all pre-trained pipeline models and reference datasets into memory cache.

    Parameters
    ----------
    models_dir : str, default='models'
        Path to model serialization directory.
    data_dir : str, default='data/processed'
        Path to processed datasets.

    Returns
    -------
    Dict[str, Any]
        Dictionary holding preprocessor, predictors, cluster models, and similarity engine.
    """
    global _CACHE
    if _CACHE:
        return _CACHE

    # Fallback paths if executed from subdirectory
    if not os.path.exists(models_dir):
        models_dir = os.path.join("..", "models")
    if not os.path.exists(data_dir):
        data_dir = os.path.join("..", "data", "processed")

    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    gbr_model = joblib.load(os.path.join(models_dir, "gradient_boosting_regressor.joblib"))
    gbc_model = joblib.load(os.path.join(models_dir, "gradient_boosting_classifier.joblib"))
    svm_model = joblib.load(os.path.join(models_dir, "svm_verdict.joblib"))
    kmeans_artifact = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    sim_finder = joblib.load(os.path.join(models_dir, "knn_similarity.joblib"))

    # Market mood reference table
    mood_path = os.path.join(data_dir, "monthly_market_mood.csv")
    mood_df = pd.read_csv(mood_path) if os.path.exists(mood_path) else pd.DataFrame()

    _CACHE = {
        "preprocessor": preprocessor,
        "gbr_model": gbr_model,
        "gbc_model": gbc_model,
        "svm_model": svm_model,
        "kmeans_model": kmeans_artifact["model"],
        "kmeans_feature_cols": kmeans_artifact["feature_cols"],
        "cluster_names": kmeans_artifact["cluster_names"],
        "sim_finder": sim_finder,
        "mood_df": mood_df,
    }
    return _CACHE


def get_market_mood_for_date(listing_date: Optional[str] = None, data_dir: str = os.path.join("data", "processed")) -> str:
    """Retrieve the causal prior-month market mood decoded by the discrete HMM.

    Parameters
    ----------
    listing_date : Optional[str]
        Date string (YYYY-MM-DD) of the upcoming or queried IPO.
    data_dir : str
        Directory containing monthly_market_mood.csv.

    Returns
    -------
    str
        'Cold', 'Warm', or 'Hot'.
    """
    cache = load_artifacts()
    mood_df = cache["mood_df"]

    if mood_df.empty:
        return "Warm"

    if listing_date is not None:
        try:
            target_dt = pd.to_datetime(listing_date)
            # Prior month period strictly before listing
            prior_period = (target_dt.to_period("M") - 1).strftime("%Y-%m")
            match = mood_df[mood_df["month"] == prior_period]
            if not match.empty:
                return str(match["market_mood"].iloc[0])
        except Exception:
            pass

    # Default to latest decoded market mood in historical database
    return str(mood_df.iloc[-1]["market_mood"])


def predict_ipo(input_dict: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """Generate comprehensive listing-day debut predictions for an IPO.

    Parameters
    ----------
    input_dict : Dict[str, Any]
        Dictionary of raw input features (issue size, subscription, GMP, sector, etc.).
    models_dir : str
        Directory containing model artifacts.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'expected_gain_pct': float
        - 'positive_listing_prob': float (0.0 to 1.0)
        - 'verdict': str ('Apply', 'Neutral', 'Avoid')
        - 'similar_ipos': pd.DataFrame (top 5 peers)
        - 'market_mood': str ('Cold', 'Warm', 'Hot')
        - 'cluster_id': int
        - 'cluster_name': str
        - 'raw_inputs': dict
    """
    cache = load_artifacts(models_dir=models_dir)
    prep: IPOFeaturePipeline = cache["preprocessor"]
    gbr = cache["gbr_model"]
    gbc = cache["gbc_model"]
    svm = cache["svm_model"]
    km_model = cache["kmeans_model"]
    km_feature_cols = cache["kmeans_feature_cols"]
    cluster_names = cache["cluster_names"]
    sim_finder: IPOSimilarityFinder = cache["sim_finder"]

    # 1. Fill sensible defaults for numeric inputs if missing
    sanitized: Dict[str, Any] = {
        "ipo_name": input_dict.get("ipo_name", "Upcoming Enterprise Debut Ltd"),
        "listing_date": input_dict.get("listing_date", "2024-12-15"),
        "sector": input_dict.get("sector", "Technology"),
        "issue_size_cr": float(input_dict.get("issue_size_cr", 500.0)),
        "price_band_upper": float(input_dict.get("price_band_upper", 450.0)),
        "issue_price": float(input_dict.get("issue_price", 450.0)),
        "qib_sub": float(input_dict.get("qib_sub", 10.0)),
        "nii_sub": float(input_dict.get("nii_sub", 10.0)),
        "retail_sub": float(input_dict.get("retail_sub", 5.0)),
        "total_sub": float(input_dict.get("total_sub", 8.0)),
        "gmp_last_day": float(input_dict.get("gmp_last_day", 50.0)),
        "fresh_issue_pct": float(input_dict.get("fresh_issue_pct", 70.0)),
        "ofs_pct": float(input_dict.get("ofs_pct", 30.0)),
        "revenue_growth_pct": float(input_dict.get("revenue_growth_pct", 20.0)),
        "profit_growth_pct": float(input_dict.get("profit_growth_pct", 15.0)),
        "pe_vs_peers": float(input_dict.get("pe_vs_peers", 0.95)),
        "nifty_7d_trend_pct": float(input_dict.get("nifty_7d_trend_pct", 1.0)),
    }

    # 2. Determine HMM Market Mood
    mood = input_dict.get("market_mood")
    if mood not in ["Cold", "Warm", "Hot"]:
        mood = get_market_mood_for_date(sanitized["listing_date"])

    # 3. Transform through fitted IPOFeaturePipeline
    df_raw = pd.DataFrame([sanitized])
    df_prep = prep.transform(df_raw)

    # 4. Attach One-Hot Mood Features
    for m in ["Cold", "Warm", "Hot"]:
        df_prep[f"mood_{m}"] = 1.0 if mood == m else 0.0

    # 5. Out-of-Sample Cluster Assignment (K-Means)
    # The first 27 columns represent the input space evaluated by KMeans
    feature_27 = km_feature_cols[:27]
    vec_27 = df_prep[feature_27].to_numpy(dtype=np.float64)
    centers_27 = km_model.cluster_centers_[:, :27]
    cluster_id = int(np.argmin(np.linalg.norm(vec_27 - centers_27, axis=1)))
    cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    # Attach One-Hot Cluster Features
    for cid in range(3):
        df_prep[f"cluster_{cid}"] = 1.0 if cluster_id == cid else 0.0

    # 6. Align with the 30-feature vector expected by final models
    feature_30 = getattr(gbr, "feature_names_in_", None)
    if feature_30 is None:
        feature_30 = [c for c in df_prep.columns if c in km_feature_cols[:27] or c.startswith("cluster_")]
    X_30 = df_prep[feature_30]

    # 7. Model Inferences
    # Expected gain % (GradientBoostingRegressor)
    expected_gain = float(gbr.predict(X_30)[0])

    # Probability of positive listing (GradientBoostingClassifier)
    pos_prob = float(gbc.predict_proba(X_30)[0, 1])

    # Investment Verdict (Support Vector Machine)
    verdict = str(svm.predict(X_30)[0])

    # Peer similarity search (k-NN)
    similar_ipos_df = sim_finder.find_similar_ipos(query=X_30.values[0], top_n=5)

    return {
        "expected_gain_pct": round(expected_gain, 2),
        "positive_listing_prob": round(pos_prob, 4),
        "verdict": verdict,
        "similar_ipos": similar_ipos_df,
        "market_mood": mood,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "raw_inputs": sanitized,
    }


if __name__ == "__main__":
    print("[TEST] Running predict.py standalone test...")
    sample_input = {
        "ipo_name": "Zenith AI Technologies Ltd",
        "listing_date": "2024-11-20",
        "sector": "Technology",
        "issue_size_cr": 850.0,
        "price_band_upper": 540.0,
        "issue_price": 540.0,
        "qib_sub": 45.2,
        "nii_sub": 28.5,
        "retail_sub": 12.0,
        "total_sub": 32.4,
        "gmp_last_day": 125.0,
        "fresh_issue_pct": 80.0,
        "ofs_pct": 20.0,
        "revenue_growth_pct": 42.0,
        "profit_growth_pct": 35.0,
        "pe_vs_peers": 0.92,
        "nifty_7d_trend_pct": 1.5,
    }
    result = predict_ipo(sample_input)
    print("\n--- Prediction Results ---")
    print(f"IPO Name               : {result['raw_inputs']['ipo_name']}")
    print(f"Decoded Market Mood    : {result['market_mood']}")
    print(f"Archetype Cluster      : {result['cluster_id']} ({result['cluster_name']})")
    print(f"Expected Listing Gain  : {result['expected_gain_pct']}%")
    print(f"Positive Debut Prob    : {result['positive_listing_prob'] * 100:.1f}%")
    print(f"SVM Investment Verdict : {result['verdict']}")
    print("\nTop 5 Most Similar Historical Peers:")
    print(result["similar_ipos"].to_string(index=False))
    print("\n[TEST] predict.py successfully executed.")
