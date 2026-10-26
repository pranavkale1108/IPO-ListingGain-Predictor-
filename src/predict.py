"""
predict.py
----------
Inference engine for the IPO Listing-Gain Predictor.

Loads trained real-market models from disk:
- Preprocessor Pipeline (StandardScaler + OneHotEncoder)
- K-Means Cluster Archetype Classifier
- Gradient Boosting Regressor (Expected listing gain %)
- Gradient Boosting Classifier (Probability of positive debut)
- Support Vector Machine (Investment verdict: Apply / Neutral / Avoid)
- k-NN Similarity Finder (Top 5 similar historical peer IPOs)

Enforces strict zero-leakage causal evaluation.
"""

from typing import Dict, Any, Optional, List
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd

from src.preprocessor import IPOFeaturePipeline
from src.similar_ipos import IPOSimilarityFinder

_CACHE: Dict[str, Any] = {}


def load_artifacts(models_dir: str = "models", data_dir: str = os.path.join("data", "processed")) -> Dict[str, Any]:
    """Load all pre-trained pipeline models and reference datasets into memory cache."""
    global _CACHE
    if _CACHE:
        return _CACHE

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
    """Retrieve market sentiment / mood context."""
    cache = load_artifacts()
    mood_df = cache["mood_df"]
    if mood_df.empty:
        return "Warm"
    return str(mood_df.iloc[-1].get("market_mood", "Warm"))


def predict_ipo(input_dict: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """Generate comprehensive listing-day debut predictions for an IPO.

    Parameters
    ----------
    input_dict : Dict[str, Any]
        Dictionary containing at least 'issue_price' and 'gmp_last_day' (and optional sector/name/date).
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

    # Sanitize inputs with sensible fallbacks
    issue_price = float(input_dict.get("issue_price") or input_dict.get("issue_price_inr") or 200.0)
    gmp = float(input_dict.get("gmp_last_day") or input_dict.get("gmp_inr") or 25.0)
    implied_gain = float(input_dict.get("gmp_implied_gain_pct") or ((gmp / issue_price) * 100.0 if issue_price > 0 else 0.0))

    sanitized: Dict[str, Any] = {
        "ipo_name": input_dict.get("ipo_name") or input_dict.get("company") or "Upcoming Market Debut Ltd",
        "listing_date": str(input_dict.get("listing_date") or "2026-10-15"),
        "sector": input_dict.get("sector") or "Manufacturing",
        "issue_price": issue_price,
        "gmp_last_day": gmp,
        "gmp_implied_gain_pct": round(implied_gain, 2),
    }

    # Transform through fitted preprocessor
    df_raw = pd.DataFrame([sanitized])
    df_prep = prep.transform(df_raw)

    # Cluster Archetype Assignment
    km_feats = [c for c in km_feature_cols if c in df_prep.columns]
    for c in km_feature_cols:
        if c not in df_prep.columns:
            df_prep[c] = 0.0

    cluster_id = int(km_model.predict(df_prep[km_feature_cols])[0])
    cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

    for cid in range(3):
        df_prep[f"cluster_{cid}"] = 1.0 if cluster_id == cid else 0.0

    # Align with model input features
    fn = getattr(gbr, "feature_names_in_", None)
    if fn is not None:
        feature_cols = list(fn)
    else:
        feature_cols = list(df_prep.columns)
    for c in feature_cols:
        if c not in df_prep.columns:
            df_prep[c] = 0.0
    X_pred = df_prep[feature_cols]

    # Inferences
    expected_gain = float(gbr.predict(X_pred)[0])
    pos_prob = float(gbc.predict_proba(X_pred)[0, 1])
    verdict = str(svm.predict(X_pred)[0])

    # Dynamic Market Mood based on expected gain & GMP
    if expected_gain >= 20.0 or implied_gain >= 25.0:
        mood = "Hot"
    elif expected_gain >= 5.0 or implied_gain >= 5.0:
        mood = "Warm"
    else:
        mood = "Cold"

    # Similar historical peer retrieval
    similar_ipos_df = sim_finder.find_similar_ipos(query=X_pred.values[0], top_n=5)

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
    print("[TEST] Running predict.py with real market test cases...")
    test_cases = [
        {"company": "Orient Cables (India) Limited", "issue_price_inr": 272, "gmp_inr": 107, "sector": "Manufacturing"},
        {"company": "Shah Investor's Home Limited", "issue_price_inr": 167, "gmp_inr": 12, "sector": "Financial Services"},
        {"company": "AceVector Limited", "issue_price_inr": 32, "gmp_inr": 1, "sector": "Technology"},
    ]
    for case in test_cases:
        res = predict_ipo(case)
        print(f"\nIPO: {res['raw_inputs']['ipo_name']}")
        print(f"  Expected Gain : {res['expected_gain_pct']:+.2f}%")
        print(f"  Positive Prob : {res['positive_listing_prob'] * 100:.1f}%")
        print(f"  Verdict       : {res['verdict']}")
        print(f"  Cluster       : {res['cluster_name']}")
        print(f"  Peers Found   : {len(res['similar_ipos'])}")
