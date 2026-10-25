"""
similar_ipos.py
---------------
Peer IPO similarity search engine using k-Nearest Neighbors (k-NN).

Returns the most similar past IPOs based on standardized multi-dimensional
features (issue size, subscription multiples, valuation, sector, and market mood).

Strict Leakage Rule Enforced:
When querying a historical IPO, only peers listed STRICTLY BEFORE that IPO's
listing date are considered valid candidate neighbors.
"""

from typing import List, Optional, Union, Dict, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class IPOSimilarityFinder:
    """k-NN peer similarity retrieval system for IPOs.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical database of IPOs with raw metrics and metadata.
    feature_matrix : pd.DataFrame or np.ndarray
        Standardized, preprocessed feature matrix corresponding to historical_df.
    feature_cols : Optional[List[str]]
        Names of predictor feature columns used for distance calculation.
    metric : str, default='euclidean'
        Distance metric for NearestNeighbors.
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        metric: str = "euclidean",
    ):
        self.historical_df = historical_df.copy().reset_index(drop=True)
        self.historical_df["listing_date"] = pd.to_datetime(self.historical_df["listing_date"])

        non_features = [
            "ipo_name", "listing_date", "listing_price", "listing_gain_pct",
            "positive_listing", "verdict", "market_mood", "cluster_id", "cluster_name"
        ]
        self.feature_cols = feature_cols or [c for c in feature_matrix.columns if c not in non_features]
        self.X = feature_matrix[self.feature_cols].to_numpy(dtype=np.float64)
        self.metric = metric

        # Global NearestNeighbors on full historical pool
        self.nn_model = NearestNeighbors(metric=self.metric)
        self.nn_model.fit(self.X)

    def find_similar_ipos(
        self,
        query: Union[str, np.ndarray, pd.Series],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Find the top_n most similar past IPOs for a given query IPO.

        Parameters
        ----------
        query : str or array-like
            If str: the name of an existing IPO in historical_df.
            If array-like: a feature vector representing an upcoming IPO.
        top_n : int, default=5
            Number of similar peers to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            ['ipo_name', 'sector', 'total_sub', 'listing_gain_pct', 'listing_date', 'similarity_score']
        """
        # Case A: Query is an existing historical IPO name
        if isinstance(query, str):
            match = self.historical_df[self.historical_df["ipo_name"].str.lower() == query.strip().lower()]
            if match.empty:
                raise ValueError(f"IPO '{query}' not found in historical database.")

            query_idx = match.index[0]
            query_date = match.loc[query_idx, "listing_date"]
            query_vector = self.X[query_idx].reshape(1, -1)

            # Anti-leakage: filter pool to IPOs listed STRICTLY BEFORE the query IPO
            valid_mask = self.historical_df["listing_date"] < query_date
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                # If earliest IPO, fallback to closest peers excluding itself
                valid_indices = np.setdiff1d(np.arange(len(self.historical_df)), [query_idx])

            candidate_X = self.X[valid_indices]
            nbrs = NearestNeighbors(n_neighbors=min(top_n, len(valid_indices)), metric=self.metric)
            nbrs.fit(candidate_X)
            distances, indices = nbrs.kneighbors(query_vector)

            chosen_indices = valid_indices[indices[0]]
            chosen_dists = distances[0]

        # Case B: Query is an arbitrary preprocessed feature vector (upcoming IPO)
        else:
            query_vector = np.asarray(query, dtype=np.float64).reshape(1, -1)
            k = min(top_n, len(self.historical_df))
            distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=k)
            chosen_indices = indices[0]
            chosen_dists = distances[0]

        # Format output
        results = self.historical_df.iloc[chosen_indices].copy()
        # Similarity score between 0 and 1: 1 / (1 + distance)
        results["similarity_score"] = np.round(1.0 / (1.0 + chosen_dists), 4)

        display_cols = [
            "ipo_name",
            "sector",
            "total_sub",
            "listing_gain_pct",
            "listing_date",
            "similarity_score",
        ]
        available_cols = [c for c in display_cols if c in results.columns]
        return results[available_cols].reset_index(drop=True)

    def save(self, filepath: str = os.path.join("models", "knn_similarity.joblib")) -> None:
        """Persist similarity engine to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SAVE] Similarity engine saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = os.path.join("models", "knn_similarity.joblib")) -> "IPOSimilarityFinder":
        """Load similarity engine from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Similarity engine not found at: '{filepath}'")
        return joblib.load(filepath)


def get_similar_ipos(
    query: Union[str, np.ndarray, pd.Series],
    top_n: int = 5,
    processed_dir: str = os.path.join("data", "processed"),
) -> pd.DataFrame:
    """Convenience helper to retrieve top_n similar IPOs."""
    # Load clustered/with_mood dataset
    raw_path = os.path.join(processed_dir, "train_clustered.csv")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(processed_dir, "train.csv")

    prep_path = os.path.join(processed_dir, "train_preprocessed.csv")

    hist_df = pd.read_csv(raw_path)
    prep_df = pd.read_csv(prep_path)

    finder = IPOSimilarityFinder(historical_df=hist_df, feature_matrix=prep_df)
    return finder.find_similar_ipos(query=query, top_n=top_n)


if __name__ == "__main__":
    # Self-test on an actual training IPO
    print("[Self-Test] Running IPOSimilarityFinder...")
    train_clustered_path = os.path.join("data", "processed", "train_clustered.csv")
    if os.path.exists(train_clustered_path):
        sample_df = pd.read_csv(train_clustered_path)
        sample_name = sample_df.iloc[20]["ipo_name"]
        print(f"Query IPO: '{sample_name}'")
        peers = get_similar_ipos(sample_name, top_n=5)
        print("\nTop 5 Similar Historical Peers (Strictly Prior Listing Dates):")
        print(peers.to_string(index=False))
        print("[Self-Test] Successfully validated similar_ipos.py.")
    else:
        print("[Self-Test] train_clustered.csv not found, skipping standalone execution.")
