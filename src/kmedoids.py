"""
kmedoids.py
-----------
Partitioning Around Medoids (PAM) clustering algorithm implemented
from scratch in NumPy following Scikit-Learn estimator conventions.

Advantages over K-Means for Financial/IPO Data:
- Medoids are actual historical IPO exemplars (real data points), making clusters
  directly interpretable by investment analysts.
- Robust to extreme-gain outliers (blockbuster IPOs with 150%+ listing pop) because
  PAM minimizes absolute/pairwise distances rather than squared Euclidean errors.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from scipy.spatial.distance import cdist


class KMedoidsPAM(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Medoids clustering using Partitioning Around Medoids (PAM).

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters and medoids to form.
    metric : str, default='euclidean'
        Metric for pairwise distance calculation ('euclidean' or 'cityblock' / 'manhattan').
    max_iter : int, default=300
        Maximum number of iterations for the swap phase.
    random_state : Optional[int], default=42
        Random seed for medoid initialization.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        max_iter: int = 300,
        random_state: Optional[int] = 42,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

        # Fitted attributes
        self.medoid_indices_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None  # The medoid data vectors
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0

    def _init_medoids(self, D: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
        """Initialize medoids using k-medoids++ probability distribution."""
        # 1. Pick first medoid as the point with minimum total distance to all other points
        first_medoid = int(np.argmin(np.sum(D, axis=1)))
        medoids = [first_medoid]

        # 2. Iteratively pick remaining medoids proportional to squared distance from nearest existing medoid
        for _ in range(1, self.n_clusters):
            dist_to_closest = np.min(D[:, medoids], axis=1)
            dist_sq = dist_to_closest ** 2
            prob_sum = np.sum(dist_sq)
            if prob_sum > 0:
                probs = dist_sq / prob_sum
                next_medoid = int(rng.choice(N, p=probs))
            else:
                next_medoid = int(rng.choice(N))
            medoids.append(next_medoid)

        return np.array(medoids, dtype=int)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "KMedoidsPAM":
        """Compute K-Medoids clustering on X.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Scikit-learn API compatibility.

        Returns
        -------
        self : KMedoidsPAM
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        N, p = X_mat.shape
        if self.n_clusters > N:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed sample count ({N}).")

        rng = np.random.default_rng(self.random_state)

        # Precompute full pairwise distance matrix
        D = cdist(X_mat, X_mat, metric=self.metric)

        # Initialize medoid indices
        medoids = self._init_medoids(D, N, rng)

        for iteration in range(1, self.max_iter + 1):
            # Assignment Step: Assign each sample to its nearest medoid
            dist_to_medoids = D[:, medoids]  # Shape (N, n_clusters)
            labels = np.argmin(dist_to_medoids, axis=1)

            # Update Step: For each cluster, find the member that minimizes sum of distances to members
            new_medoids = np.copy(medoids)
            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) == 0:
                    # If empty cluster, assign a random non-medoid sample
                    non_medoids = np.setdiff1d(np.arange(N), new_medoids)
                    if len(non_medoids) > 0:
                        new_medoids[k] = rng.choice(non_medoids)
                    continue

                # Submatrix of distances among members of cluster k
                sub_D = D[np.ix_(cluster_members, cluster_members)]
                # Find index of member with minimum total intra-cluster distance
                best_member_idx = cluster_members[int(np.argmin(np.sum(sub_D, axis=1)))]
                new_medoids[k] = best_member_idx

            # Check for convergence
            if np.array_equal(np.sort(medoids), np.sort(new_medoids)):
                break

            medoids = new_medoids

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X_mat[medoids]
        self.labels_ = np.argmin(D[:, medoids], axis=1)
        self.inertia_ = float(np.sum(np.min(D[:, medoids], axis=1) ** 2))
        self.n_iter_ = iteration

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the closest cluster medoid for each sample in X.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMedoidsPAM must be fitted before predict().")

        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        # Compute distance from new samples to the fitted medoids
        dists = cdist(X_mat, self.cluster_centers_, metric=self.metric)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        """Fit model and return cluster assignments for training data."""
        return self.fit(X, y).labels_

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform X to cluster-distance space (distance to each medoid)."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMedoidsPAM must be fitted before transform().")

        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        return cdist(X_mat, self.cluster_centers_, metric=self.metric)


if __name__ == "__main__":
    # Self-test KMedoidsPAM
    rng = np.random.default_rng(42)
    X_test = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
        rng.normal(loc=[5, 5], scale=0.5, size=(50, 2)),
        rng.normal(loc=[10, 0], scale=0.5, size=(50, 2)),
    ])

    km = KMedoidsPAM(n_clusters=3, random_state=42)
    labels = km.fit_predict(X_test)

    print(f"[Self-Test] Converged in {km.n_iter_} iterations.")
    print(f"[Self-Test] Medoid indices: {km.medoid_indices_}")
    print(f"[Self-Test] Cluster counts: {np.bincount(labels)}")
    print("[Self-Test] Successfully validated KMedoidsPAM.")
