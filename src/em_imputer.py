"""
em_imputer.py
-------------
Expectation-Maximization (EM) Imputation for Multivariate Gaussian Data,
implemented from scratch using NumPy.

Mathematical Foundation (Little & Rubin, 2002):
Given a dataset X with missing entries under Missing At Random (MAR) assumption,
we assume X ~ N(mu, Sigma).
- E-Step:
  Compute the expected values and covariances of the missing entries conditioned
  on the observed values for each observation i:
    E[x_{i,M} | x_{i,O}] = mu_M + Sigma_{MO} * (Sigma_{OO} + lambda*I)^{-1} * (x_{i,O} - mu_O)
    Cov[x_{i,M} | x_{i,O}] = Sigma_{MM} - Sigma_{MO} * (Sigma_{OO} + lambda*I)^{-1} * Sigma_{OM}
- M-Step:
  Update the maximum likelihood estimates of mean mu and covariance Sigma using
  the completed sufficient statistics:
    mu^{(t+1)} = (1 / N) * sum_{i=1}^N x̂_i
    Sigma^{(t+1)} = (1 / N) * sum_{i=1}^N [ (x̂_i - mu^{(t+1)})(x̂_i - mu^{(t+1)})^T + C_i ] + lambda*I
- Convergence:
  Stop when relative change in parameter estimates is less than tolerance 'tol'.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EMGaussianImputer(BaseEstimator, TransformerMixin):
    """Expectation-Maximization (EM) Imputer for a Multivariate Gaussian distribution.

    Implemented from scratch using NumPy with Scikit-Learn estimator conventions.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Tolerance for convergence based on relative parameter update magnitude.
    ridge : float, default=1e-4
        Ridge regularization term added to covariance diagonal to guarantee
        positive definiteness and numerical stability during matrix inversion.
    verbose : bool, default=False
        If True, prints progress and convergence step info.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        ridge: float = 1e-4,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.verbose = verbose
        self.random_state = random_state

        # Fitted parameters
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.convergence_history_: List[float] = []
        self.n_iter_: int = 0
        self.feature_names_: Optional[List[str]] = None

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Helper to convert DataFrame/ndarray to float64 numpy array."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            return X.to_numpy(dtype=np.float64, copy=True)
        return np.asarray(X, dtype=np.float64)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "EMGaussianImputer":
        """Fit the multivariate Gaussian distribution (mu and Sigma) using EM on training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training data containing missing values (np.nan).
        y : Ignored
            Scikit-learn API compatibility.

        Returns
        -------
        self : EMGaussianImputer
            Fitted estimator.
        """
        X_mat = self._validate_input(X)
        N, p = X_mat.shape

        if N == 0 or p == 0:
            raise ValueError("Input array cannot be empty.")

        # Initial Mean (column-wise mean of observed values)
        mu = np.nanmean(X_mat, axis=0)
        # Fallback for columns with all NaN
        mu = np.where(np.isnan(mu), 0.0, mu)

        # Initial baseline imputation (mean) to estimate initial covariance
        X_init = X_mat.copy()
        for j in range(p):
            nan_mask = np.isnan(X_init[:, j])
            X_init[nan_mask, j] = mu[j]

        # Initial sample covariance with ridge
        sigma = np.cov(X_init, rowvar=False) + self.ridge * np.eye(p)
        if sigma.ndim == 0:  # If p == 1
            sigma = np.array([[float(sigma)]])

        self.convergence_history_ = []

        # EM Iteration Loop
        for iteration in range(1, self.max_iter + 1):
            mu_old = mu.copy()
            sigma_old = sigma.copy()

            # -------------------------------------------------------------
            # E-Step: Compute conditional expectations and covariances
            # -------------------------------------------------------------
            X_completed = np.empty_like(X_mat)
            C_sum = np.zeros((p, p), dtype=np.float64)

            for i in range(N):
                row = X_mat[i]
                nan_mask = np.isnan(row)
                obs_idx = np.where(~nan_mask)[0]
                mis_idx = np.where(nan_mask)[0]

                if len(mis_idx) == 0:
                    # Fully observed sample
                    X_completed[i] = row
                elif len(obs_idx) == 0:
                    # Fully missing sample
                    X_completed[i] = mu
                    C_sum += sigma
                else:
                    # Partition parameters into observed and missing blocks
                    mu_O = mu[obs_idx]
                    mu_M = mu[mis_idx]
                    Sigma_OO = sigma[np.ix_(obs_idx, obs_idx)] + self.ridge * np.eye(len(obs_idx))
                    Sigma_MM = sigma[np.ix_(mis_idx, mis_idx)]
                    Sigma_MO = sigma[np.ix_(mis_idx, obs_idx)]
                    Sigma_OM = sigma[np.ix_(obs_idx, mis_idx)]

                    # Invert Sigma_OO using solve / pinv for maximum numerical stability
                    try:
                        W = np.linalg.solve(Sigma_OO, Sigma_OM).T  # Sigma_MO @ inv(Sigma_OO)
                    except np.linalg.LinAlgError:
                        W = Sigma_MO @ np.linalg.pinv(Sigma_OO)

                    # Conditional expectation: E[x_M | x_O]
                    diff_O = row[obs_idx] - mu_O
                    cond_mean = mu_M + W @ diff_O

                    # Conditional covariance: Cov[x_M | x_O]
                    cond_cov = Sigma_MM - W @ Sigma_OM
                    # Ensure positive semi-definite
                    cond_cov = 0.5 * (cond_cov + cond_cov.T)

                    # Populate completed vector
                    completed_row = row.copy()
                    completed_row[mis_idx] = cond_mean
                    X_completed[i] = completed_row

                    # Accumulate conditional covariance into missing submatrix
                    C_sum[np.ix_(mis_idx, mis_idx)] += cond_cov

            # -------------------------------------------------------------
            # M-Step: Update mu and Sigma
            # -------------------------------------------------------------
            mu_new = np.mean(X_completed, axis=0)

            # Center completed data with new mean
            diff = X_completed - mu_new
            # Expected scatter matrix: (diff.T @ diff) + sum(C_i)
            sigma_new = (diff.T @ diff + C_sum) / N + self.ridge * np.eye(p)
            sigma_new = 0.5 * (sigma_new + sigma_new.T)

            # Compute relative parameter change for convergence check
            mu_diff = np.linalg.norm(mu_new - mu_old) / (np.linalg.norm(mu_old) + 1e-6)
            sigma_diff = np.linalg.norm(sigma_new - sigma_old, ord="fro") / (np.linalg.norm(sigma_old, ord="fro") + 1e-6)
            total_change = float(mu_diff + sigma_diff)
            self.convergence_history_.append(total_change)

            mu = mu_new
            sigma = sigma_new

            if self.verbose:
                print(f"[EM Iteration {iteration:02d}] Parameter delta: {total_change:.6e}")

            if total_change < self.tol:
                if self.verbose:
                    print(f"[EM Converged] Reached tolerance {self.tol} at iteration {iteration}.")
                break

        self.mu_ = mu
        self.sigma_ = sigma
        self.n_iter_ = iteration
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Impute missing values using conditional expectation based on fitted mu_ and sigma_.

        Note: Parameters are strictly NOT updated, ensuring zero data leakage when
        applied to test data.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Dataset with missing values to impute.

        Returns
        -------
        X_imputed : array-like or pd.DataFrame of shape (n_samples, n_features)
            Fully imputed dataset.
        """
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("EMGaussianImputer must be fitted before calling transform().")

        is_pandas = isinstance(X, pd.DataFrame)
        cols = list(X.columns) if is_pandas else None
        index = X.index if is_pandas else None

        X_mat = self._validate_input(X)
        N, p = X_mat.shape

        if p != len(self.mu_):
            raise ValueError(f"Feature count mismatch: Expected {len(self.mu_)}, got {p}")

        X_imputed = X_mat.copy()

        for i in range(N):
            row = X_mat[i]
            nan_mask = np.isnan(row)
            obs_idx = np.where(~nan_mask)[0]
            mis_idx = np.where(nan_mask)[0]

            if len(mis_idx) == 0:
                continue
            elif len(obs_idx) == 0:
                X_imputed[i] = self.mu_
            else:
                mu_O = self.mu_[obs_idx]
                mu_M = self.mu_[mis_idx]
                Sigma_OO = self.sigma_[np.ix_(obs_idx, obs_idx)] + self.ridge * np.eye(len(obs_idx))
                Sigma_MO = self.sigma_[np.ix_(mis_idx, obs_idx)]
                Sigma_OM = self.sigma_[np.ix_(obs_idx, mis_idx)]

                try:
                    W = np.linalg.solve(Sigma_OO, Sigma_OM).T
                except np.linalg.LinAlgError:
                    W = Sigma_MO @ np.linalg.pinv(Sigma_OO)

                diff_O = row[obs_idx] - mu_O
                cond_mean = mu_M + W @ diff_O
                X_imputed[i, mis_idx] = cond_mean

        if is_pandas:
            return pd.DataFrame(X_imputed, columns=cols, index=index)
        return X_imputed

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> Union[pd.DataFrame, np.ndarray]:
        """Fit EM model and return imputed training dataset."""
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    # Quick self-test with known multivariate normal data
    rng = np.random.default_rng(42)
    true_mu = np.array([10.0, 50.0, 100.0])
    true_cov = np.array([
        [4.0, 3.0, 1.0],
        [3.0, 9.0, 2.0],
        [1.0, 2.0, 16.0]
    ])
    synthetic_X = rng.multivariate_normal(true_mu, true_cov, size=200)

    # Introduce 15% random missing values
    X_missing = synthetic_X.copy()
    mask = rng.uniform(0, 1, size=synthetic_X.shape) < 0.15
    X_missing[mask] = np.nan

    imputer = EMGaussianImputer(max_iter=50, tol=1e-5, verbose=True)
    X_filled = imputer.fit_transform(X_missing)

    rmse = np.sqrt(np.mean((synthetic_X[mask] - X_filled[mask]) ** 2))
    print(f"\n[Self-Test] Completed in {imputer.n_iter_} iterations.")
    print(f"[Self-Test] Imputation RMSE on masked entries: {rmse:.4f}")
    assert not np.isnan(X_filled).any(), "Result contains unhandled NaNs!"
    print("[Self-Test] Successfully validated EMGaussianImputer.")
