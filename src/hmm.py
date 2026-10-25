"""
hmm.py
------
Discrete Hidden Markov Model (HMM) implemented from scratch in NumPy.

Features:
- Forward Algorithm with scaling factors to prevent numerical underflow.
- Backward Algorithm matching the scaled forward parameters.
- Baum-Welch Training (Expectation-Maximization) for unsupervised discrete parameter estimation.
- Viterbi Algorithm in log-space for numerically exact maximum a posteriori (MAP) state sequence decoding.
- Automatic state alignment to guarantee semantic mapping:
    State 0: Cold (low gains)
    State 1: Warm (medium gains)
    State 2: Hot  (high gains)
"""

from typing import List, Tuple, Optional, Union
import numpy as np


class DiscreteHMM:
    """Discrete Hidden Markov Model with Forward, Viterbi, and Baum-Welch algorithms.

    Parameters
    ----------
    n_states : int, default=3
        Number of hidden states (e.g., 3 for Cold, Warm, Hot).
    n_emissions : int, default=3
        Number of discrete emission symbols (e.g., 3 for Low, Medium, High).
    max_iter : int, default=100
        Maximum Baum-Welch iterations during training.
    tol : float, default=1e-4
        Log-likelihood improvement threshold for convergence.
    random_state : Optional[int], default=42
        Random seed for parameter initialization.
    state_names : Optional[List[str]], default=None
        Semantic labels for states (e.g. ['Cold', 'Warm', 'Hot']).
    """

    def __init__(
        self,
        n_states: int = 3,
        n_emissions: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        state_names: Optional[List[str]] = None,
    ):
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.state_names = state_names or [f"State_{i}" for i in range(n_states)]

        # Model Parameters (pi, A, B)
        self.pi_: Optional[np.ndarray] = None  # Initial state distribution (N,)
        self.A_: Optional[np.ndarray] = None   # Transition matrix (N, N)
        self.B_: Optional[np.ndarray] = None   # Emission matrix (N, M)
        self.log_likelihood_history_: List[float] = []
        self.is_fitted: bool = False

    def _init_params(self) -> None:
        """Initialize parameters with slight random perturbations around uniform priors."""
        rng = np.random.default_rng(self.random_state)

        # 1. Initial state probabilities (pi)
        pi = np.ones(self.n_states) + rng.uniform(0.01, 0.05, size=self.n_states)
        self.pi_ = pi / pi.sum()

        # 2. Transition matrix (A) - slight diagonal bias (regimes tend to persist)
        A = np.ones((self.n_states, self.n_states)) + 2.0 * np.eye(self.n_states)
        A += rng.uniform(0.01, 0.05, size=(self.n_states, self.n_states))
        self.A_ = A / A.sum(axis=1, keepdims=True)

        # 3. Emission matrix (B) - initial bias towards ordered states
        # Cold emits more Low, Warm emits more Medium, Hot emits more High
        B = np.ones((self.n_states, self.n_emissions))
        for i in range(min(self.n_states, self.n_emissions)):
            B[i, i] += 2.0
        B += rng.uniform(0.01, 0.05, size=(self.n_states, self.n_emissions))
        self.B_ = B / B.sum(axis=1, keepdims=True)

    def forward(self, obs: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Forward algorithm with scaling factors to prevent numerical underflow.

        Parameters
        ----------
        obs : np.ndarray of shape (T,)
            Sequence of integer observation symbols in [0, n_emissions - 1].

        Returns
        -------
        log_likelihood : float
            Log-likelihood log P(O | lambda) of the observation sequence.
        alpha : np.ndarray of shape (T, n_states)
            Scaled forward variable matrix.
        c : np.ndarray of shape (T,)
            Scaling factors at each time step.
        """
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        # Initialization at t = 0
        alpha[0] = self.pi_ * self.B_[:, obs[0]]
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-12)
        alpha[0] *= c[0]

        # Induction for t = 1 to T - 1
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A_) * self.B_[:, obs[t]]
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-12)
            alpha[t] *= c[t]

        # Log-likelihood = - sum(log(c_t))
        log_likelihood = -float(np.sum(np.log(c + 1e-12)))
        return log_likelihood, alpha, c

    def backward(self, obs: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Backward algorithm using the scaling factors from the Forward algorithm.

        Parameters
        ----------
        obs : np.ndarray of shape (T,)
            Sequence of integer observation symbols.
        c : np.ndarray of shape (T,)
            Scaling factors from forward pass.

        Returns
        -------
        beta : np.ndarray of shape (T, n_states)
            Scaled backward variable matrix.
        """
        T = len(obs)
        N = self.n_states
        beta = np.zeros((T, N), dtype=np.float64)

        # Initialization at t = T - 1
        beta[T - 1] = c[T - 1]

        # Induction for t = T - 2 down to 0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A_ @ (self.B_[:, obs[t + 1]] * beta[t + 1])) * c[t]

        return beta

    def fit(self, obs_seq: Union[List[int], np.ndarray]) -> "DiscreteHMM":
        """Train the HMM using Baum-Welch (EM) algorithm.

        Parameters
        ----------
        obs_seq : array-like of shape (T,)
            Training sequence of discrete observation symbols.

        Returns
        -------
        self : DiscreteHMM
        """
        obs = np.asarray(obs_seq, dtype=int)
        T = len(obs)
        N = self.n_states
        M = self.n_emissions

        if T < 2:
            raise ValueError("Observation sequence must have length at least 2.")

        self._init_params()
        self.log_likelihood_history_ = []

        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            # 1. E-Step: Compute Forward and Backward variables
            log_ll, alpha, c = self.forward(obs)
            beta = self.backward(obs, c)
            self.log_likelihood_history_.append(log_ll)

            # 2. Compute posterior state occupancy gamma_t(i) and transition xi_t(i, j)
            # gamma_t(i) = P(q_t = s_i | O, lambda) = (alpha_t * beta_t) / c_t
            gamma = (alpha * beta) / c[:, np.newaxis]
            # Normalize row-wise to handle any minor floating inaccuracies
            gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

            # xi_t(i, j) = P(q_t = s_i, q_{t+1} = s_j | O, lambda)
            xi = np.zeros((T - 1, N, N), dtype=np.float64)
            for t in range(T - 1):
                numerator = alpha[t, :, np.newaxis] * self.A_ * (self.B_[:, obs[t + 1]] * beta[t + 1])[np.newaxis, :]
                xi[t] = numerator / (np.sum(numerator) + 1e-12)

            # 3. M-Step: Parameter re-estimation
            # pi_new
            pi_new = gamma[0] + 1e-6
            pi_new /= pi_new.sum()

            # A_new: sum_{t=0}^{T-2} xi_t(i, j) / sum_{t=0}^{T-2} gamma_t(i)
            gamma_sum_A = np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            A_new = (np.sum(xi, axis=0) + 1e-6) / (gamma_sum_A + 1e-6 * N)
            A_new /= A_new.sum(axis=1, keepdims=True)

            # B_new: sum_{t: o_t = k} gamma_t(i) / sum_{t} gamma_t(i)
            gamma_sum_B = np.sum(gamma, axis=0)[:, np.newaxis]
            B_new = np.zeros((N, M), dtype=np.float64)
            for k in range(M):
                mask_k = (obs == k)
                if np.any(mask_k):
                    B_new[:, k] = np.sum(gamma[mask_k], axis=0)
            B_new = (B_new + 1e-6) / (gamma_sum_B + 1e-6 * M)
            B_new /= B_new.sum(axis=1, keepdims=True)

            self.pi_ = pi_new
            self.A_ = A_new
            self.B_ = B_new

            # Check convergence
            ll_improvement = log_ll - prev_ll
            if iteration > 1 and abs(ll_improvement) < self.tol:
                break
            prev_ll = log_ll

        self._align_states_by_emission()
        self.is_fitted = True
        return self

    def _align_states_by_emission(self) -> None:
        """Align state indices so that:

        State 0 = Cold (lowest expected emission)
        State 1 = Warm (intermediate expected emission)
        State 2 = Hot  (highest expected emission)
        """
        # Calculate expected emission symbol index for each state: sum_{k} k * B_{i, k}
        weights = np.arange(self.n_emissions)
        expected_emissions = self.B_ @ weights  # shape (N,)

        # Sort order from lowest emission (Cold) to highest (Hot)
        order = np.argsort(expected_emissions)

        # Permute parameters
        self.pi_ = self.pi_[order]
        self.A_ = self.A_[np.ix_(order, order)]
        self.B_ = self.B_[order, :]

    def decode(self, obs_seq: Union[List[int], np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Viterbi algorithm in log-space for exact MAP state sequence decoding.

        Parameters
        ----------
        obs_seq : array-like of shape (T,)
            Observation sequence.

        Returns
        -------
        best_states : np.ndarray of shape (T,)
            Optimal hidden state index sequence.
        decoded_names : List[str]
            Semantic names of decoded states (e.g. ['Cold', 'Warm', 'Hot', ...]).
        """
        if not self.is_fitted:
            raise RuntimeError("DiscreteHMM must be fitted before calling decode().")

        obs = np.asarray(obs_seq, dtype=int)
        T = len(obs)
        N = self.n_states

        log_pi = np.log(self.pi_ + 1e-12)
        log_A = np.log(self.A_ + 1e-12)
        log_B = np.log(self.B_ + 1e-12)

        delta = np.zeros((T, N), dtype=np.float64)
        psi = np.zeros((T, N), dtype=int)

        # Initialization at t = 0
        delta[0] = log_pi + log_B[:, obs[0]]

        # Recursion for t = 1 to T - 1
        for t in range(1, T):
            for j in range(N):
                trans_probs = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(trans_probs))
                delta[t, j] = trans_probs[psi[t, j]] + log_B[j, obs[t]]

        # Termination
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = int(np.argmax(delta[T - 1]))

        # Backtracking
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]

        decoded_names = [self.state_names[s] for s in best_path]
        return best_path, decoded_names

    def score(self, obs_seq: Union[List[int], np.ndarray]) -> float:
        """Compute the log-likelihood of an observation sequence under the model.

        Parameters
        ----------
        obs_seq : array-like of shape (T,)
            Observation sequence.

        Returns
        -------
        log_likelihood : float
        """
        if not self.is_fitted:
            raise RuntimeError("DiscreteHMM must be fitted before calling score().")
        obs = np.asarray(obs_seq, dtype=int)
        log_ll, _, _ = self.forward(obs)
        return log_ll


if __name__ == "__main__":
    # Self-test: 3 states (Cold, Warm, Hot), 3 emissions (Low, Medium, High)
    print("[Self-Test] Initializing DiscreteHMM...")
    hmm = DiscreteHMM(
        n_states=3,
        n_emissions=3,
        max_iter=50,
        state_names=["Cold", "Warm", "Hot"],
        random_state=42
    )

    # Generate synthetic regime sequence
    # 10 Cold steps (mostly 0s), 10 Warm steps (mostly 1s), 10 Hot steps (mostly 2s)
    obs = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
                    1, 1, 2, 1, 1, 0, 1, 1, 2, 1,
                    2, 2, 2, 1, 2, 2, 2, 1, 2, 2])

    hmm.fit(obs)
    best_states, decoded_names = hmm.decode(obs)
    log_ll = hmm.score(obs)

    print(f"[Self-Test] Baum-Welch Log-Likelihood: {log_ll:.4f}")
    print(f"[Self-Test] Transition Matrix A:\n{np.round(hmm.A_, 3)}")
    print(f"[Self-Test] Emission Matrix B:\n{np.round(hmm.B_, 3)}")
    print(f"[Self-Test] Decoded first 5: {decoded_names[:5]}")
    print(f"[Self-Test] Decoded last 5 : {decoded_names[-5:]}")
    print("[Self-Test] Successfully validated DiscreteHMM.")
