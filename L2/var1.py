"""VAR(1) estimation and simulation: r_t = c + A r_{t-1} + eps_t, eps_t ~ N(0, Sigma)."""
import numpy as np


def fit_var1(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS estimate of c (n,), A (n, n), Sigma (n, n) from a (T, n) return matrix.

    Row i of A is asset i's equation: r_{i,t} = c_i + sum_j A[i, j] r_{j,t-1}.
    """
    Y = returns[1:]                                       # weeks 2..T
    X = np.column_stack([np.ones(len(Y)), returns[:-1]])  # intercept + weeks 1..T-1
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)             # (1 + n, n)

    c = B[0]
    A = B[1:].T  # Y = X B  =>  y_t = c + B[1:]^T r_{t-1}
    residuals = Y - X @ B
    sigma = residuals.T @ residuals / (len(Y) - X.shape[1])
    return c, A, sigma


def simulate_var1(c, A, sigma, r0, n_scenarios: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate returns, shape (n_scenarios, n_steps, n). All scenarios start from r0."""
    L = np.linalg.cholesky(sigma)
    out = np.empty((n_scenarios, n_steps, len(c)))
    r = np.broadcast_to(r0, (n_scenarios, len(c)))
    for t in range(n_steps):
        r = c + r @ A.T + rng.standard_normal((n_scenarios, len(c))) @ L.T
        out[:, t] = r
    return out


def returns_to_levels(returns: np.ndarray, start_levels: np.ndarray) -> np.ndarray:
    """Cumulate log-returns (axis 1 = time) into price / rate levels."""
    return start_levels * np.exp(np.cumsum(returns, axis=1))
