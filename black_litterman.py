"""
black_litterman.py

Pure math, no I/O. Implements:

  Pi        = lambda * Sigma @ w_mkt                              (implied equilibrium returns)
  Omega     = P @ (tau * Sigma) @ P.T                              (view uncertainty)
  mu_BL     = [(tau*Sigma)^-1 + P.T @ Omega^-1 @ P]^-1
              @ [(tau*Sigma)^-1 @ Pi + P.T @ Omega^-1 @ Q]         (posterior / "blended" returns)

Shapes:
  Sigma : (n, n)      covariance matrix
  w_mkt : (n,)        market-cap weights, sums to 1
  P     : (k, n)      pick matrix, one row per view
  Q     : (k,)        view returns (decimal, e.g. 0.05 for +5%)
  Omega : (k, k)
"""

from __future__ import annotations

import numpy as np


class BlackLittermanError(Exception):
    pass


def implied_equilibrium_returns(sigma: np.ndarray, w_mkt: np.ndarray, risk_aversion: float = 2.5) -> np.ndarray:
    """Pi = lambda * Sigma @ w_mkt"""
    sigma = np.asarray(sigma, dtype=float)
    w_mkt = np.asarray(w_mkt, dtype=float)
    return risk_aversion * sigma @ w_mkt


def view_uncertainty(sigma: np.ndarray, P: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Omega = P @ (tau * Sigma) @ P.T"""
    sigma = np.asarray(sigma, dtype=float)
    P = np.asarray(P, dtype=float)
    tau_sigma = tau * sigma
    omega = P @ tau_sigma @ P.T
    return omega


def posterior_returns(
    sigma: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full Black-Litterman blend. Returns (mu_bl, omega).

    If P/Q are empty (no views supplied), mu_bl collapses to pi, which is
    the correct limiting case of the formula as the investor expresses no
    active views.
    """
    sigma = np.asarray(sigma, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1, 1)
    n = sigma.shape[0]

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if P.size == 0:
        return pi.flatten(), np.zeros((0, 0))

    P = P.reshape(-1, n)
    Q = Q.reshape(-1, 1)

    tau_sigma = tau * sigma
    try:
        tau_sigma_inv = np.linalg.inv(tau_sigma)
    except np.linalg.LinAlgError as e:
        raise BlackLittermanError(f"tau*Sigma is not invertible: {e}") from e

    omega = view_uncertainty(sigma, P, tau)
    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError as e:
        raise BlackLittermanError(
            "Omega is not invertible. This usually means two views are "
            "expressed identically (linearly dependent pick-matrix rows)."
        ) from e

    middle = tau_sigma_inv + P.T @ omega_inv @ P
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError as e:
        raise BlackLittermanError(f"Posterior precision matrix is not invertible: {e}") from e

    right = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    mu_bl = middle_inv @ right

    return mu_bl.flatten(), omega