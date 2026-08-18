"""
optimizer.py

Closed-form maximum-Sharpe-ratio portfolio via the Schaible transformation:

    w* = Sigma^-1 (mu - Rf*1) / [1^T Sigma^-1 (mu - Rf*1)]

`mu` here is mu_BL (the Black-Litterman posterior returns), per the
substitution mu -> mu_p described in the derivation. No long-only / no
short-selling constraint is imposed, matching the closed-form derivation;
this can produce negative (short) weights.
"""

from __future__ import annotations

import numpy as np


class OptimizationError(Exception):
    pass


def max_sharpe_weights(mu: np.ndarray, sigma: np.ndarray, risk_free_rate: float = 0.02) -> np.ndarray:
    mu = np.asarray(mu, dtype=float).flatten()
    sigma = np.asarray(sigma, dtype=float)

    excess_returns = mu - risk_free_rate

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError as e:
        raise OptimizationError(f"Covariance matrix is not invertible: {e}") from e

    y = sigma_inv @ excess_returns
    denom = np.ones_like(y) @ y

    if np.isclose(denom, 0.0):
        raise OptimizationError(
            "1^T * Sigma^-1 * (mu - Rf*1) is ~0, so the max-Sharpe weights "
            "are undefined for this input (expected returns net of Rf sum "
            "to no directional edge)."
        )

    w = y / denom
    return w


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_free_rate: float = 0.02) -> dict:
    w = np.asarray(w, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    exp_return = float(w @ mu)
    variance = float(w @ sigma @ w)
    vol = float(np.sqrt(max(variance, 0.0)))
    sharpe = (exp_return - risk_free_rate) / vol if vol > 1e-12 else float("nan")

    return {
        "expected_return": exp_return,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }