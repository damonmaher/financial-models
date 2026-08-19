#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------ Risk-Free Interest ------------------------

def calc_sofr_rates(T, r_3m=0.0430, r_5y=0.0380):
    """Calculates interpolated interest rates using an offline yield curve (no API calls)."""
    curve_maturities = np.array([0.0, 30 / 365, 90 / 365, 180 / 365, 365 / 365, 5.0])
    
    r_0d = r_3m - 0.0015
    r_30d = r_3m
    r_90d = r_3m
    r_180d = r_3m + (r_5y - r_3m) * (90 / (5 * 365 - 90))
    r_365d = r_3m + (r_5y - r_3m) * (275 / (5 * 365 - 90))

    curve_rates = np.array([r_0d, r_30d, r_90d, r_180d, r_365d, r_5y])
    return np.interp(T, curve_maturities, curve_rates)

# --------- EKF State-Space Filter & Parameter Estimation -----------

class _HestonEKF:
    """Extended Kalman Filter for Heston latent variance state-space estimation."""

    def __init__(self, dt=1.0 / 252.0):
        self.dt = dt
        self.R = (np.pi**2) / 2.0
        self.euler_const = 1.2704078700237718

    def filter_pass(self, log_returns, kappa, theta, sigma):
        N = len(log_returns)
        dt = self.dt

        r2 = np.maximum(log_returns**2, 1e-12)
        z = np.log(r2) + self.euler_const

        V_prior = float(theta)
        P_prior = 0.01

        nll = 0.0
        v_filtered = np.zeros(N)

        for t in range(N):
            V_prior = max(V_prior, 1e-6)
            h_val = np.log(V_prior * dt)
            H = 1.0 / V_prior

            y_t = z[t] - h_val
            S_t = H * P_prior * H + self.R

            K_t = (P_prior * H) / S_t
            V_post = max(V_prior + K_t * y_t, 1e-6)
            P_post = (1.0 - K_t * H) * P_prior

            v_filtered[t] = V_post

            nll += 0.5 * (np.log(2.0 * np.pi * S_t) + (y_t**2) / S_t)

            F = 1.0 - kappa * dt
            V_prior = V_post + kappa * (theta - V_post) * dt
            Q = (sigma**2) * V_post * dt
            P_prior = F * P_post * F + Q

        return nll, v_filtered

def calc_params(history_data: pd.DataFrame):
    """Calibrates Heston parameters using pre-fetched historical data (0 extra API calls)."""
    S = history_data['Close'].values
    log_returns = np.log(S[1:] / S[:-1])

    if len(log_returns) < 30:
        raise ValueError('Insufficient historical price data')

    dt = 1.0 / 252.0
    ekf = _HestonEKF(dt=dt)

    sample_annual_var = float(np.var(log_returns) / dt)
    x0 = [1.5, max(sample_annual_var, 0.005), 0.4]

    bounds = [(0.05, 15.0), (0.001, 2.0), (0.01, 3.0)]

    feller_constraint = {
        'type': 'ineq',
        'fun': lambda p: 2.0 * p[0] * p[1] - p[2] ** 2 - 1e-4,
    }

    def loss_function(params):
        k, th, sg = params
        nll, _ = ekf.filter_pass(log_returns, k, th, sg)
        return nll

    res = minimize(
        loss_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=[feller_constraint],
        options={'ftol': 1e-5, 'maxiter': 200},
    )

    kappa, theta, vol_of_vol = res.x

    _, v_filtered = ekf.filter_pass(log_returns, kappa, theta, vol_of_vol)
    v0 = float(v_filtered[-1])

    mu = np.mean(log_returns) / dt
    v_lagged = v_filtered[:-1]
    r_trimmed = log_returns[1:]

    asset_shocks = (r_trimmed - (mu - 0.5 * v_lagged) * dt) / np.sqrt(v_lagged * dt)

    delta_v = np.diff(v_filtered)
    expected_delta_v = kappa * (theta - v_lagged) * dt
    variance_shocks = (delta_v - expected_delta_v) / (vol_of_vol * np.sqrt(v_lagged * dt))

    rho = float(np.corrcoef(asset_shocks, variance_shocks)[0, 1])
    rho = np.clip(rho, -0.99, 0.99)

    return (
        float(kappa),
        float(theta),
        float(vol_of_vol),
        float(rho),
        float(v0),
    )
