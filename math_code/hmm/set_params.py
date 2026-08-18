#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class _HestonEKF:
  """Internal Extended Kalman Filter for Quasi-Maximum Likelihood estimation."""

  def __init__(self, dt=1.0 / 252.0):
    self.dt = dt
    # Variance of ln(chi-square(1)) measurement noise
    self.R = (np.pi**2) / 2.0
    self.euler_const = 1.2704078700237718

  def filter_pass(self, log_returns, kappa, theta, sigma):
    N = len(log_returns)
    dt = self.dt

    # Measurement z_t = ln(r_t^2) + E[ln(e_t^2)]
    r2 = np.maximum(log_returns**2, 1e-12)
    z = np.log(r2) + self.euler_const

    V_prior = float(theta)
    P_prior = 0.01
    nll = 0.0

    for t in range(N):
      # --- Measurement Update ---
      V_prior = max(V_prior, 1e-6)
      h_val = np.log(V_prior * dt)
      H = 1.0 / V_prior  # Measurement Jacobian

      y_t = z[t] - h_val  # Innovation
      S_t = H * P_prior * H + self.R  # Innovation covariance

      K_t = (P_prior * H) / S_t  # Kalman gain
      V_post = max(V_prior + K_t * y_t, 1e-6)
      P_post = (1.0 - K_t * H) * P_prior

      nll += 0.5 * (np.log(2.0 * np.pi * S_t) + (y_t**2) / S_t)

      # --- State Prediction (t -> t+1) ---
      F = 1.0 - kappa * dt  # Transition Jacobian
      V_prior = V_post + kappa * (theta - V_post) * dt
      Q = (sigma**2) * V_post * dt  # Process noise
      P_prior = F * P_post * F + Q

    return nll


def calc_params(ohlc_df, enforce_feller=True):
  """Calibrates continuous-time Heston parameters (kappa, theta, sigma)

  from an OHLC DataFrame using an Extended Kalman Filter (EKF).

  Signature matches set_params.py format: returns (kappa, theta, sigma).
  """
  df = ohlc_df.copy()

  # 1. Compute Daily Log Returns from Close prices
  close_prices = df['Close']
  log_returns = np.log(close_prices / close_prices.shift(1)).dropna().values

  if len(log_returns) == 0:
    raise ValueError('OHLC DataFrame contains insufficient price data.')

  dt = 1.0 / 252.0
  ekf = _HestonEKF(dt=dt)

  # 2. Initial Guess & Bounds
  sample_annual_var = float(np.var(log_returns) / dt)
  x0 = [1.5, max(sample_annual_var, 0.005), 0.4]

  bounds = [(0.05, 15.0), (0.001, 2.0), (0.01, 3.0)]

  constraints = []
  if enforce_feller:
    # Enforce 2 * kappa * theta - sigma^2 >= 1e-4
    constraints.append({
        'type': 'ineq',
        'fun': lambda p: 2.0 * p[0] * p[1] - p[2] ** 2 - 1e-4,
    })

  def loss(params):
    k, th, sg = params
    return ekf.filter_pass(log_returns, k, th, sg)

  # 3. Minimize EKF Negative Log-Likelihood
  res = minimize(
      loss,
      x0,
      method='SLSQP',
      bounds=bounds,
      constraints=constraints,
      options={'ftol': 1e-5, 'maxiter': 200},
  )

  kappa_opt, theta_opt, sigma_opt = res.x

  return float(kappa_opt), float(theta_opt), float(sigma_opt)