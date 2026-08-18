#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import yfinance as yf


# ------------------------ Risk-Free Interest ------------------------


def calc_sofr_rates(T):
  # define a robust array we can interpolate with (x-axis)
  curve_maturities = np.array([0.0, 30 / 365, 90 / 365, 180 / 365, 365 / 365, 5.0])
  try:
    # fetch current market yields
    r_3m = yf.Ticker('^IRX').history(period='1d')['Close'].iloc[-1] / 100.0
    r_5y = yf.Ticker('^FVX').history(period='1d')['Close'].iloc[-1] / 100.0

    r_0d = r_3m - 0.0015  # Assume 1m is slightly higher
    r_30d = r_3m
    r_90d = r_3m

    r_180d = r_3m + (r_5y - r_3m) * (90 / (5 * 365 - 90))
    r_365d = r_3m + (r_5y - r_3m) * (275 / (5 * 365 - 90))

    # y-axis for the interpolation
    curve_rates = np.array([r_0d, r_30d, r_90d, r_180d, r_365d, r_5y])

    exact_r = np.interp(T, curve_maturities, curve_rates)

    return exact_r

  except Exception as e:
    if hasattr(T, '__len__'):
      return np.full(len(T), 0.0525)
    return 0.0525  # if API fails, return to safe benchmark


# --------- EKF State-Space Filter & Parameter Estimation -----------


class _HestonEKF:
  """Extended Kalman Filter for Heston latent variance state-space estimation."""

  def __init__(self, dt=1.0 / 252.0):
    self.dt = dt
    # Measurement noise variance for log-squared normal shocks: E[ln(chi^2_1)] variance = pi^2 / 2
    self.R = (np.pi**2) / 2.0
    self.euler_const = 1.2704078700237718

  def filter_pass(self, log_returns, kappa, theta, sigma):
    """Runs a forward EKF pass and calculates the Quasi-Negative Log-Likelihood."""
    N = len(log_returns)
    dt = self.dt

    # Log-squared return transformation: z_t = ln(r_t^2) + E[ln(e_t^2)]
    r2 = np.maximum(log_returns**2, 1e-12)
    z = np.log(r2) + self.euler_const

    V_prior = float(theta)
    P_prior = 0.01

    nll = 0.0
    v_filtered = np.zeros(N)

    for t in range(N):
      # --- 1. Measurement Update ---
      V_prior = max(V_prior, 1e-6)
      h_val = np.log(V_prior * dt)
      H = 1.0 / V_prior  # Measurement Jacobian: dh/dV

      y_t = z[t] - h_val  # Innovation
      S_t = H * P_prior * H + self.R  # Innovation variance

      K_t = (P_prior * H) / S_t  # Kalman Gain
      V_post = max(V_prior + K_t * y_t, 1e-6)  # Updated state estimate
      P_post = (1.0 - K_t * H) * P_prior

      v_filtered[t] = V_post

      # Accumulate Log-Likelihood
      nll += 0.5 * (np.log(2.0 * np.pi * S_t) + (y_t**2) / S_t)

      # --- 2. State Prediction (t -> t+1) ---
      F = 1.0 - kappa * dt  # Transition Jacobian: df/dV
      V_prior = V_post + kappa * (theta - V_post) * dt
      Q = (sigma**2) * V_post * dt  # Process noise variance
      P_prior = F * P_post * F + Q

    return nll, v_filtered


def calc_params(ticker):
  """Calibrates Heston model parameters (kappa, theta, vol_of_vol, rho, v0)

  using an Extended Kalman Filter (EKF) state-space model.
  """
  # Fetch 2-year chronological historical data
  history_data = yf.Ticker(ticker).history(period='2y')
  S = history_data['Close'].values
  log_returns = np.log(S[1:] / S[:-1])

  if len(log_returns) < 30:
    raise ValueError(f'Insufficient historical price data for ticker: {ticker}')

  dt = 1.0 / 252.0
  ekf = _HestonEKF(dt=dt)

  # Initial parameter guesses: [kappa, theta, sigma]
  sample_annual_var = float(np.var(log_returns) / dt)
  x0 = [1.5, max(sample_annual_var, 0.005), 0.4]

  # Parameter bounds
  bounds = [(0.05, 15.0), (0.001, 2.0), (0.01, 3.0)]

  # Enforce Feller condition: 2 * kappa * theta >= sigma^2
  feller_constraint = {
      'type': 'ineq',
      'fun': lambda p: 2.0 * p[0] * p[1] - p[2] ** 2 - 1e-4,
  }

  def loss_function(params):
    k, th, sg = params
    nll, _ = ekf.filter_pass(log_returns, k, th, sg)
    return nll

  # Calibrate kappa, theta, vol_of_vol via QMLE
  res = minimize(
      loss_function,
      x0,
      method='SLSQP',
      bounds=bounds,
      constraints=[feller_constraint],
      options={'ftol': 1e-5, 'maxiter': 200},
  )

  kappa, theta, vol_of_vol = res.x

  # Re-run filter with optimal parameters to extract latent variance trajectory
  _, v_filtered = ekf.filter_pass(log_returns, kappa, theta, vol_of_vol)

  # Current spot variance (v0 is the latest EKF state estimate)
  v0 = float(v_filtered[-1])

  # --- Compute Leverage Correlation (rho) ---
  # Asset shocks: e_S = (r_t - drift) / sqrt(V_{t-1} * dt)
  mu = np.mean(log_returns) / dt
  v_lagged = v_filtered[:-1]
  r_trimmed = log_returns[1:]

  asset_shocks = (r_trimmed - (mu - 0.5 * v_lagged) * dt) / np.sqrt(
      v_lagged * dt
  )

  # Variance shocks: e_V = (V_t - V_{t-1} - drift_V) / (sigma * sqrt(V_{t-1} * dt))
  delta_v = np.diff(v_filtered)
  expected_delta_v = kappa * (theta - v_lagged) * dt
  variance_shocks = (delta_v - expected_delta_v) / (
      vol_of_vol * np.sqrt(v_lagged * dt)
  )

  # Correlation coefficient rho
  rho = float(np.corrcoef(asset_shocks, variance_shocks)[0, 1])

  # Guard against NaN/boundary issues in correlation
  rho = np.clip(rho, -0.99, 0.99)

  return (
      float(kappa),
      float(theta),
      float(vol_of_vol),
      float(rho),
      float(v0),
  )


# ------------------------ Test Run ------------------------
if __name__ == '__main__':
  ticker = 'AAPL'
  print(f'Calibrating EKF Heston Parameters for {ticker}...')
  kappa, theta, vol_of_vol, rho, v0 = calc_params(ticker)

  print('\n--- EKF Calibrated Parameters ---')
  print(f'Speed of Mean Reversion (kappa) : {kappa:.4f}')
  print(f'Long-Term Variance (theta)      : {theta:.4f} (Vol: {np.sqrt(theta)*100:.2f}%)')
  print(f'Vol-of-Vol (sigma)              : {vol_of_vol:.4f}')
  print(f'Leverage Correlation (rho)      : {rho:.4f}')
  print(f'Current Spot Variance (v0)      : {v0:.4f} (Vol: {np.sqrt(v0)*100:.2f}%)')