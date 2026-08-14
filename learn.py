import numpy as np
import pandas as pd

def matrices(kappa, sigma, theta, closes):
    #Instantiate Transition Matrix
    A_base = np.array([
        [0.70, 0.20, 0.08, 0.02, 0.00, 0.00, 0.00], # 0: Sev Bullish
        [0.10, 0.70, 0.15, 0.05, 0.00, 0.00, 0.00], # 1: Bullish
        [0.02, 0.12, 0.75, 0.09, 0.02, 0.00, 0.00], # 2: Sl Bullish
        [0.00, 0.03, 0.12, 0.70, 0.12, 0.03, 0.00], # 3: Stagnant
        [0.00, 0.00, 0.02, 0.12, 0.70, 0.14, 0.02], # 4: Sl Bearish
        [0.00, 0.00, 0.00, 0.05, 0.15, 0.65, 0.15], # 5: Bearish
        [0.00, 0.00, 0.00, 0.00, 0.05, 0.25, 0.70]  # 6: Sev Bearish
    ])

    #Instantiate Emission Matrix
    B_base = np.array([
        [0.85, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00], # 0: Sev Bullish
        [0.10, 0.75, 0.10, 0.05, 0.00, 0.00, 0.00], # 1: Bullish
        [0.02, 0.10, 0.76, 0.10, 0.02, 0.00, 0.00], # 2: Sl Bullish
        [0.01, 0.04, 0.15, 0.60, 0.15, 0.04, 0.01], # 3: Stagnant
        [0.00, 0.02, 0.08, 0.15, 0.65, 0.08, 0.02], # 4: Sl Bearish
        [0.00, 0.00, 0.05, 0.10, 0.15, 0.60, 0.10], # 5: Bearish
        [0.02, 0.00, 0.00, 0.05, 0.10, 0.23, 0.60]  # 6: Sev Bearish
    ])

    daily_log_returns = np.log(closes / closes.shift(1))
    rolling_drift_10d = daily_log_returns.rolling(window=10).mean()
    drift = rolling_drift_10d.dropna().values


    A = set_transition_matrix(A_base, kappa, sigma, theta)
    B = set_emission_matrix(B_base, kappa, sigma, theta)
    pi = set_init(drift, theta)

    return A, B, pi



def set_transition_matrix(matrix, kappa, sigma, theta):
    mod_matrix = matrix.copy()
    for i in range(7):
        row = mod_matrix[i]

        #1: Vol-of-Vol, move values closer to the average
        row_avg = np.mean(row)
        dist_to_avg = row-row_avg
        row -= 0.2 * dist_to_avg * sigma

        row = np.maximum(row, 0.0001)

        #2: Mean reversion, increase probabilities for matrix[i][i]s
        row[i] += 0.02 * kappa

        #3: Long-term var, bearish skew for bearish states
        row[6] += 0.5 * theta
        row[5] += 0.25 * theta
        row[4] += 0.125 * theta

        mod_matrix[i] = row

    final_matrix = mod_matrix / mod_matrix.sum(axis=1, keepdims=True)
    return final_matrix

def set_emission_matrix(matrix, kappa, sigma, theta):
    mod_matrix = matrix.copy()
    for i in range(7):
       row = mod_matrix[i]

       #1: Long-term var-> purely quantitative matrix, so widens distribution
       row_avg = np.mean(row)
       dist_to_avg = row - row_avg
       row -= 0.5 * dist_to_avg * theta
       row = np.maximum(row, 0.0001)

       #2: Vol-of-vol means more kurtosis
       row[0] += 0.1 * sigma
       row[6] += 0.1 * sigma

       #3: Mean-R means increases diagonals
       row[i] += 0.05 * kappa
       mod_matrix[i] = row
    return mod_matrix / mod_matrix.sum(axis=1, keepdims=True)


def set_init(drifts, theta):
    pi_base = np.zeros(7)
    total_obs = len(drifts)

    #Define annualized drift boundaries
    b_30 = 0.30 / 252
    b_15 =  0.15 / 252
    b_05 = 0.05 / 252

    pi_base[0] = np.sum(drifts > b_30)                               # 0: Sev Bullish
    pi_base[1] = np.sum((drifts > b_15) & (drifts <= b_30))      # 1: Bullish
    pi_base[2] = np.sum((drifts > b_05) & (drifts <= b_15))      # 2: Sl Bullish
    pi_base[3] = np.sum((drifts >= -b_05) & (drifts <= b_05))    # 3: Stagnant
    pi_base[4] = np.sum((drifts >= -b_15) & (drifts < -b_05))    # 4: Sl Bearish
    pi_base[5] = np.sum((drifts >= -b_30) & (drifts < -b_15))    # 5: Bearish
    pi_base[6] = np.sum(drifts < -b_30)                              # 6: Sev Bearish

    pi_base = pi_base / total_obs
    pi_base = np.maximum(pi_base, 0.001)
    pi_base = pi_base / pi_base.sum()

    pi_mod = pi_base.copy()
    pi_mod[6] += 0.5 * theta
    pi_mod[5] += 0.25 * theta
    pi_mod[4] += 0.125 * theta

    pi_final = pi_mod / pi_mod.sum()
    return pi_final
