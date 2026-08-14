#!/usr/bin/env python3

import numpy as np
from scipy.special import erf

def raw_bs(s_nought, strike, risk_free, vol, ttm):
    vol = np.where(vol<1e-4, 1e-4, vol) # ensure vol can't be zero
    d1 = (np.log(s_nought/strike)+(risk_free+(vol*vol)/2)*ttm)/(vol * np.sqrt(ttm))
    d2 = d1 - vol * np.sqrt(ttm)
    N_d1 = 0.5 * (1.0 + erf(d1/np.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + erf(d2/np.sqrt(2.0)))
    return s_nought * N_d1 - strike * np.exp(-1 * risk_free * ttm) * N_d2

def inv_bs(s_nought, strike, risk_free, ttm, c_market, q):
    # Initial volatlity guess
    vol = np.full_like(strike, 0.4, dtype=float)

    # Convert ITM calls (K<S0) to OTM puts using Put-Call Parity
    otm_target_prices = np.where(strike < s_nought,
                                 c_market - s_nought * np.exp(-q * ttm) + strike * np.exp(-risk_free * ttm),
                                 c_market)

    for _ in range(20): #increased iterations
        vol = np.maximum(vol, 1e-4)
        d1 = (np.log(s_nought/strike)+(risk_free-q+(vol*vol)/2)*ttm)/(vol * np.sqrt(ttm))
        d2 = d1 - vol * np.sqrt(ttm)
        N_d1 = 0.5 * (1.0 + erf(d1/np.sqrt(2.0)))
        N_d2 = 0.5 * (1.0 + erf(d2/np.sqrt(2.0)))

        #Black-Scholes solution
        call_prices = s_nought * np.exp(-q * ttm) * N_d1 - strike * np.exp(-risk_free * ttm) * N_d2

        calc_otm_prices = np.where(strike < s_nought,
                                   call_prices - s_nought * np.exp(-q * ttm) + strike * np.exp(-risk_free * ttm),
                                   call_prices)

        vega = s_nought * np.exp(-q * ttm) * np.sqrt(ttm) * (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
        vega = np.maximum(vega, 1e-6)

        vol = vol - (calc_otm_prices - otm_target_prices) / vega

    vol = np.where((vol<=0.001) | (vol > 3.5) | np.isnan(vol), 0.0, vol)
    return vol
