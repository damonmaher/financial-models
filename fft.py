 #!/usr/bin/env python3

import numpy as np
import yfinance as yf


def heston_fft_main(S0, T, kappa, theta, sigma, rho, v0, r, q, is_call=True):
    N=4096
    B=100.0
    dv = B/N #frequency step size
    dk = (2*np.pi) / (N*dv) #strike step size

    alpha = 1.5 if is_call else -1.5 #dampening factor

    j = np.arange(N) #base iteraitons grid
    v = j * dv #frequency grid

    b=(N*dk)/2 #minimum strike price
    k_m = -b+j*dk #center around minimum strike

    def heston_char_func(u):
        u_shifted = u - (alpha + 1) * 1j #transformed frequency for dampening

        #Heston structure:
        d = np.sqrt((rho * sigma * 1j * u_shifted - kappa)**2 + (sigma**2) * (u_shifted**2 + 1j * u_shifted))
        g = (kappa - rho * sigma * 1j * u_shifted - d) / (kappa - rho * sigma * 1j * u_shifted + d)

        A_term = 1j * u_shifted * (np.log(S0) + (r-q) * T)
        B_term = (kappa * theta / (sigma**2)) * (
            (kappa-rho*sigma*1j*u_shifted - d) * T - 2 * np.log((1-g*np.exp(-d*T)) / (1-g))
        )
        C_term = (v0 / (sigma**2)) * (kappa-rho*sigma*1j*u_shifted-d)*(
            (1-np.exp(-d*T)) / (1-g*np.exp(-d*T))
        )
        return np.exp(A_term+B_term+C_term)

    #c(v) function
    phi = heston_char_func(v)
    denominator = (alpha + 1j * v) * (alpha + 1.0 + 1j  * v)
    psi = (np.exp(-r * T) * phi) / denominator

    psi[0] = psi[0]/2.0 #handle NaNs

    #simpson's integration weights
    w = np.ones(N)
    w[0] = 1/3
    w[1::2] = 4/3
    w[2:-1:2] = 2/3
    w[-1] = 1/3

    input_vector = np.exp(1j*b*v)*psi*dv*w #final complex input

    fft_output = np.fft.fft(input_vector)

    #extract option prices from complex space
    option_prices = (np.exp(-alpha*k_m)/np.pi)*np.real(fft_output)
    strikes = np.exp(k_m)

    #interpolate to find the option prices, also slice grid for a more applicable range
    valid_range = (strikes > S0 * 0.4) & (strikes < S0 * 2.0)
    return option_prices[valid_range], strikes[valid_range]
