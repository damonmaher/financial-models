#!/usr/bin/env python3
import numpy as np

def forward_vec(obs, A, B, pi):
    #Vectorized forward algorithm
    T = len(obs)
    N = A.shape[0]
    alpha = np.zeros((T,N))
    c=np.zeros(T) #scaling factors

    #Initialization
    alpha[0] = pi * B[:, obs[0]]
    c[0] = 1.0 / (np.sum(alpha[0]) + 1e-300)
    alpha[0] *= c[0]

    #Induction step
    for t in range(1, T):
        #vectorized matrix multiplication
        alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        c[t] = 1.0 / (np.sum(alpha[t]) + 1e-300)
        alpha[t] *= c[t]

    return alpha, c

def backward_vec(obs, A, B, c):
    #Vectorized backward algorithm
    T = len(obs)
    N = A.shape[0]
    beta = np.zeros((T,N))

    #Initialization
    beta[T-1]=c[T-1]

    #Induction
    for t in range(T-2, -1, -1):
        beta[t] = (A @ (B[:, obs[t+1]] * beta[t+1])) * c[t]

    return beta

def baum_welch_vec(obs, A, B, pi, n_iterations=5, inertia=0.85):
    #Vectorized baum-welch with laplace smoothing
    T = len(obs)
    N = A.shape[0]
    M = B.shape[1]

    A_opt = A.copy()
    B_opt = B.copy()
    pi_opt = pi.copy()

    pseudocount = 1e-4

    for iteration in range(n_iterations):
        alpha, c = forward_vec(obs, A_opt, B_opt, pi_opt)
        beta = backward_vec(obs, A_opt, B_opt, c)

        #Vectorized Gamma Calculation
        gamma = (alpha * beta) / c[:, np.newaxis]
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-300)

        #Vectorized Xi Calculation
        xi = np.zeros((T-1,N,N))
        for t in range(T-1):
            xi_t = (alpha[t][:, np.newaxis] * A_opt) * (B_opt[:, obs[t+1]] * beta[t+1])
            xi[t] = xi_t / (np.sum(xi_t) + 1e-300)

        #MAP blending for initial probabilities
        pi_mle = gamma[0] + pseudocount
        pi_mle = pi_mle / np.sum(pi_mle)
        pi_opt = (inertia * pi) + ((1-inertia) * pi_mle)

        #MAP blending for transition
        gamma_sum_T1 = np.sum(gamma[:-1], axis=0)
        A_mle = np.sum(xi, axis=0) + pseudocount
        A_mle = A_mle / np.sum(A_mle, axis=1, keepdims=True)
        A_opt = (inertia * A) + ((1-inertia) * A_mle)

        #Map blending for emission matrix
        gamma_sum_all = np.sum(gamma, axis=0)
        B_mle = np.zeros_like(B_opt)
        for k in range(M):
            mask = (obs == k)
            if np.any(mask):
                B_mle[:, k] = np.sum(gamma[mask], axis=0)

        B_mle += pseudocount
        B_mle = B_mle / np.sum(B_mle, axis=1, keepdims=True)
        B_opt = (inertia * B) + ((1-inertia) * B_mle)

    return A_opt, B_opt, pi_opt
