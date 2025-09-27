import time

import numpy as np


def GBOA(X, objective, lb, ub, max_iter):
    pop_size, dim = X.shape[0], X.shape[1]


    def ensure_bounds(vec):
        lower = np.array([b for b in lb])
        upper = np.array([b for b in ub])
        return np.clip(vec, lower, upper)

    def levy_flight(beta=1.5, size=None):
        if size is None:
            size = (1,)
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / (np.abs(v) ** (1.0 / beta))
        return 0.01 * step

    def spiral_region(Xi, sperm_j, b=0.2):
        t = np.random.uniform(-1.0, 1.0)
        Wi = Xi - sperm_j
        return Wi * np.exp(b * t) * np.cos(2 * np.pi * t) + sperm_j

    def compute_Hs(it, max_iter, Hs_max=1.5, Hs_min=0.2):
        return Hs_max - (Hs_max - Hs_min) * (it / float(max_iter))

    # extension lengths l (random [0,1], but not used explicitly in update here)
    l = np.random.rand(pop_size, dim)

    # sperm regions
    S = np.random.rand(pop_size, dim)
    for d in range(dim):
        S[:, d] = lb + S[:, d] * (lb - ub)

    Delta = np.zeros_like(X)
    fitness = np.array([objective(X[i]) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    GlobalBest = X[best_idx].copy()
    GlobalBest_score = fitness[best_idx]

    history = []
    ct = time.time()
    # --- main loop ---
    for it in range(max_iter):
        Hs = compute_Hs(it, max_iter)
        S = 0.5 * (S + GlobalBest)  # sperm regions drift toward best

        for i in range(pop_size):
            j = np.random.randint(0, pop_size)
            Xi = X[i].copy()
            sperm_j = S[j]

            # wind direction
            if dim == 2:
                angle_rad = np.deg2rad(np.random.randint(0, 360))
                WD_i = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * 0.01
            else:
                WD_i = np.random.normal(0, 1, size=dim) * 0.01

            # target dimension (move toward best)
            T_dim = (GlobalBest - Xi) * np.random.rand(dim)

            # spiral sampling
            spiral_pt = spiral_region(Xi, sperm_j, b=0.2)

            # movement update
            Delta_i_next = WD_i + T_dim + Hs * Delta[i]
            Xi_next = Xi + Delta_i_next

            if np.random.rand() > 0.2:
                Xi_next = Xi_next + 0.1 * (spiral_pt - Xi_next)
            else:
                Xi_next = Xi_next + levy_flight(size=dim) * Xi_next

            Xi_next = ensure_bounds(Xi_next)

            f_new = objective(Xi_next)
            if f_new < fitness[i]:
                X[i] = Xi_next
                fitness[i] = f_new
                Delta[i] = Delta_i_next

        # update global best
        curr_best_idx = np.argmin(fitness)
        if fitness[curr_best_idx] < GlobalBest_score:
            GlobalBest_score = fitness[curr_best_idx]
            GlobalBest = X[curr_best_idx].copy()

        history.append(GlobalBest_score)

    ct = time.time() - ct
    return GlobalBest_score, history, GlobalBest, ct
