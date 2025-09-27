import time
import numpy as np
from scipy.special import gamma


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# Ship Rescue Optimization (SRO)
def SRO(SearchAgents_no, fhd, lb, ub, Max_Iteration):
    N, dim = SearchAgents_no.shape[0], SearchAgents_no.shape[1]
    ship_pos = SearchAgents_no
    ship_v = SearchAgents_no
    C = 2
    xbest = float('inf')
    fbest = np.zeros((dim, 1))
    ship_fitness = fhd(SearchAgents_no[:])
    ship_pos_new = np.copy(ship_pos)
    conv = []

    ct = time.time()
    for iter in range(1, Max_Iteration + 1):
        ship_pos_new = np.clip(ship_pos_new, lb, ub)
        ship_fitness_new = np.array([fhd(ind.T) for ind in ship_pos_new])
        for i in range(SearchAgents_no):
            if ship_fitness_new[i] < ship_fitness[i]:
                ship_pos[i, :] = ship_pos_new[i, :]
                ship_fitness[i] = ship_fitness_new[i]

        fbest = np.min(ship_fitness)
        I = np.argmin(ship_fitness)
        xbest = ship_pos[I, :]
        conv.append(fbest)

        F_wind = 3 * np.random.rand(dim)
        m = np.ones(dim)
        a = F_wind / m
        beta = 3 / 2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

        for i in range(SearchAgents_no):
            ship_v[i, :] += a
            p = np.random.rand()
            u = np.random.randn() * sigma
            v = np.random.randn()
            step = u / abs(v) ** (1 / beta)
            stepsize = 0.01 * step * (ship_pos[i, :] - xbest)
            direction = 1 if p < 0.6 else -1
            ship_pos_new[i, :] = ship_pos[i, :] + ship_v[i, :] + (
                direction if p > 0.3 else 0) * stepsize * np.random.randn(dim)

        groups = 4
        groupsize = SearchAgents_no // groups
        group = []

        for g in range(groups):
            start_idx = g * groupsize
            g_pos = np.copy(ship_pos_new[start_idx:start_idx + groupsize, :])
            g_fitness = np.copy(ship_fitness[start_idx:start_idx + groupsize])
            xbest_idx = np.argmin(g_fitness)
            xworse_idx = np.argmax(g_fitness)
            group.append({
                'pos': g_pos,
                'fitness': g_fitness,
                'xbest': g_pos[xbest_idx, :],
                'fbest': g_fitness[xbest_idx],
                'xworse': g_pos[xworse_idx, :],
                'fworse': g_fitness[xworse_idx]
            })

        for g in range(groups):
            for i in range(groupsize):
                rssi = [1 / dist(group[g]['pos'][i], group[tg]['xbest']) for tg in range(groups)]
                sorted_rssi = np.argsort(rssi)
                dist_ib = 1 / np.array(rssi)
                VT = np.random.rand(dim) * (1 / iter) ** (
                        (group[g]['fitness'][i] - group[g]['fbest']) / (group[g]['fbest'] - group[g]['fworse'] + 1e-9))
                if dist(group[g]['pos'][i], group[g]['xbest']) < (
                        dist_ib[sorted_rssi[0]] + dist_ib[sorted_rssi[1]]) / 2:
                    group[g]['pos'][i] += C * np.random.rand() * (group[g]['xbest'] - group[g]['pos'][i]) + VT
                else:
                    g1 = sorted_rssi[0]
                    group[g]['pos'][i] += C * np.random.rand() * (group[g1]['xbest'] - group[g]['pos'][i]) + VT

        if iter % 20 == 0:
            for g in range(groups):
                sorted_idx = np.argsort(group[g]['fitness'])
                for a in range(2 * groupsize // 3, groupsize):
                    group[g]['pos'][sorted_idx[a], :] = group[(g + 1) % 4]['xbest']

        for g in range(groups):
            start_idx = g * groupsize
            ship_pos_new[start_idx:start_idx + groupsize, :] = group[g]['pos']

    ct = time.time() - ct
    return xbest, conv, fbest, ct
