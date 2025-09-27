import time
import numpy as np


# Remora Optimization Algorithm (ROA)
def ROA(Remora, objective, Lowerbound, Upperbound, Max_iterations):
    N, dimensions = Remora.shape[0], Remora.shape[1]
    BestRemora = np.zeros(dimensions)
    Score = float('inf')
    Prevgen = [Remora]
    Convergence = np.zeros(Max_iterations)
    t = 0
    ct = time.time()
    while t < Max_iterations:
        # Memory of previous generation
        if t <= 1:
            PreviousRemora = Prevgen[0]
        else:
            PreviousRemora = Prevgen[t - 1]

        # Boundary check
        for i in range(Remora.shape[0]):
            Flag4Upperbound = Remora[i, :] > Upperbound
            Flag4Lowerbound = Remora[i, :] < Lowerbound
            Remora[i, :] = (Remora[i, :] * ~(
                    Flag4Upperbound | Flag4Lowerbound)) + Upperbound * Flag4Upperbound + Lowerbound * Flag4Lowerbound
            fitness = objective(Remora[i, :])

            # Evaluate fitness function of search agents
            if fitness < Score:
                Score = fitness
                BestRemora = Remora[i, :]

        # Make an experience attempt through equation (2)
        for j in range(Remora.shape[0]):
            RemoraAtt = Remora[j, :] + (Remora[j, :] - PreviousRemora[j, :]) * np.random.randn()  # Equation(2)

            # Calculate the fitness function value of the attempted solution (fitnessAtt)
            fitnessAtt = objective(RemoraAtt)

            # Calculate the fitness function value of the current solution (fitnessI)
            fitnessI = objective(Remora[j, :])

            # Check if the current fitness (fitnessI) is better than the attempted fitness(fitnessAtt)
            # if No, Perform host feeding by equation (9)
            if fitnessI > fitnessAtt:
                V = 2 * (1 - t / Max_iterations)  # Equation (12)
                B = 2 * V * np.random.rand() - V  # Equation (11)
                C = 0.1
                A = B * (Remora[j, :] - C * BestRemora)  # Equation (10)
                Remora[j, :] = Remora[j, :] + A  # Equation (9)

            # If yes perform host conversion using equation (1) and (5)
            elif np.random.randint(0, 2) == 0:
                a = -(1 + t / Max_iterations)  # Equation (7)
                alpha = np.random.rand() * (a - 1) + 1  # Equation (6)
                D = np.abs(BestRemora - Remora[j, :])  # Equation (8)
                Remora[j, :] = D * np.exp(alpha) * np.cos(2 * np.pi * a) + Remora[j, :]  # Equation (5)
            else:
                m = np.random.permutation(Remora.shape[0])
                Remora[j, :] = BestRemora - (
                        (np.random.rand() * (BestRemora + Remora[m[0], :]) / 2) - Remora[m[0], :])  # Equation (1)

        t += 1
        Prevgen.append(Remora)
        Convergence[t - 1] = Score
    ct = time.time() - ct
    return Score, Convergence, BestRemora, ct
