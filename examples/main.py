import logging

import numpy as np
from fst_pso3 import PSO
from fst_pso3.fuzzy import FuzzyPSO


def fitness(x):
    return sum(x**2)


def rastrigin(x):
    # x = copy.deepcopy(x) - 10
    A = 10
    return A * len(x) + sum((x**2 - A * np.cos(2 * np.pi * x)))


def drop_wave(x):
    # x = copy.deepcopy(x) - 10
    numerator = 1 + np.cos(12 * np.sqrt(np.sum(x**2)))
    denominator = 0.5 * np.sum(x**2) + 2
    return -numerator / denominator


def ackley(x):
    # x = copy.deepcopy(x) - 10
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = sum(x**2)
    sum2 = sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    psolist = []
    fuzzilist = []
    for i in range(100):
        pso = PSO(ackley, 50, 20, [-10] * 2, [10] * 2, num_particles=20)
        fuzzypso = FuzzyPSO(ackley, 50, 20, [-10] * 2, [10] * 2, num_particles=20)
        result = pso.solve()
        resultfuzzy = fuzzypso.solve()

        psolist.append(result[1])
        fuzzilist.append(resultfuzzy[1])

    print("PSO")
    print(np.mean(psolist))
    print(np.std(psolist))

    print("FuzzyPSO")
    print(np.mean(fuzzilist))
    print(np.std(fuzzilist))
