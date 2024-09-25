import numpy as np
from fstpso import FuzzyPSO as ripPSO


def example_fitness(particle):
    return sum(map(lambda x: x**2, particle))


def ackley(x):
    # x = copy.deepcopy(x) - 10
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = sum(np.pow(x, 2))
    sum2 = sum(np.cos(np.multiply(c, x)))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


if __name__ == "__main__":
    dims = 2

    psolist = []
    fuzzilist = []
    for i in range(100):
        fuzzypso = ripPSO()

        fuzzypso.set_search_space([[-10, 10]] * dims)
        fuzzypso.set_fitness(ackley)
        result = fuzzypso.solve_with_fstpso(max_iter=50)

        psolist.append(np.linalg.norm(result[0].X))

    print("caccaPSO")
    print(np.mean(psolist))
    print(np.std(psolist))
