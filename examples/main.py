import logging
from fst_pso3 import PSO
from fst_pso3.fuzzy import FuzzyPSO


def fitness(x):
    return sum(x**2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pso = FuzzyPSO(fitness, 400, 20, [-10] * 2, [10] * 2, num_particles=5)
    result = pso.solve()

    print(result)
