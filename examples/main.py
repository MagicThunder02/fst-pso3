import logging
from fst_pso3 import PSO


def fitness(x):
    return sum(x**2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pso = PSO(fitness, 10, 100, 20, [-10] * 2, [10] * 2)
    result = pso.solve()

    print(result)
