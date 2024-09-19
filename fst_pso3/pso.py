import copy
import logging
import sys
import concurrent.futures
from typing import Callable, Optional

import numpy as np


class Particle(object):
    def __init__(self, dimentions: int, lower_bound: list, upper_bound: list):
        assert len(lower_bound) == len(upper_bound)
        assert dimentions == len(lower_bound)

        self.position = np.random.uniform(lower_bound, upper_bound, dimentions)
        self.velocity = np.zeros(dimentions)
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = sys.float_info.max

        self.step_magnitude = 0.0
        self.distance_from_best = sys.float_info.max

        self.cognitive_factor = 2.0
        self.social_factor = 2.0
        self.inertia = 0.5
        self.max_velocity_multiplier = 0.25
        self.min_velocity_multiplier = 0.0


class PSO:
    def __init__(
        self,
        fitness: Callable[[np.array], float],
        num_particles: int,
        max_iteration: int,
        max_stagnation: int,
        lower_bound: list,
        upper_bound: list,
    ):
        assert len(lower_bound) == len(upper_bound)
        self.dimentions = len(lower_bound)

        self.fitness = fitness
        self.iteration = 0
        self.max_iteration = max_iteration
        self.since_last_improvement = 0
        self.max_stagnation = max_stagnation

        self.global_best_position: Optional[np.array] = None
        self.global_best_fitness = sys.float_info.max

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.particles = [
            Particle(self.dimentions, self.lower_bound, self.upper_bound)
            for _ in range(num_particles)
        ]

        self.max_velocity = np.abs(np.array(upper_bound) - np.array(lower_bound))

    def solve(self) -> tuple[np.array, float]:
        logging.info("Launching parallel optimization.")

        while not self.termination_criterion():
            # with concurrent.futures.ProcessPoolExecutor() as executor:

            #     futures = {
            #         executor.submit(self.update_particle, p): p for p in self.particles
            #     }
            #     updated_particles: list[Particle] = []
            #     for future in concurrent.futures.as_completed(futures):
            #         updated_particles.append(future.result())

            #     self.particles = updated_particles
            updated_particles: list[Particle] = []

            for particle in self.particles:
                updated_particle = self.update_particle(particle)
                updated_particles.append(updated_particle)
            self.particles = updated_particles

            self.update_global_best()
            self.iteration += 1

        return self.global_best_position, self.global_best_fitness

    def update_particle(self, particle: Particle) -> Particle:
        if self.iteration > 0:
            self.update_velocity(particle)
            self.update_position(particle)
        self.update_best_position(particle)
        return particle

    def update_velocity(self, particle: Particle):
        inertia_velocity = particle.inertia * particle.velocity
        cognitive_velocity = (
            particle.cognitive_factor
            * np.random.uniform()
            * (particle.best_position - particle.position)
        )
        social_velocity = (
            particle.social_factor
            * np.random.uniform()
            * (self.global_best_position - particle.position)
        )

        velocity = inertia_velocity + cognitive_velocity + social_velocity

        # Apply velocity limits
        velocity = np.minimum(
            velocity, self.max_velocity * particle.max_velocity_multiplier
        )
        velocity = np.maximum(
            velocity, self.max_velocity * particle.min_velocity_multiplier
        )

        particle.velocity = velocity

    def update_position(self, particle: Particle):
        particle.position += particle.velocity

        # Apply position limits
        particle.position = np.minimum(particle.position, self.upper_bound)
        particle.position = np.maximum(particle.position, self.lower_bound)

        particle.step_magnitude = np.linalg.norm(particle.velocity)
        particle.distance_from_best = np.linalg.norm(
            particle.position - particle.best_position
        )

    def update_best_position(self, particle: Particle):
        fitness = self.calculate_fitness(particle)
        if fitness < particle.best_fitness:
            particle.best_position = copy.deepcopy(particle.position)
            particle.best_fitness = fitness

    def update_global_best(self):
        updated = False
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_position = copy.deepcopy(particle.best_position)
                self.global_best_fitness = particle.best_fitness
                updated = True

        if not updated:
            self.since_last_improvement += 1
        else:
            self.since_last_improvement = 0

    def calculate_fitness(self, particle: Particle) -> float:
        return self.fitness(particle.position)

    def termination_criterion(self) -> bool:
        if self.iteration >= self.max_iteration:
            logging.info("Max iterations reached")
            return True
        if self.since_last_improvement >= self.max_stagnation:
            logging.info("Max stagnation reached")
            return True

        return False
