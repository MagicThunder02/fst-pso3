import logging
import sys
import concurrent.futures
from typing import Optional

import numpy as np


class Particle(object):
    def __init__(self, dimentions: int, lower_bound: list, upper_bound: list):
        self.position = np.random.uniform(dimentions, lower_bound, upper_bound)
        self.velocity = np.zeros(dimentions)
        self.best_position = np.copy.deepcopy(self.position)
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
        num_particles: int,
        max_iteration: int,
        max_stagnation: int,
        lower_bound: list,
        upper_bound: list,
    ):
        self.iteration = 0
        self.max_iteration = max_iteration
        self.since_last_improvement = 0
        self.max_stagnation = max_stagnation

        self.particles = [Particle() for _ in range(num_particles)]
        self.global_best_position: Optional[np.array] = None
        self.global_best_fitness = sys.float_info.max

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.max_velocity = np.abs(np.array(upper_bound) - np.array(lower_bound))

    def update_particle(self, particle: Particle):
        self.update_velocity(particle)
        self.update_position(particle)
        self.update_best_position(particle)
        return particle

    def solve(self):
        logging.info("Launching parallel optimization.")

        while not self.termination_criterion():
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.update_particle, p): p for p in self.particles
                }
                updated_particles: list[Particle] = []
                for future in concurrent.futures.as_completed(futures):
                    updated_particles.append(future.result())

                self.particles = updated_particles
            # Update global best and worst positions
            self.update_global_best()

        logging.info("Best solution: " + str(self.global_best_position))
        logging.info(
            "Fitness of best solution: "
            + str(self.calculate_fitness(self.global_best_position))
        )

    def update_particle(self, particle: Particle):
        self.update_velocity(particle)
        self.update_position(particle)
        self.update_best_position(particle)

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
        fitness = self.calculate_fitness(particle.position)
        if fitness < particle.best_fitness:
            particle.best_position = np.copy.deepcopy(particle.position)

    def update_global_best(self):
        updated = False
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = np.copy.deepcopy(particle.best_position)
                updated = True

        if not updated:
            self.since_last_improvement += 1

    def calculate_fitness(self, position):
        # Implement fitness calculation
        pass

    def termination_criterion(self):
        if self.iteration >= self.max_iteration:
            return True
        if self.since_last_improvement >= self.max_stagnation:
            return True

        return False


# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pso = PSO(num_particles=100)
    pso.Solve()
