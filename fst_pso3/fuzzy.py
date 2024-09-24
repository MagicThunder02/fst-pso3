import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
from dataclasses import dataclass
from miniful import FuzzySet, MembershipFunction, FuzzyRule, FuzzyReasoner, IF, THEN
from .pso import (
    PSO,
    Particle,
)  # Assuming the base PSO class is in a file named pso.py in the same directory


@dataclass
class FuzzyThresholds:
    SAME: float = 0.2
    NEAR: float = 0.4
    FAR: float = 0.6


class FuzzyParticle(Particle):
    def __init__(self, dimensions: int, lower_bound: list, upper_bound: list):
        super().__init__(dimensions, lower_bound, upper_bound)
        self.phi = 0.0  # Performance change
        self.delta = 0.0  # Distance from personal best


class FuzzyPSO(PSO):
    def __init__(
        self,
        fitness: Callable[[np.array], float],
        max_iteration: int,
        max_stagnation: int,
        lower_bound: list,
        upper_bound: list,
        num_particles: Optional[int] = None,
    ):
        super().__init__(
            fitness,
            max_iteration,
            max_stagnation,
            lower_bound,
            upper_bound,
            num_particles,
        )
        self.thresholds = FuzzyThresholds()
        self.fuzzy_reasoner = self.setup_fuzzy_reasoner(
            max_delta=np.max(self.max_velocity)
        )
        self.particles = [
            FuzzyParticle(self.dimensions, self.lower_bound, self.upper_bound)
            for _ in range(self.num_particles)
        ]

    def setup_fuzzy_reasoner(self, max_delta: float) -> FuzzyReasoner:
        fuzzy_reasoner = FuzzyReasoner()

        phi_mf = self._create_phi_membership_function()
        delta_mf = self._create_delta_membership_function(max_delta)

        rules = self._generate_rules(phi_mf, delta_mf)
        fuzzy_reasoner.add_rules(rules)

        return fuzzy_reasoner

    def _create_fuzzy_set(self, points: List[List[float]], term: str) -> FuzzySet:
        return FuzzySet(points=points, term=term, high_quality_interpolate=False)

    def _create_phi_membership_function(self) -> MembershipFunction:
        phi_fuzzy_sets = [
            self._create_fuzzy_set([[0, 0], [1.0, 1.0]], "WORSE"),
            self._create_fuzzy_set([[-1.0, 0], [0, 1.0], [1.0, 0]], "SAME"),
            self._create_fuzzy_set([[-1.0, 1.0], [0, 0]], "BETTER"),
        ]
        return MembershipFunction(phi_fuzzy_sets, concept="PHI")

    def _create_delta_membership_function(self, max_delta: float) -> MembershipFunction:
        same_threshold = max_delta * self.thresholds.SAME
        near_threshold = max_delta * self.thresholds.NEAR
        far_threshold = max_delta * self.thresholds.FAR

        delta_fuzzy_sets = [
            self._create_fuzzy_set(
                [[0, 1.0], [same_threshold, 1.0], [near_threshold, 0]], "SAME"
            ),
            self._create_fuzzy_set(
                [[same_threshold, 0], [near_threshold, 1.0], [far_threshold, 0]], "NEAR"
            ),
            self._create_fuzzy_set(
                [[near_threshold, 0], [far_threshold, 1.0], [max_delta, 1.0]], "FAR"
            ),
        ]
        return MembershipFunction(delta_fuzzy_sets, concept="DELTA")

    def _generate_rules(
        self, phi_mf: MembershipFunction, delta_mf: MembershipFunction
    ) -> List[FuzzyRule]:
        outcome_mapping = self._get_outcome_mapping()

        phi_rules = self._create_rules(
            phi_mf, ["WORSE", "SAME", "BETTER"], outcome_mapping
        )
        delta_rules = self._create_rules(
            delta_mf, ["SAME", "NEAR", "FAR"], outcome_mapping, prefix="DELTA_"
        )

        return phi_rules + delta_rules

    def _get_outcome_mapping(self) -> Dict[str, Dict[str, float]]:
        return {
            "WORSE": {
                "INERTIA": 0.1,
                "SOCIAL": 2.0,
                "COGNITIVE": 1.5,
                "MIN_VELOCITY": 0.1,
                "MAX_VELOCITY": 1.0,
            },
            "SAME": {
                "INERTIA": 0.5,
                "SOCIAL": 1.5,
                "COGNITIVE": 1.5,
                "MIN_VELOCITY": 0.05,
                "MAX_VELOCITY": 0.5,
            },
            "BETTER": {
                "INERTIA": 0.9,
                "SOCIAL": 1.0,
                "COGNITIVE": 2.0,
                "MIN_VELOCITY": 0.0,
                "MAX_VELOCITY": 0.5,
            },
            "DELTA_SAME": {
                "INERTIA": 0.1,
                "SOCIAL": 1.5,
                "COGNITIVE": 1.5,
                "MIN_VELOCITY": 0.1,
                "MAX_VELOCITY": 0.3,
            },
            "DELTA_NEAR": {
                "INERTIA": 0.5,
                "SOCIAL": 1.0,
                "COGNITIVE": 1.5,
                "MIN_VELOCITY": 0.05,
                "MAX_VELOCITY": 0.5,
            },
            "DELTA_FAR": {
                "INERTIA": 0.1,
                "SOCIAL": 1.5,
                "COGNITIVE": 1.5,
                "MIN_VELOCITY": 0.1,
                "MAX_VELOCITY": 0.3,
            },
        }

    def _create_rules(
        self,
        mf: MembershipFunction,
        mf_terms: List[str],
        outcome_map: Dict[str, Dict[str, float]],
        prefix: str = "",
    ) -> List[FuzzyRule]:
        rules = []
        for term in mf_terms:
            map_key = f"{prefix}{term}"
            for outcome_name, outcome_value in outcome_map[map_key].items():
                rule = FuzzyRule(
                    IF(mf, term),
                    THEN(outcome_name, outcome_value),
                )
                rules.append(rule)
        return rules

    def update_particle(self, particle: FuzzyParticle) -> FuzzyParticle:
        if self.iteration > 0:
            self.update_fuzzy_parameters(particle)
            self.update_velocity(particle)
            self.update_position(particle)
        self.update_best_position(particle)
        return particle

    def update_fuzzy_parameters(self, particle: FuzzyParticle):
        current_fitness = particle.current_fitness
        particle.phi = (particle.best_fitness - current_fitness) / abs(
            particle.best_fitness
        )
        particle.delta = np.linalg.norm(particle.position - particle.best_position)

        self.fuzzy_reasoner.set_variable("PHI", particle.phi)
        self.fuzzy_reasoner.set_variable("DELTA", particle.delta)
        fuzzy_output = self.fuzzy_reasoner.evaluate_rules()

        particle.inertia = fuzzy_output.get("INERTIA", particle.inertia)
        particle.social_factor = fuzzy_output.get("SOCIAL", particle.social_factor)
        particle.cognitive_factor = fuzzy_output.get(
            "COGNITIVE", particle.cognitive_factor
        )
        particle.min_velocity_multiplier = fuzzy_output.get(
            "MIN_VELOCITY", particle.min_velocity_multiplier
        )
        particle.max_velocity_multiplier = fuzzy_output.get(
            "MAX_VELOCITY", particle.max_velocity_multiplier
        )

    def solve(self) -> tuple[np.array, float]:
        return super().solve()
