from __future__ import annotations

import numpy as np

from .base import BaseSubsetProblem


class InfluenceMaximizationProblem(BaseSubsetProblem):
    def __init__(self, weight_matrix: np.matrix, budget: float, eps_values: list[float]):
        node_num = int(np.shape(weight_matrix)[0])
        super().__init__(n=node_num, budget=budget)
        self.weight_matrix = weight_matrix
        self.solution = []
        self.all_nodes = np.ones((1, self.n))
        self._initialize_costs(eps_values)

    def _initialize_costs(self, eps_values: list[float]) -> None:
        for i in range(self.n):
            out_degree = int((self.weight_matrix[i, :] > 0).sum())
            self.cost[i] = 1.0 + (1 + abs(eps_values[i])) * out_degree

    def _final_active_nodes(self) -> float:
        active_nodes = np.zeros((1, self.n)) + self.solution
        current_active = np.zeros((1, self.n)) + self.solution
        temp_num = int(current_active.sum(axis=1)[0, 0])

        while temp_num > 0:
            inactive_nodes = self.all_nodes - active_nodes
            random_matrix = np.random.rand(temp_num, self.n)
            activated = (
                sum(random_matrix < self.weight_matrix[current_active.nonzero()[-1], :]) > 0
            )
            current_active = np.multiply(inactive_nodes, activated)
            active_nodes = (current_active + active_nodes) > 0
            temp_num = int(current_active.sum(axis=1)[0, 0])

        return float(active_nodes.sum(axis=1)[0, 0])

    def FS(self, solution: np.matrix, real_evaluate: bool = False) -> float:
        simulate_times = 10000 if real_evaluate else 500
        self.solution = solution
        value = 0.0
        for _ in range(simulate_times):
            value += self._final_active_nodes()
        return value / (simulate_times * 1.0)

    def CS(self, solution: np.matrix) -> float:
        total_cost = 0.0
        for item in self.position(solution):
            total_cost += self.cost[item]
        return total_cost
