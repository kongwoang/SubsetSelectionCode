from __future__ import annotations

import numpy as np

from .base import BaseSubsetProblem


class MaxCoverProblem(BaseSubsetProblem):
    def __init__(self, data: list[list[int]], budget: float, n: int, q: int):
        super().__init__(n=n, budget=budget)
        self.data = data
        self.q = q
        self._initialize_costs()

    def _initialize_costs(self) -> None:
        for i in range(self.n):
            temp_elements = [i]
            temp_elements.extend(self.data[i])
            temp_value = len(list(set(temp_elements))) - self.q
            if temp_value > 0:
                self.cost[i] = temp_value + 1
            else:
                self.cost[i] = 1

    def FS(self, solution: np.matrix, real_evaluate: bool = False) -> float:
        del real_evaluate
        positions = self.position(solution)
        covered_set: list[int] = []
        for node in positions:
            covered_set.extend(self.data[node])
        covered_set.extend(positions)
        return float(len(set(covered_set)))

    def CS(self, solution: np.matrix) -> float:
        total_cost = 0.0
        for item in self.position(solution):
            total_cost += self.cost[item]
        return total_cost
