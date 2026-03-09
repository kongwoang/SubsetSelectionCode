from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from typing import List

import numpy as np

from EPOL_modified.common.solution import position


class BaseSubsetProblem(ABC):
    def __init__(self, n: int, budget: float):
        self.n = n
        self.budget = budget
        self.cost: List[float] = [0.0] * n
        self.dp = None

    def position(self, solution: np.matrix) -> np.ndarray:
        return position(solution)

    @abstractmethod
    def FS(self, solution: np.matrix, real_evaluate: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def CS(self, solution: np.matrix) -> float:
        raise NotImplementedError

    def max_subset_size(self) -> int:
        self.dp = [[0] * (ceil(self.budget) + 1) for _ in range(self.n + 1)]

        for i in range(1, self.n + 1):
            item_cost = self.cost[i - 1]
            for j in range(ceil(self.budget) + 1):
                self.dp[i][j] = self.dp[i - 1][j]
                if j >= item_cost:
                    prev_index = round(j - item_cost)
                    self.dp[i][j] = max(self.dp[i][j], self.dp[i - 1][prev_index] + 1)

        return self.dp[self.n][ceil(self.budget)]
