from __future__ import annotations

import numpy as np


def position(solution: np.matrix) -> np.ndarray:
    return np.array(np.where(solution[0, :] == 1)[1]).reshape(-1)


def solution_plus_single_item(solution: np.matrix, index: int) -> np.matrix:
    next_solution = solution.copy()
    next_solution[0, index] = 1
    return next_solution
