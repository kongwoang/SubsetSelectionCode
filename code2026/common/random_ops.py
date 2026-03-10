from __future__ import annotations

import numpy as np


def mutation(n: int, solution: np.matrix) -> np.matrix:
    rand_rate = 1.0 / n
    change = np.random.binomial(1, rand_rate, n)
    return np.abs(solution - change)
