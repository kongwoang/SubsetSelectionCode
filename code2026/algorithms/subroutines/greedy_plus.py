from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def greedy_plus(
    ground_set: List[int],
    budget: float,
    func_f: Callable[[np.matrix], float],
    func_c: Callable[[np.matrix], float],
) -> Tuple[np.matrix, float]:
    available = ground_set.copy()
    n = len(available)

    result = np.mat(np.zeros((1, n)), "int8")
    best_solution = np.mat(np.zeros((1, n)), "int8")
    f_best = -1.0
    max_f = -1.0

    while sum(available) > 0:
        f = func_f(result)
        c = func_c(result)

        candidate = result.copy()
        max_volume = -1.0
        value_1 = -1.0
        selected_index = -1

        for j in range(n):
            if available[j] == 1:
                candidate[0, j] = 1
                c_value = func_c(candidate)
                if c_value > budget:
                    candidate[0, j] = 0
                    available[j] = 0
                    continue
                f_value = func_f(candidate)
                temp_volume = f_value - f
                if temp_volume > max_volume:
                    max_volume = temp_volume
                    selected_index = j
                    value_1 = f_value
                candidate[0, j] = 0

        if selected_index != -1:
            candidate[0, selected_index] = 1
            if value_1 > f_best:
                best_solution = candidate
                f_best = value_1

        max_volume = -1.0
        selected_index = -1

        for j in range(n):
            if available[j] == 1:
                result[0, j] = 1
                c_value = func_c(result)
                if c_value > budget:
                    result[0, j] = 0
                    available[j] = 0
                    continue
                f_value = func_f(result)
                temp_volume = 1.0 * (f_value - f) / (c_value - c)
                if temp_volume > max_volume:
                    max_volume = temp_volume
                    selected_index = j
                    max_f = f_value
                result[0, j] = 0

        if selected_index != -1:
            result[0, selected_index] = 1
            available[selected_index] = 0

    if max_f > f_best:
        best_solution = result
        f_best = max_f

    return best_solution, f_best
