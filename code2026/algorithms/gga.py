from __future__ import annotations

import numpy as np

from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def run_gga(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget
    result = np.mat(np.zeros((1, n)), "int8")
    available = [1] * n
    selected_index = 0

    while sum(available) > 0:
        f = problem.FS(result)
        c = problem.CS(result)
        max_volume = -1.0

        for j in range(n):
            if available[j] == 1:
                result[0, j] = 1
                c_value = problem.CS(result)
                if c_value > budget:
                    result[0, j] = 0
                    available[j] = 0
                    continue

                f_value = problem.FS(result)
                temp_volume = 1.0 * (f_value - f) / (c_value - c)
                if temp_volume > max_volume:
                    max_volume = temp_volume
                    selected_index = j
                result[0, j] = 0

        result[0, selected_index] = 1
        available[selected_index] = 0

    singleton_best_value = 0.0
    singleton_solution = np.mat(np.zeros((1, n)), "int8")
    selected_singleton = 0

    for i in range(n):
        if problem.cost[i] <= budget:
            singleton_solution[0, i] = 1
            candidate = problem.FS(singleton_solution)
            if candidate > singleton_best_value:
                singleton_best_value = candidate
                selected_singleton = i
            singleton_solution[0, i] = 0

    singleton_solution[0, selected_singleton] = 1

    final_value = problem.FS(result)
    if final_value < singleton_best_value:
        final_value = singleton_best_value
        result = singleton_solution

    final_cost = problem.CS(result)
    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)

    algorithm_result = AlgorithmResult(
        solution=result,
        value=float(final_value),
        cost=float(final_cost),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
    )

    if config.logger is not None:
        message = (
            f"value = {algorithm_result.value} cpu_time_used ="
            f"{round(cpu_time_used, 3)} wall_time_used = {round(wall_time_used, 3)}"
            f" cost = {algorithm_result.cost} budget = {budget}"
        )
        config.logger.write_main(message, algorithm_result.solution)

    return algorithm_result
