from __future__ import annotations

import numpy as np

from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def _best_s_add_item(ground_set, candidate_solution, problem, f_value):
    n = problem.n
    budget = problem.budget
    max_volume = -1.0
    max_f = -1.0
    selected_index = -1

    for j in range(n):
        if ground_set[j] == 1:
            candidate_solution[0, j] = 1
            c_value = problem.CS(candidate_solution)
            if c_value > budget:
                candidate_solution[0, j] = 0
                ground_set[j] = 0
                continue

            next_f = problem.FS(candidate_solution)
            temp_volume = next_f - f_value
            if temp_volume > max_volume:
                max_volume = temp_volume
                selected_index = j
                max_f = next_f
            candidate_solution[0, j] = 0

    if selected_index != -1:
        candidate_solution[0, selected_index] = 1

    return candidate_solution, max_f


def run_greedy_max(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget
    ground_set = [1] * n

    result = np.mat(np.zeros((1, n)), "int8")
    best_solution = np.mat(np.zeros((1, n)), "int8")

    f_best = -1.0
    max_f = -1.0

    while sum(ground_set) > 0:
        f = problem.FS(result)
        c = problem.CS(result)

        candidate_solution, candidate_value = _best_s_add_item(ground_set, result.copy(), problem, f)
        if candidate_value > f_best:
            best_solution = candidate_solution
            f_best = candidate_value

        max_volume = -1.0
        selected_index = -1

        for j in range(n):
            if ground_set[j] == 1:
                result[0, j] = 1
                cv = problem.CS(result)
                if cv > budget:
                    result[0, j] = 0
                    ground_set[j] = 0
                    continue

                fv = problem.FS(result)
                temp_volume = 1.0 * (fv - f) / (cv - c)
                if temp_volume > max_volume:
                    max_volume = temp_volume
                    selected_index = j
                    max_f = fv
                result[0, j] = 0

        if selected_index != -1:
            result[0, selected_index] = 1
            ground_set[selected_index] = 0

    if max_f > f_best:
        best_solution = result
        f_best = max_f

    final_cost = problem.CS(best_solution)
    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)

    algorithm_result = AlgorithmResult(
        solution=best_solution,
        value=float(f_best),
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
