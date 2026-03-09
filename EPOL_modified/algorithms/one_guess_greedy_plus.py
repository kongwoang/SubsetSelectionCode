from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from algorithms.subroutines.greedy_plus import greedy_plus
from common.solution import solution_plus_single_item
from common.timing import elapsed_time, start_timing
from common.types import AlgorithmConfig, AlgorithmResult


def _process_item(j: int, problem, budget: float):
    n = problem.n
    zero_solution = np.mat(np.zeros((1, n)), "int8")
    zero_solution[0, j] = 1

    c_single_item = problem.CS(zero_solution)
    if c_single_item > budget:
        zero_solution[0, j] = 0
        return None

    f_single_item = problem.FS(zero_solution)

    def new_func_f(solution: np.matrix) -> float:
        return problem.FS(solution_plus_single_item(solution, j)) - f_single_item

    def new_func_c(solution: np.matrix) -> float:
        return problem.CS(solution_plus_single_item(solution, j)) - c_single_item

    available = [1] * n
    available[j] = 0

    solution, solution_value = greedy_plus(available, budget - c_single_item, new_func_f, new_func_c)
    solution[0, j] = 1
    f_value = solution_value + f_single_item

    zero_solution[0, j] = 0
    return j, f_value, solution


def run_one_guess_greedy_plus(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    budget = problem.budget
    n = problem.n

    best_solution = np.mat(np.zeros((1, n)), "int8")
    f_best = problem.FS(best_solution)

    max_workers = config.max_workers if config.max_workers is not None else 100

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_item, j, problem, budget) for j in range(n)]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            _, f_value, solution = result
            if f_value > f_best:
                f_best = f_value
                best_solution = solution

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
