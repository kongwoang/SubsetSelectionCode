from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .subroutines.sub_pomc import run_sub_pomc
from ..common.solution import solution_plus_single_item
from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def _process_subpomc(index: int, value: float, cost: float, problem, config_payload: dict):
    budget = problem.budget
    if cost > budget:
        return None

    def new_func_f(solution: np.matrix) -> float:
        return problem.FS(solution_plus_single_item(solution, index)) - value

    def new_func_c(solution: np.matrix) -> float:
        return problem.CS(solution_plus_single_item(solution, index)) - cost

    sub_config = AlgorithmConfig(
        trial_id=config_payload["trial_id"],
        T=config_payload["T"],
        greedy_evaluate=config_payload["greedy_evaluate"],
        result_dir=config_payload.get("result_dir"),
        enable_progress_bar=config_payload.get("enable_progress_bar", False),
        checkpoint_patience=config_payload["checkpoint_patience"],
    )

    sub_result = run_sub_pomc(
        n=problem.n,
        fixed_index=index,
        budget=budget - cost,
        func_f=new_func_f,
        func_c=new_func_c,
        base_value=value,
        base_cost=cost,
        config=sub_config,
    )

    solution = sub_result.solution
    solution[0, index] = 1
    return index, solution, sub_result.value + value, sub_result.cost + cost


def run_epomc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget

    singleton_scores = []
    zero_solution = np.mat(np.zeros([1, n], "int8"))

    best_solution = np.mat(np.zeros([1, n], "int8"))
    f_best = problem.FS(best_solution)
    c_best = problem.CS(best_solution)
    solution_id = None

    k_b = int(config.greedy_evaluate / problem.n) if problem.n > 0 else 0

    for i in range(n):
        zero_solution[0, i] = 1
        value = problem.FS(zero_solution)
        cost = problem.CS(zero_solution)
        if cost < budget:
            singleton_scores.append((i, value, cost))
        zero_solution[0, i] = 0

    singleton_scores = sorted(singleton_scores, key=lambda item: item[1], reverse=True)
    singleton_scores = singleton_scores[:k_b]

    payload = {
        "trial_id": config.trial_id,
        "T": config.T,
        "greedy_evaluate": config.greedy_evaluate,
        "result_dir": config.result_dir,
        "enable_progress_bar": False,
        "checkpoint_patience": config.checkpoint_patience,
    }

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(_process_subpomc, index, value, cost, problem, payload)
            for index, value, cost in singleton_scores
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            current_id, solution, f_value, c_cost = result
            if f_value > f_best:
                f_best = f_value
                c_best = c_cost
                best_solution = solution
                solution_id = current_id

                cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
                if config.logger is not None:
                    message = (
                        f"value = {f_best} single_item = {solution_id}"
                        f" cpu_time_used = {round(cpu_time_used, 3)}"
                        f" wall_time_used = {round(wall_time_used, 3)}"
                        f" cost = {c_best} budget = {budget}"
                    )
                    config.logger.write_main(message, solution)

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    return AlgorithmResult(
        solution=best_solution,
        value=float(f_best),
        cost=float(c_best),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        extra={"single_item": solution_id},
    )
