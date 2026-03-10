from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .subroutines.sub_pomc import run_sub_pomc
from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def _process_sub_p_pomc(problem, n: int, fixed_index: int, budget: float, greedy_eval: int, config_payload: dict):
    sub_config = AlgorithmConfig(
        trial_id=config_payload["trial_id"],
        T=config_payload["T"],
        greedy_evaluate=greedy_eval,
        result_dir=config_payload.get("result_dir"),
        enable_progress_bar=config_payload.get("enable_progress_bar", False),
        checkpoint_patience=config_payload["checkpoint_patience"],
    )

    sub_result = run_sub_pomc(
        n=n,
        fixed_index=fixed_index,
        budget=budget,
        func_f=problem.FS,
        func_c=problem.CS,
        base_value=0,
        base_cost=0,
        config=sub_config,
    )
    return fixed_index, sub_result.solution, sub_result.value, sub_result.cost


def run_p_pomc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget

    singleton_scores = []
    zero_solution = np.mat(np.zeros([1, n], "int8"))

    best_solution = np.mat(np.zeros([1, n], "int8"))
    f_best = 0.0
    c_best = 0.0
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
    if k_b > len(singleton_scores):
        k_b = len(singleton_scores)
    singleton_scores = singleton_scores[:k_b]

    if k_b == 0:
        cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
        return AlgorithmResult(
            solution=best_solution,
            value=f_best,
            cost=c_best,
            cpu_time_used=cpu_time_used,
            wall_time_used=wall_time_used,
            extra={"id": solution_id},
        )

    greedy_evaluates = []
    for index in range(k_b):
        cost = singleton_scores[index][2]
        new_k_b = problem.dp[n][int(budget - cost)]
        greedy_evaluates.append(n * new_k_b)

    payload = {
        "trial_id": config.trial_id,
        "T": config.T,
        "result_dir": config.result_dir,
        "enable_progress_bar": False,
        "checkpoint_patience": config.checkpoint_patience,
    }

    max_workers = config.max_workers if config.max_workers is not None else k_b

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_sub_p_pomc,
                problem,
                n,
                index,
                budget,
                greedy_evaluates[index],
                payload,
            )
            for index in range(k_b)
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
                        f"value = {f_best} id = {solution_id}"
                        f" cpu_time_used = {round(cpu_time_used, 3)}"
                        f" wall_time_used = {round(wall_time_used, 3)}"
                        f" cost = {c_best} budget = {budget}"
                    )
                    config.logger.write_main(message, best_solution)

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    return AlgorithmResult(
        solution=best_solution,
        value=float(f_best),
        cost=float(c_best),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        extra={"id": solution_id},
    )
