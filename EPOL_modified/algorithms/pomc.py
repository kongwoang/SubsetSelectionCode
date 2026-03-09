from __future__ import annotations

from random import randint

import numpy as np
from tqdm import tqdm

from EPOL_modified.common.random_ops import mutation
from EPOL_modified.common.timing import elapsed_time, start_timing
from EPOL_modified.common.types import AlgorithmConfig, AlgorithmResult


def run_pomc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget

    population = np.mat(np.zeros([1, n], "int8"))
    fitness = np.mat(np.zeros([1, 2]))
    pop_size = 1

    iteration = 0
    nn = config.greedy_evaluate
    total_time = config.T * config.greedy_evaluate

    best_value = 0.0
    best_times = 0

    pbar = tqdm(range(total_time), position=0, leave=True, disable=not config.enable_progress_bar)
    current_progress = 0

    while current_progress < total_time:
        if iteration >= nn:
            iteration = 0
            result_index = -1
            max_value = float("-inf")
            for p in range(pop_size):
                if fitness[p, 1] <= budget and fitness[p, 0] > max_value:
                    max_value = fitness[p, 0]
                    result_index = p

            if result_index == -1:
                result_index = 0

            best_times = best_times + 1 if fitness[result_index, 0] == best_value else 0
            if best_times >= config.checkpoint_patience:
                break

            best_value = float(fitness[result_index, 0])
            cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)

            if config.logger is not None:
                message = (
                    f"value = {fitness[result_index, 0]} cpu_time_used ="
                    f"{round(cpu_time_used, 3)} wall_time_used = {round(wall_time_used, 3)}"
                    f" cost = {fitness[result_index, 1]} budget = {budget} population = {pop_size}"
                )
                config.logger.write_main(message, population[result_index, :])

        current = population[randint(1, pop_size) - 1, :]
        offspring = mutation(n, current)

        offspring_fit = np.mat(np.zeros([1, 2]))
        offspring_fit[0, 1] = problem.CS(offspring)
        if offspring_fit[0, 1] == 0 or offspring_fit[0, 1] > budget:
            continue

        offspring_fit[0, 0] = problem.FS(offspring)
        has_better = False

        for i in range(pop_size):
            if (fitness[i, 0] > offspring_fit[0, 0] and fitness[i, 1] <= offspring_fit[0, 1]) or (
                fitness[i, 0] >= offspring_fit[0, 0] and fitness[i, 1] < offspring_fit[0, 1]
            ):
                has_better = True
                break

        if not has_better:
            kept = []
            for j in range(pop_size):
                if offspring_fit[0, 0] >= fitness[j, 0] and offspring_fit[0, 1] <= fitness[j, 1]:
                    continue
                kept.append(j)

            fitness = np.vstack((offspring_fit, fitness[kept, :]))
            population = np.vstack((offspring, population[kept, :]))

        pop_size = np.shape(fitness)[0]
        iteration += 1
        current_progress += 1
        pbar.update(1)

    result_index = -1
    max_value = float("-inf")
    for p in range(pop_size):
        if fitness[p, 1] <= budget and fitness[p, 0] > max_value:
            max_value = fitness[p, 0]
            result_index = p

    if result_index == -1:
        result_index = 0

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    algorithm_result = AlgorithmResult(
        solution=population[result_index, :],
        value=float(fitness[result_index, 0]),
        cost=float(fitness[result_index, 1]),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        population=pop_size,
    )

    if config.logger is not None:
        message = (
            f"value = {algorithm_result.value} cpu_time_used = {round(cpu_time_used, 3)}"
            f" wall_time_used = {round(wall_time_used, 3)} cost = {algorithm_result.cost}"
            f" budget = {budget} population = {pop_size}"
        )
        config.logger.write_main(message, algorithm_result.solution)

    pbar.close()
    return algorithm_result
