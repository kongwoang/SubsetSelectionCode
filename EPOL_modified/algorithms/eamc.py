from __future__ import annotations

from functools import lru_cache
from math import exp
from random import randint

import numpy as np
from tqdm import tqdm

from EPOL_modified.common.random_ops import mutation
from EPOL_modified.common.timing import elapsed_time, start_timing
from EPOL_modified.common.types import AlgorithmConfig, AlgorithmResult


@lru_cache(maxsize=None)
def _gs(alpha: float, fitness: float, cost: float, budget: float) -> float:
    if fitness > 0:
        return 1.0 * fitness / (1.0 - (1.0 / exp(alpha * cost / budget)))
    return 0.0


def run_eamc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget

    population_u = np.mat(np.zeros([1, n], "int8"))
    population_v = np.mat(np.zeros([1, n], "int8"))

    fitness_u = np.mat(np.zeros([1, 3]))
    fitness_v = np.mat(np.zeros([1, 3]))

    best_value = 0.0
    best_times = 0

    iteration = 0
    nn = config.greedy_evaluate
    total_time = config.T * config.greedy_evaluate

    pbar = tqdm(range(total_time), position=0, leave=True, disable=not config.enable_progress_bar)
    current_progress = 0

    while current_progress < total_time:
        if iteration >= nn:
            iteration = 0
            result_index = -1
            max_value = float("-inf")

            fitness = np.vstack((fitness_v, fitness_u))
            population = np.vstack((population_v, population_u))
            pop_size = population.shape[0]

            for p in range(pop_size):
                if fitness[p, 1] <= budget and fitness[p, 0] > max_value:
                    max_value = fitness[p, 0]
                    result_index = p

            if result_index == -1:
                result_index = 0

            if fitness[result_index, 0] == best_value:
                best_times += 1
            else:
                best_times = 0

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

        population = np.vstack((population_u, population_v))
        pop_size = population.shape[0]
        current = population[randint(1, pop_size) - 1, :]

        offspring = mutation(n, current)
        offspring_fit = np.mat(np.zeros([1, 3]))
        offspring_fit[0, 1] = problem.CS(offspring)

        if offspring_fit[0, 1] == 0 or offspring_fit[0, 1] > budget:
            continue

        offspring_fit[0, 0] = problem.FS(offspring)
        offspring_fit[0, 2] = _gs(1.0, float(offspring_fit[0, 0]), float(offspring_fit[0, 1]), budget)
        index = int(offspring[0, :].sum())

        pop_size_v = population_v.shape[0]
        no_solution = True
        for i in range(pop_size_v):
            if population_v[i, :].sum() == index:
                no_solution = False
                if fitness_v[i, 0] < offspring_fit[0, 0]:
                    population_v[i, :] = offspring
                    fitness_v[i, :] = offspring_fit
                break

        if no_solution:
            population_v = np.vstack((population_v, offspring))
            fitness_v = np.vstack((fitness_v, offspring_fit))

        pop_size_u = population_u.shape[0]
        no_solution = True
        for i in range(pop_size_u):
            if population_u[i, :].sum() == index:
                no_solution = False
                if fitness_u[i, 2] < offspring_fit[0, 2]:
                    population_u[i, :] = offspring
                    fitness_u[i, :] = offspring_fit
                break

        if no_solution:
            population_u = np.vstack((population_u, offspring))
            fitness_u = np.vstack((fitness_u, offspring_fit))

        iteration += 1
        current_progress += 1
        pbar.update(1)

    result_index = -1
    max_value = float("-inf")

    fitness = np.vstack((fitness_v, fitness_u))
    population = np.vstack((population_v, population_u))
    pop_size = population.shape[0]

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
