from __future__ import annotations

import math
import random
from math import ceil, exp
from random import randint

import numpy as np
from tqdm import tqdm

from ..common.random_ops import mutation
from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def _g_plus_greedy(solution: np.matrix, problem):
    n = problem.n
    budget = problem.budget

    result = solution.copy()

    max_volume = -1.0
    corresponding_cost = -1.0
    evaluations = 0
    selected_index = -1

    for j in range(n):
        if result[0, j] == 0:
            result[0, j] = 1
            c_value = problem.CS(result)
            if c_value > budget:
                result[0, j] = 0
                continue

            f_value = problem.FS(result)
            evaluations += 1

            if f_value > max_volume:
                max_volume = float(f_value)
                corresponding_cost = float(c_value)
                selected_index = j
            result[0, j] = 0

    if selected_index != -1:
        result[0, selected_index] = 1

    return evaluations, result, max_volume, corresponding_cost


def run_sto_evo_smc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget

    population_f = np.mat(np.zeros([1, n], "int8"))
    population_g = np.mat(np.zeros([1, n], "int8"))
    population_g_prime = np.mat(np.zeros([1, n], "int8"))

    fitness_f = np.mat(np.zeros([1, 3]))
    fitness_g = np.mat(np.zeros([1, 3]))
    fitness_g_prime = np.mat(np.zeros([1, 3]))

    best_value = 0.0
    best_times = 0

    iteration = 0
    nn = config.greedy_evaluate
    total_time = config.T * config.greedy_evaluate

    pbar = tqdm(range(total_time), position=0, leave=True, disable=not config.enable_progress_bar)
    current_progress = 0

    epsilon = max(config.epsilon, 1e-12)
    prob = config.prob

    ell = 1
    w = 0
    h_threshold = ceil(2 * exp(1) * n * math.log(1 / epsilon))

    while current_progress < total_time:
        if iteration >= nn:
            iteration = 0
            result_index = -1
            max_value = float("-inf")

            fitness = np.vstack((fitness_f, fitness_g))
            fitness = np.vstack((fitness, fitness_g_prime))

            population = np.vstack((population_f, population_g))
            population = np.vstack((population, population_g_prime))
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

        if random.random() < prob:
            ell += 1
            pop_size_g = population_g.shape[0]
            no_solution = True
            for i in range(pop_size_g):
                if population_g[i, :].sum() == w:
                    no_solution = False
                    current = population_g[i, :]
                    break
            if no_solution:
                current = np.mat(np.zeros([1, n], "int8"))

            if ell >= h_threshold:
                ell = 0
                w += 1
        else:
            population = np.vstack((population_f, population_g))
            pop_size = population.shape[0]
            current = population[randint(1, pop_size) - 1, :]

        offspring = mutation(n, current)
        offspring_fit = np.mat(np.zeros([1, 3]))
        offspring_fit[0, 1] = problem.CS(offspring)

        if offspring_fit[0, 1] == 0 or offspring_fit[0, 1] > budget:
            continue

        offspring_fit[0, 0] = problem.FS(offspring)
        index = int(offspring[0, :].sum())
        offspring_fit[0, 2] = 0 if index == 0 else 1.0 * offspring_fit[0, 0] / offspring_fit[0, 1]

        pop_size_f = population_f.shape[0]
        no_solution = True
        for i in range(pop_size_f):
            if population_f[i, :].sum() == index:
                no_solution = False
                if fitness_f[i, 0] < offspring_fit[0, 0]:
                    population_f[i, :] = offspring
                    fitness_f[i, :] = offspring_fit

        if no_solution:
            population_f = np.vstack((population_f, offspring))
            fitness_f = np.vstack((fitness_f, offspring_fit))

        pop_size_g = population_g.shape[0]
        no_solution = True

        for i in range(pop_size_g):
            if population_g[i, :].sum() == index:
                no_solution = False
                if fitness_g[i, 2] < offspring_fit[0, 2]:
                    population_g[i, :] = offspring
                    fitness_g[i, :] = offspring_fit

                    add_eval, solution_g_prime, g_value, g_cost = _g_plus_greedy(population_g[i, :], problem)
                    current_progress += add_eval
                    iteration += add_eval
                    pbar.update(add_eval)

                    pop_size_g_prime = population_g_prime.shape[0]
                    no_solution_g_prime = True

                    for j in range(pop_size_g_prime):
                        if population_g_prime[j, :].sum() == index:
                            no_solution_g_prime = False
                            if fitness_g_prime[j, 0] < g_value:
                                population_g_prime[j, :] = solution_g_prime
                                fitness_g_prime[j, 0] = g_value
                                fitness_g_prime[j, 1] = g_cost

                    if no_solution_g_prime:
                        new_fit = np.mat(np.zeros([1, 3]))
                        new_fit[0, 0] = g_value
                        new_fit[0, 1] = g_cost
                        new_fit[0, 2] = 0.0
                        population_g_prime = np.vstack((population_g_prime, solution_g_prime))
                        fitness_g_prime = np.vstack((fitness_g_prime, new_fit))

        if no_solution:
            population_g = np.vstack((population_g, offspring))
            fitness_g = np.vstack((fitness_g, offspring_fit))

        iteration += 1
        current_progress += 1
        pbar.update(1)

    result_index = -1
    max_value = float("-inf")

    fitness = np.vstack((fitness_f, fitness_g))
    fitness = np.vstack((fitness, fitness_g_prime))

    population = np.vstack((population_f, population_g))
    population = np.vstack((population, population_g_prime))
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
