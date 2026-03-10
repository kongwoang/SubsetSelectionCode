from __future__ import annotations

import copy
from random import randint

import numpy as np
from tqdm import tqdm

from ..common.random_ops import mutation
from ..common.solution import position
from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def _h(z, x_f: float, x_c: float) -> float:
    constant = 100000
    if x_c > z["c_value"]:
        return (x_f - z["f_value"]) / (x_c - z["c_value"])
    return (x_f - z["f_value"]) * constant + z["c_value"] - x_c


def _select(problem, populations, fitness_list, refer_points):
    non_empty_populations = []
    aligned_fitness = []

    for index, population in enumerate(populations):
        if np.shape(population)[0] > 0:
            non_empty_populations.append(population)
            aligned_fitness.append(fitness_list[index])

    i = randint(0, len(non_empty_populations) - 1)
    population_i = non_empty_populations[i]

    previous_indexes = [k for k in range(i) if refer_points[k]]
    if not previous_indexes:
        return population_i[randint(0, np.shape(population_i)[0] - 1) :,]

    k = max(previous_indexes)
    z = refer_points[k]

    max_h_value = float("-inf")
    candidates = []

    for index, x in enumerate(population_i):
        current_h = _h(z, aligned_fitness[i][index, 0], aligned_fitness[i][index, 1])
        if current_h > max_h_value:
            max_h_value = current_h
            candidates = [x]
        elif current_h == max_h_value:
            candidates.append(x)

    if "point" in refer_points[i]:
        found = False
        for matrix_item in candidates:
            if np.array_equal(refer_points[i]["point"], matrix_item):
                found = True
        if found:
            s = refer_points[i]["point"]
        else:
            s = candidates[randint(0, len(candidates) - 1)]
    else:
        s = candidates[randint(0, len(candidates) - 1)]

    if np.random.rand() < 0.5:
        x = s
    else:
        x = population_i[randint(0, np.shape(population_i)[0] - 1), :]

    return x


def _local_search(problem, point_info):
    evaluations = 0
    y = copy.deepcopy(point_info["point"])
    y_f = problem.FS(y)
    y_c = problem.CS(y)
    evaluations += 1

    for i in range(problem.n):
        if point_info["point"][0, i] == 0:
            candidate = copy.deepcopy(point_info["point"])
            candidate[0, i] = 1
            candidate_cost = problem.CS(candidate)
            if candidate_cost <= problem.budget:
                candidate_value = problem.FS(candidate)
                evaluations += 1
                if _h(point_info, candidate_value, candidate_cost) >= _h(point_info, y_f, y_c):
                    y = candidate
                    y_f = candidate_value
                    y_c = candidate_cost

    return y, y_f, y_c, evaluations


def run_fpomc(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n

    populations = [[] for _ in range(n + 1)]
    populations[0] = np.mat(np.zeros((1, n), dtype="int8"))

    fitness_list = [[] for _ in range(n + 1)]
    fitness_list[0] = np.mat(np.zeros([1, 2]))

    refer_points = [[] for _ in range(n + 1)]
    refer_points[0] = {"point": np.mat(np.zeros((1, n)), dtype="int8"), "f_value": 0, "c_value": 0}

    iteration = 0
    best_times = 0
    best_value = 0.0

    nn = config.greedy_evaluate
    total_time = config.T * config.greedy_evaluate

    pbar = tqdm(range(total_time), position=0, leave=True, disable=not config.enable_progress_bar)
    current_progress = 0

    while current_progress < total_time:
        if iteration >= nn:
            iteration = 0
            result_index = -1
            subpopulation = -1
            max_value = float("-inf")
            pop_size_sum = 0

            for pop_index in range(n + 1):
                pop_size = np.shape(populations[pop_index])[0]
                pop_size_sum += pop_size
                for p in range(pop_size):
                    if (
                        fitness_list[pop_index][p, 1] <= problem.budget
                        and fitness_list[pop_index][p, 0] > max_value
                    ):
                        max_value = fitness_list[pop_index][p, 0]
                        result_index = p
                        subpopulation = pop_index

            if subpopulation == -1:
                subpopulation = 0
                result_index = 0

            if fitness_list[subpopulation][result_index, 0] == best_value:
                best_times += 1

            if best_times >= config.checkpoint_patience:
                break

            best_value = float(fitness_list[subpopulation][result_index, 0])
            cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)

            if config.logger is not None:
                message = (
                    f" value = {fitness_list[subpopulation][result_index, 0]} cpu_time_used ="
                    f"{round(cpu_time_used, 3)} wall_time_used = {round(wall_time_used, 3)}"
                    f" cost = {fitness_list[subpopulation][result_index, 1]}"
                    f" budget = {problem.budget} population = {pop_size_sum}"
                )
                config.logger.write_main(message, populations[subpopulation][result_index, :])

        x = _select(problem, populations, fitness_list, refer_points)
        x_prime = mutation(n, x)

        offspring_fit = np.mat(np.zeros([1, 2]))
        offspring_fit[0, 1] = problem.CS(x_prime)
        if offspring_fit[0, 1] == 0 or offspring_fit[0, 1] > problem.budget:
            continue

        offspring_fit[0, 0] = problem.FS(x_prime)

        j = int(np.sum(x_prime))
        pop_size_j = np.shape(populations[j])[0]

        has_better = False
        for i in range(pop_size_j):
            if (fitness_list[j][i, 0] > offspring_fit[0, 0] and fitness_list[j][i, 1] <= offspring_fit[0, 1]) or (
                fitness_list[j][i, 0] >= offspring_fit[0, 0] and fitness_list[j][i, 1] < offspring_fit[0, 1]
            ):
                has_better = True
                break

        if not has_better:
            kept = []
            for q in range(pop_size_j):
                if offspring_fit[0, 0] >= fitness_list[j][q, 0] and offspring_fit[0, 1] <= fitness_list[j][q, 1]:
                    continue
                kept.append(q)

            if np.shape(fitness_list[j])[0] == 0:
                fitness_list[j] = offspring_fit
                populations[j] = x_prime
            else:
                fitness_list[j] = np.vstack((offspring_fit, fitness_list[j][kept, :]))
                populations[j] = np.vstack((x_prime, populations[j][kept, :]))

            if not refer_points[j]:
                refer_points[j] = {
                    "point": x_prime,
                    "f_value": float(offspring_fit[0, 0]),
                    "c_value": float(offspring_fit[0, 1]),
                }
            else:
                k = max([k for k in range(j) if refer_points[k]])
                z = refer_points[k]
                value = _h(z, float(offspring_fit[0, 0]), float(offspring_fit[0, 1]))
                if value >= _h(z, refer_points[j]["f_value"], refer_points[j]["c_value"]):
                    _, ff, cc, tt = _local_search(problem, z)
                    iteration += tt
                    current_progress += tt
                    pbar.update(tt)

                    if value >= _h(z, ff, cc):
                        refer_points[j] = {
                            "point": x_prime,
                            "f_value": float(offspring_fit[0, 0]),
                            "c_value": float(offspring_fit[0, 1]),
                        }

                        y, ff, cc, tt = _local_search(problem, refer_points[j])
                        iteration += tt
                        current_progress += tt
                        pbar.update(tt)

                        if j + 1 <= n:
                            next_fit = np.mat(np.zeros([1, 2]))
                            next_fit[0, 1] = cc
                            if next_fit[0, 1] <= problem.budget:
                                next_fit[0, 0] = ff
                                pop_size_next = np.shape(populations[j + 1])[0]

                                has_better = False
                                for i in range(pop_size_next):
                                    if (
                                        fitness_list[j + 1][i, 0] > next_fit[0, 0]
                                        and fitness_list[j + 1][i, 1] <= next_fit[0, 1]
                                    ) or (
                                        fitness_list[j + 1][i, 0] >= next_fit[0, 0]
                                        and fitness_list[j + 1][i, 1] < next_fit[0, 1]
                                    ):
                                        has_better = True
                                        break

                                if not has_better:
                                    kept_next = []
                                    for q in range(pop_size_next):
                                        if (
                                            next_fit[0, 0] >= fitness_list[j + 1][q, 0]
                                            and next_fit[0, 1] <= fitness_list[j + 1][q, 1]
                                        ):
                                            continue
                                        kept_next.append(q)

                                    if np.shape(fitness_list[j + 1])[0] == 0:
                                        fitness_list[j + 1] = next_fit
                                        populations[j + 1] = y
                                    else:
                                        fitness_list[j + 1] = np.vstack((next_fit, fitness_list[j + 1][kept_next, :]))
                                        populations[j + 1] = np.vstack((y, populations[j + 1][kept_next, :]))

        iteration += 1
        current_progress += 1
        pbar.update(1)

    result_index = -1
    subpopulation = -1
    max_value = float("-inf")
    pop_size_sum = 0

    for pop_index in range(n + 1):
        pop_size = np.shape(populations[pop_index])[0]
        pop_size_sum += pop_size
        for p in range(pop_size):
            if fitness_list[pop_index][p, 1] <= problem.budget and fitness_list[pop_index][p, 0] > max_value:
                max_value = fitness_list[pop_index][p, 0]
                result_index = p
                subpopulation = pop_index

    if subpopulation == -1:
        subpopulation = 0
        result_index = 0

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)

    algorithm_result = AlgorithmResult(
        solution=populations[subpopulation][result_index, :],
        value=float(fitness_list[subpopulation][result_index, 0]),
        cost=float(fitness_list[subpopulation][result_index, 1]),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        population=pop_size_sum,
    )

    if config.logger is not None:
        message = (
            f" value = {algorithm_result.value} cpu_time_used = {round(cpu_time_used, 3)}"
            f" wall_time_used = {round(wall_time_used, 3)} cost = {algorithm_result.cost}"
            f" budget = {problem.budget} population = {pop_size_sum}"
        )
        config.logger.write_main(message, algorithm_result.solution)

    pbar.close()
    return algorithm_result
