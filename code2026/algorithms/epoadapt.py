from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import randint
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult
from ..io_utils.result_writer import ResultWriter


def _unit_vector(index: int, n: int) -> np.matrix:
    vector = np.mat(np.zeros((1, n)), "int8")
    vector[0, index] = 1
    return vector


def _population_diversity(
    population: np.matrix,
    cov_sets: Dict[int, Set[int]],
    universe_size: int,
) -> float:
    if universe_size <= 0 or population.shape[0] <= 1:
        return 1.0

    all_covered: Set[int] = set()
    for solution in np.asarray(population):
        ones = np.where(solution.ravel() == 1)[0]
        for idx in ones:
            all_covered |= cov_sets.get(int(idx), set())
    return len(all_covered) / universe_size


def _adaptive_mutation(
    n: int,
    solution: np.matrix,
    generation: int,
    max_gen: int,
    diversity: float,
    alpha: float = 3.0,
    beta: float = 1.0,
    burst_prob: float = 0.05,
    burst_mult: float = 5.0,
    max_rate: float = 0.5,
    costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    kappa: float = 0.5,
    enforce_add_floor: bool = True,
) -> np.matrix:
    base_rate = alpha / n
    proposal = base_rate * (1.0 - diversity) * (1.0 + beta * (max_gen - generation) / max_gen)
    proposal = min(proposal, max_rate)

    p_del = max(proposal, base_rate)
    p_add = proposal
    q_slack = None

    if (costs is not None) and (budget is not None):
        bits = np.asarray(solution).ravel()
        slack = float(budget - np.dot(bits, costs))

        if slack <= 0.0:
            p_add = base_rate if enforce_add_floor else 0.0
        else:
            sum_zero_costs = float(np.sum(costs[bits == 0]))
            if sum_zero_costs <= 0.0:
                p_add = 0.0
            else:
                q_slack = (kappa * slack) / sum_zero_costs
                p_add = min(p_add, q_slack, max_rate)
                if enforce_add_floor and q_slack >= base_rate:
                    p_add = max(p_add, base_rate)

    if burst_prob > 0.0 and random.random() < burst_prob:
        p_del = min(p_del * burst_mult, max_rate)
        if p_add > 0.0:
            p_add = min(p_add * burst_mult, max_rate)
            if q_slack is not None:
                p_add = min(p_add, q_slack)
                if enforce_add_floor and q_slack >= base_rate:
                    p_add = max(p_add, base_rate)

    p_del = max(min(p_del, max_rate), base_rate)

    uniforms = np.random.rand(n)
    flips = np.zeros(n, dtype=np.int8)

    bits = np.asarray(solution).ravel()
    ones = bits == 1
    zeros = ~ones

    flips[ones] = (uniforms[ones] < p_del).astype(np.int8)
    flips[zeros] = (uniforms[zeros] < p_add).astype(np.int8)

    base = np.asarray(solution, dtype=np.int8).reshape(1, -1)
    return np.mat(np.abs(base - flips), dtype="int8")


def _enforce_seed(solution: np.matrix, fixed_index: int) -> np.matrix:
    if solution.ndim == 1:
        solution = solution.reshape(1, -1)
    solution[0, fixed_index] = 0
    return solution


def _penalized_greedy_diverse(
    cov_sets: Dict[int, Set[int]],
    f_scores: Dict[int, float],
    k: int,
    lambda_penalty: float,
) -> List[int]:
    selected: List[int] = []
    remaining = set(f_scores.keys())
    overlap_sum: Dict[int, int] = {i: 0 for i in remaining}

    for _ in range(k):
        if not remaining:
            break

        best_item = None
        best_score = float("-inf")
        for item in remaining:
            score = f_scores[item] - lambda_penalty * overlap_sum[item]
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)
        overlap_sum.pop(best_item, None)

        cov_best = cov_sets.get(best_item, set())
        if cov_best:
            for item in remaining:
                cov_item = cov_sets.get(item, set())
                if cov_item:
                    overlap_sum[item] += len(cov_item & cov_best)

    return selected


def _submodular_k_b(
    problem,
    f_scores: Dict[int, float],
    cov_sets: Dict[int, Set[int]],
    greedy_evaluate: int,
    sample_size: int = 20,
    min_marginal: float = 0.05,
) -> int:
    n = problem.n
    keys = list(f_scores.keys())
    if len(keys) < 2:
        return max(1, greedy_evaluate // n)

    sample = random.sample(keys, min(sample_size, len(keys)))
    ratios = []
    for i in sample:
        for j in sample:
            if i >= j:
                continue
            ci, cj = cov_sets[i], cov_sets[j]
            union_size = len(ci | cj)
            if union_size > 0:
                ratios.append(1.0 - len(ci & cj) / union_size)

    avg_rho = float(np.median(ratios)) if ratios else 0.5
    avg_rho = max(avg_rho, min_marginal)

    base_k = max(1, greedy_evaluate // n)
    theoretical_k = int(base_k / avg_rho)
    comp_cap = max(5, greedy_evaluate // (2 * n))
    return min(theoretical_k, comp_cap, n)


def _get_weight_matrix(problem) -> np.matrix:
    if hasattr(problem, "weight_matrix"):
        return problem.weight_matrix
    if hasattr(problem, "weightMatrix"):
        return problem.weightMatrix
    raise AttributeError("Problem does not provide weight matrix")


def _build_fast_coverage_sets(problem) -> Tuple[Dict[int, Set[int]], int]:
    n = problem.n
    weight_matrix = _get_weight_matrix(problem)
    cov_sets: Dict[int, Set[int]] = {}

    for i in range(n):
        cover = {i}

        row_i = np.asarray(weight_matrix[i, :]).ravel()
        direct_neighbors = np.where(row_i > 0)[0]
        cover.update(int(node) for node in direct_neighbors)

        for neighbor in direct_neighbors:
            row_neighbor = np.asarray(weight_matrix[int(neighbor), :]).ravel()
            second_hop = np.where(row_neighbor > 0)[0]
            for second in second_hop:
                first_prob = float(weight_matrix[i, int(neighbor)])
                second_prob = float(weight_matrix[int(neighbor), int(second)])
                if first_prob * second_prob > 0.01:
                    cover.add(int(second))

        cov_sets[i] = cover

    universe = set().union(*cov_sets.values()) if cov_sets else set()
    return cov_sets, len(universe)


def _build_rr_sets(problem, theta: int) -> List[Set[int]]:
    n = problem.n
    weight_matrix = _get_weight_matrix(problem)
    rr_sets: List[Set[int]] = []

    for _ in range(theta):
        root = np.random.randint(n)
        rr = {int(root)}
        queue = [int(root)]

        while queue:
            u = queue.pop()
            for w in range(n):
                if w == u or w in rr:
                    continue
                prob = float(weight_matrix[w, u])
                if prob > 0 and np.random.rand() < prob:
                    rr.add(w)
                    queue.append(w)

        rr_sets.append(rr)

    return rr_sets


def _estimate_influence_rr(rr_sets: List[Set[int]], n: int) -> Dict[int, float]:
    theta = len(rr_sets)
    counts = np.zeros(n, dtype=int)
    for rr in rr_sets:
        for node in rr:
            counts[node] += 1

    if theta == 0:
        return {i: 0.0 for i in range(n)}
    return {i: (n / theta) * counts[i] for i in range(n)}


def _get_f_scores_via_rr_sets(problem, budget: float, theta: int = 1000, max_candidates: int = 20) -> Dict[int, float]:
    rr_sets = _build_rr_sets(problem, theta)
    all_scores = _estimate_influence_rr(rr_sets, problem.n)

    candidates = []
    for i in range(problem.n):
        singleton = _unit_vector(i, problem.n)
        cost = problem.CS(singleton)
        if cost <= budget:
            estimated = all_scores[i]
            efficiency = estimated / cost if cost > 0 else estimated
            candidates.append((efficiency, i, estimated))

    candidates.sort(reverse=True, key=lambda item: item[0])
    return {idx: estimated for _, idx, estimated in candidates[:max_candidates]}


def _choose_theta(n: int, k: int, eps: float = 0.1, delta: Optional[float] = None, constant: float = 2.0) -> int:
    if delta is None:
        delta = 1.0 / n
    ln_n_choose_k = k * math.log(n * math.e / max(k, 1))
    term = ln_n_choose_k + math.log(2.0 / delta)
    theta = constant * n * term / (eps**2)
    return int(math.ceil(theta))


def _effective_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def _choose_max_procs(num_tasks: int, cap: Optional[int] = None, physical_core_hint: Optional[int] = None) -> int:
    workers = min(num_tasks, _effective_cpus())
    if physical_core_hint is not None:
        workers = min(workers, physical_core_hint)
    if cap is not None:
        workers = min(workers, cap)
    return max(1, workers)


def _greedy_plus_residual(
    initial: np.matrix,
    n: int,
    budget: float,
    func_f: Callable[[np.matrix], float],
    func_c: Callable[[np.matrix], float],
    exclude_index: Optional[int] = None,
    problem=None,
) -> Tuple[np.matrix, float]:
    solution = initial.copy()
    remaining = set(range(n)) - set(np.flatnonzero(np.asarray(initial)[0] == 1).tolist())
    if exclude_index is not None:
        remaining.discard(exclude_index)

    current_cost = float(func_c(solution))

    while True:
        base_value = float(func_f(solution))
        base_cost = current_cost
        best_idx = None
        best_ratio = 0.0

        for idx in remaining:
            if problem is not None:
                delta_cost = float(problem.cost[idx])
                next_cost = base_cost + delta_cost
                if next_cost > budget:
                    continue
            else:
                solution[0, idx] = 1
                next_cost = float(func_c(solution))
                solution[0, idx] = 0
                if next_cost > budget:
                    continue
                delta_cost = next_cost - base_cost

            solution[0, idx] = 1
            next_value = float(func_f(solution))
            solution[0, idx] = 0
            delta_value = next_value - base_value
            ratio = float("inf") if delta_cost == 0 else delta_value / delta_cost

            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx

        if best_idx is None or best_ratio <= 0:
            break

        solution[0, best_idx] = 1
        remaining.remove(best_idx)
        if problem is not None:
            current_cost = base_cost + float(problem.cost[best_idx])
        else:
            current_cost = float(func_c(solution))

    return solution, float(func_f(solution))


def _sub_pomc_gr(
    n: int,
    fixed_index: int,
    budget: float,
    func_f: Callable[[np.matrix], float],
    func_c: Callable[[np.matrix], float],
    base_value: float,
    base_cost: float,
    greedy_evaluate: int,
    iterations: int,
    cov_sets: Dict[int, Set[int]],
    universe_size: int,
    problem,
    result_dir: Optional[str],
    trial_id: int,
    enable_progress_bar: bool,
    top_k: int,
    sub_patience: int,
    mutation_params: Dict[str, float],
) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()
    writer = ResultWriter(result_dir, trial_id) if result_dir else None

    zero = np.mat(np.zeros((1, n)), "int8")
    population = [zero]
    fitness = [(0.0, 0.0)]

    singleton_scores = []
    for i in range(n):
        if i == fixed_index:
            continue
        candidate = zero.copy()
        candidate[0, i] = 1
        candidate_cost = float(func_c(candidate))
        if 0 < candidate_cost <= budget:
            candidate_value = float(func_f(candidate))
            if candidate_value > 0:
                singleton_scores.append((candidate_value, i, candidate.copy(), candidate_cost))

    singleton_scores.sort(reverse=True, key=lambda item: item[0])
    for candidate_value, _, candidate, candidate_cost in singleton_scores[:top_k]:
        population.append(candidate.copy())
        fitness.append((candidate_value, candidate_cost))

        greedy_solution, greedy_value = _greedy_plus_residual(
            candidate,
            n,
            budget,
            func_f,
            func_c,
            exclude_index=fixed_index,
            problem=problem,
        )
        greedy_cost = float(func_c(greedy_solution))
        if greedy_value > candidate_value and 0 < greedy_cost <= budget:
            population.append(greedy_solution)
            fitness.append((greedy_value, greedy_cost))

    full_solution, full_value = _greedy_plus_residual(
        zero,
        n,
        budget,
        func_f,
        func_c,
        exclude_index=fixed_index,
        problem=problem,
    )
    full_cost = float(func_c(full_solution))
    if 0 < full_cost <= budget:
        population.append(full_solution)
        fitness.append((full_value, full_cost))

    population_matrix = np.vstack(population)
    fitness_matrix = np.vstack(fitness)
    pop_size = population_matrix.shape[0]

    iteration = 0
    best_times = 0
    best_value = 0.0
    total_time = iterations * greedy_evaluate

    pbar = tqdm(
        range(total_time),
        position=0,
        leave=True,
        desc=f"SubPOMC {fixed_index}",
        disable=not enable_progress_bar,
    )
    current_progress = 0

    while current_progress < total_time:
        if iteration >= greedy_evaluate:
            iteration = 0
            result_index = -1
            max_value = float("-inf")
            for p in range(pop_size):
                if fitness_matrix[p, 1] <= budget and fitness_matrix[p, 0] > max_value:
                    max_value = float(fitness_matrix[p, 0])
                    result_index = p

            if result_index == -1:
                result_index = 0

            if float(fitness_matrix[result_index, 0]) == best_value:
                best_times += 1
            else:
                best_times = 0

            if best_times >= sub_patience:
                break

            best_value = float(fitness_matrix[result_index, 0])
            cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
            if writer is not None:
                message = (
                    f"value = {fitness_matrix[result_index, 0]} total_value = {fitness_matrix[result_index, 0] + base_value}"
                    f" cpu_time_used = {round(cpu_time_used, 3)} wall_time_used = {round(wall_time_used, 3)}"
                    f" cost = {fitness_matrix[result_index, 1]} total_cost = {fitness_matrix[result_index, 1] + base_cost}"
                    f" budget = {budget} total_budget = {budget + base_cost} population = {pop_size}"
                )
                writer.write_subpomc(fixed_index, message, population_matrix[result_index, :])

        current = population_matrix[randint(0, pop_size - 1), :]
        diversity = _population_diversity(population_matrix, cov_sets, universe_size)
        offspring = _adaptive_mutation(
            n,
            current,
            iteration,
            total_time,
            diversity,
            alpha=float(mutation_params["alpha"]),
            beta=float(mutation_params["beta"]),
            burst_prob=float(mutation_params["burst_prob"]),
            burst_mult=float(mutation_params["burst_mult"]),
            max_rate=float(mutation_params["max_rate"]),
            costs=np.asarray(problem.cost, dtype=float),
            budget=float(budget),
            kappa=float(mutation_params["kappa"]),
        )
        offspring = _enforce_seed(offspring, fixed_index)

        offspring_fit = np.mat(np.zeros([1, 2]))
        offspring_fit[0, 1] = func_c(offspring)
        if offspring_fit[0, 1] == 0 or offspring_fit[0, 1] > budget:
            iteration += 1
            current_progress += 1
            pbar.update(1)
            continue

        offspring_fit[0, 0] = func_f(offspring)
        has_better = False
        for i in range(pop_size):
            if (fitness_matrix[i, 0] > offspring_fit[0, 0] and fitness_matrix[i, 1] <= offspring_fit[0, 1]) or (
                fitness_matrix[i, 0] >= offspring_fit[0, 0] and fitness_matrix[i, 1] < offspring_fit[0, 1]
            ):
                has_better = True
                break

        if not has_better:
            kept = []
            for j in range(pop_size):
                if offspring_fit[0, 0] >= fitness_matrix[j, 0] and offspring_fit[0, 1] <= fitness_matrix[j, 1]:
                    continue
                kept.append(j)

            fitness_matrix = np.vstack((offspring_fit, fitness_matrix[kept, :]))
            population_matrix = np.vstack((offspring, population_matrix[kept, :]))

        pop_size = np.shape(fitness_matrix)[0]
        iteration += 1
        current_progress += 1
        pbar.update(1)

    result_index = -1
    max_value = float("-inf")
    for p in range(pop_size):
        if fitness_matrix[p, 1] <= budget and fitness_matrix[p, 0] > max_value:
            max_value = float(fitness_matrix[p, 0])
            result_index = p

    if result_index == -1:
        result_index = 0

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    if writer is not None:
        message = (
            f"value = {fitness_matrix[result_index, 0]} total_value = {fitness_matrix[result_index, 0] + base_value}"
            f" cpu_time_used = {round(cpu_time_used, 3)} wall_time_used = {round(wall_time_used, 3)}"
            f" cost = {fitness_matrix[result_index, 1]} total_cost = {fitness_matrix[result_index, 1] + base_cost}"
            f" budget = {budget} total_budget = {budget + base_cost} population = {pop_size}"
        )
        writer.write_subpomc(fixed_index, message, population_matrix[result_index, :])

    pbar.close()
    return AlgorithmResult(
        solution=population_matrix[result_index, :],
        value=float(fitness_matrix[result_index, 0]),
        cost=float(fitness_matrix[result_index, 1]),
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        population=pop_size,
    )


def _process_subpomc_gr(
    problem,
    n: int,
    fixed_index: int,
    value: float,
    cost: float,
    budget: float,
    config_payload: dict,
    cov_sets: Dict[int, Set[int]],
    universe_size: int,
):
    if cost > budget:
        return None

    def new_func_f(solution: np.matrix) -> float:
        candidate = solution.copy()
        candidate[0, fixed_index] = 1
        return float(problem.FS(candidate) - value)

    def new_func_c(solution: np.matrix) -> float:
        candidate = solution.copy()
        candidate[0, fixed_index] = 1
        return float(problem.CS(candidate) - cost)

    sub_result = _sub_pomc_gr(
        n=n,
        fixed_index=fixed_index,
        budget=budget - cost,
        func_f=new_func_f,
        func_c=new_func_c,
        base_value=value,
        base_cost=cost,
        greedy_evaluate=config_payload["greedy_evaluate"],
        iterations=config_payload["T"],
        cov_sets=cov_sets,
        universe_size=universe_size,
        problem=problem,
        result_dir=config_payload.get("result_dir"),
        trial_id=config_payload["trial_id"],
        enable_progress_bar=config_payload.get("enable_progress_bar", False),
        top_k=config_payload["top_k"],
        sub_patience=config_payload["sub_patience"],
        mutation_params=config_payload["mutation_params"],
    )

    solution = sub_result.solution.copy()
    solution[0, fixed_index] = 1
    return fixed_index, solution, sub_result.value + value, sub_result.cost + cost


def run_epoadapt(problem, config: AlgorithmConfig) -> AlgorithmResult:
    start_cpu, start_wall = start_timing()

    n = problem.n
    budget = problem.budget
    algo_params = config.algo_params or {}
    lambda_penalty = float(algo_params.get("lambda_penalty", 1.0))
    theta_eps = float(algo_params.get("theta_eps", 0.5))
    theta_min = int(algo_params.get("theta_min", 1000))
    rr_max_candidates = int(algo_params.get("rr_max_candidates", 20))
    top_k = int(algo_params.get("top_k", 20))
    sub_patience = int(algo_params.get("sub_patience", 15))
    mutation_params = {
        "alpha": float(algo_params.get("alpha", 3.0)),
        "beta": float(algo_params.get("beta", 1.0)),
        "burst_prob": float(algo_params.get("burst_prob", 0.05)),
        "burst_mult": float(algo_params.get("burst_mult", 5.0)),
        "max_rate": float(algo_params.get("max_rate", 0.5)),
        "kappa": float(algo_params.get("kappa", 0.5)),
    }

    f_scores: Dict[int, float] = {}
    cov_sets: Dict[int, Set[int]] = {}

    is_influence_problem = hasattr(problem, "weight_matrix") or hasattr(problem, "weightMatrix")
    if is_influence_problem:
        cov_sets, universe_size = _build_fast_coverage_sets(problem)
        singleton_costs = [float(problem.CS(_unit_vector(i, n))) for i in range(n)]
        c_min = min(singleton_costs) if singleton_costs else 1.0
        if c_min <= 0:
            c_min = 1.0
        k_max = max(1, min(n, int(budget // c_min)))
        theta = max(theta_min, _choose_theta(n, k_max, eps=theta_eps))
        f_scores = _get_f_scores_via_rr_sets(problem, budget, theta=theta, max_candidates=rr_max_candidates)
    else:
        for i in range(n):
            singleton = _unit_vector(i, n)
            singleton_cost = float(problem.CS(singleton))
            if singleton_cost > budget:
                continue
            singleton_value = float(problem.FS(singleton))
            f_scores[i] = singleton_value

            cover = {i}
            for j in range(n):
                if j == i:
                    continue
                candidate = singleton.copy()
                candidate[0, j] = 1
                if float(problem.FS(candidate) - singleton_value) > 0:
                    cover.add(j)
            cov_sets[i] = cover

        universe = set().union(*cov_sets.values()) if cov_sets else set()
        universe_size = len(universe)

    best_solution = np.mat(np.zeros((1, n)), "int8")
    f_best = float(problem.FS(best_solution))
    c_best = float(problem.CS(best_solution))
    solution_id = None

    if not f_scores:
        cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
        return AlgorithmResult(
            solution=best_solution,
            value=f_best,
            cost=c_best,
            cpu_time_used=cpu_time_used,
            wall_time_used=wall_time_used,
        )

    k_b = _submodular_k_b(problem, f_scores, cov_sets, config.greedy_evaluate)
    selected_ids = _penalized_greedy_diverse(cov_sets, f_scores, k=k_b, lambda_penalty=lambda_penalty)

    singleton_list = []
    for idx in selected_ids:
        singleton_cost = float(problem.CS(_unit_vector(idx, n)))
        singleton_list.append((idx, float(f_scores[idx]), singleton_cost))

    if not singleton_list:
        ranked = sorted(f_scores.items(), key=lambda item: item[1], reverse=True)
        for idx, score in ranked:
            singleton_cost = float(problem.CS(_unit_vector(idx, n)))
            if singleton_cost <= budget:
                singleton_list.append((idx, float(score), singleton_cost))
                break

    if not singleton_list:
        cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
        return AlgorithmResult(
            solution=best_solution,
            value=f_best,
            cost=c_best,
            cpu_time_used=cpu_time_used,
            wall_time_used=wall_time_used,
            extra={"single_item": solution_id, "k_b": k_b},
        )

    payload = {
        "trial_id": config.trial_id,
        "T": config.T,
        "greedy_evaluate": config.greedy_evaluate,
        "result_dir": config.result_dir,
        "enable_progress_bar": False,
        "top_k": top_k,
        "sub_patience": sub_patience,
        "mutation_params": mutation_params,
    }

    max_workers = _choose_max_procs(len(singleton_list), cap=config.max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_subpomc_gr,
                problem,
                n,
                idx,
                value,
                cost,
                budget,
                payload,
                cov_sets,
                universe_size,
            )
            for idx, value, cost in singleton_list
        ]

        with tqdm(
            total=len(futures),
            desc="epoadapt",
            position=0,
            leave=True,
            disable=not config.enable_progress_bar,
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    pbar.update(1)
                    continue

                current_id, solution, f_value, c_cost = result
                if f_value > f_best:
                    f_best = float(f_value)
                    c_best = float(c_cost)
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
                        config.logger.write_main(message, best_solution)
                pbar.update(1)

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    return AlgorithmResult(
        solution=best_solution,
        value=f_best,
        cost=c_best,
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
        extra={"single_item": solution_id, "k_b": k_b},
    )
