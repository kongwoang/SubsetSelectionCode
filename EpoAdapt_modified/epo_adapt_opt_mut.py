import numpy as np
import math
from random import randint,random
import random
from tqdm import tqdm
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Set, List, Optional

#*-------------------------------------------------------------------------------------------------------------------------
def Position(s):
    a = np.array(s)
    row = a.ravel()
    return np.where(row == 1)[0]

#*-------------------------------------------------------------------------------------------------------------------------
def save_result(res_file, times, tempmax1, cost, budget, result, cpu_time_used, wall_time_used):
    log = open(res_file+'/result_'+str(times)+'.txt', 'a')
    log.write("value = "+str(tempmax1) + " cpu_time_used =" +str(round(cpu_time_used,3)) + " wall_time_used =" +str(round(wall_time_used,3)) + " cost = "+str(cost)+" budget = "+str(budget))
    log.write("\n")
    for item in Position(result):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
#*-------------------------------------------------------------------------------------------------------------------------


#*-------------------------------------------------------------------------------------------------------------------------
def solution_plus_singleItem(solution,index):
    s=solution.copy() 
    s[0,index]=1
    return s
#*-------------------------------------------------------------------------------------------------------------------------

def population_diversity(population: np.ndarray,
                         cov_sets: Dict[int, Set[int]],
                         universe_size: int) -> float:
    """
    Fraction of total universe covered by at least one solution in 'population'.
    """
    m, _ = population.shape
    if m <= 1:
        return 1.0

    all_cov = set()
    for sol in population:
        # grab the indices where sol has a 1
        ones = np.where(sol.ravel() == 1)[0]
        for j in ones:
            all_cov |= cov_sets[j]
    # cov_frac in [0,1]
    return len(all_cov) / universe_size

#*-------------------------------------------------------------------------------------------------------------------------

def adaptive_mutation(
    n: int,
    s: np.ndarray,
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
) -> np.ndarray:

    base_rate = alpha / n

    # proposal in [0, +infty), then only cap above
    p = base_rate * (1.0 - diversity) * (1.0 + beta * (max_gen - generation) / max_gen)
    p = min(p, max_rate)

    # deletion always has EPOL-style floor
    p_del = max(p, base_rate)

    # addition starts from proposal
    p_add = p
    q_slack = None

    if (costs is not None) and (budget is not None):
        x = s.ravel()
        slack = float(budget - np.dot(x, costs))

        if slack <= 0.0:
            p_add = base_rate if enforce_add_floor else 0.0
        else:
            sum_zero_costs = float(np.sum(costs[x == 0]))
            if sum_zero_costs <= 0.0:
                p_add = 0.0
            else:
                q_slack = (kappa * slack) / sum_zero_costs
                p_add = min(p_add, q_slack, max_rate)

                # enforce floor only when it does not contradict slack cap
                if enforce_add_floor and q_slack >= base_rate:
                    p_add = max(p_add, base_rate)

    # burst with feasibility logic
    if burst_prob > 0.0 and random.random() < burst_prob:
        p_del = min(p_del * burst_mult, max_rate)

        if p_add > 0.0:
            p_add = min(p_add * burst_mult, max_rate)

            # if slack cap exists, enforce it again after burst
            if q_slack is not None:
                p_add = min(p_add, q_slack)

                if enforce_add_floor and q_slack >= base_rate:
                    p_add = max(p_add, base_rate)

    p_del = max(min(p_del, max_rate), base_rate)

    u = np.random.rand(n)
    flips = np.zeros(n, dtype=np.int8)

    bits = s.ravel()
    ones = (bits == 1)
    zeros = ~ones

    flips[ones]  = (u[ones]  < p_del).astype(np.int8)
    flips[zeros] = (u[zeros] < p_add).astype(np.int8)

    return np.abs(s - flips)

#*-------------------------------------------------------------------------------------------------------------------------
def enforce_seed(off: np.ndarray, idx: int) -> np.ndarray:
    """Make sure the fixed seed bit stays zero."""
    if off.ndim == 1:
        off = off.reshape(1, -1)
    off[0, idx] = 0
    return off

#*-------------------------------------------------------------------------------------------------------------------------

#*-------------------------------------------------------------------------------------------------------------------------
def process_subPOMC_gr(
    res_file, times, T, GreedyEvaluate,
    problem, index, value, cost,
    cov_sets, universe_size
):
    
    n=problem.n
    budget=problem.budget

    if cost > budget:
        return None

    def new_func_f(x):
        return problem.FS(solution_plus_singleItem(x, index)) - value

    def new_func_c(x):
        return problem.CS(solution_plus_singleItem(x, index)) - cost
    
    solution, solution_value, solution_cost = sub_POMC_gr(res_file, times, T, GreedyEvaluate, n, index,
                                                       budget - cost, new_func_f, new_func_c, value, cost,
                                                       cov_sets, universe_size, problem)
    #*------------------------------------------------

    solution = np.array(solution)
    # if it's 1D, promote to 2D
    if solution.ndim == 1:
        solution = solution.reshape(1, -1)
    # now safe
    solution[0, index] = 1

    #*-----------------------------------------------
    f_value = solution_value + value
    c_cost = solution_cost + cost

    return index, solution, f_value, c_cost
 
#*-------------------------------------------------------------------------------------------------------------------------

#─────────────────────────────────────────────────────────────────────────────

def penalized_greedy_diverse(
    cov_sets: Dict[int, Set[int]],
    f_scores: Dict[int, float],
    K: int,
    lambda_penalty: float
) -> List[int]:
    """
    Select K seeds by maximizing gain penalized by overlap.
    """
    selected: List[int] = []

    # Keep a mutable remaining set 
    remaining = set(f_scores.keys())

    overlap_sum: Dict[int, int] = {i: 0 for i in remaining}

    cov_local = cov_sets
    f_local = f_scores

    for _ in range(K):
        if not remaining:
            break

        best_item = None
        best_score = -np.inf

        # scores are exact: gain - lambda_penalty * overlap_sum[i]
        for i in remaining:
            score = f_local[i] - lambda_penalty * overlap_sum[i]
            if score > best_score:
                best_score = score
                best_item = i

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)
        overlap_sum.pop(best_item, None)

        # Incrementally update overlap_sum for all remaining candidates.
        cov_best = cov_local.get(best_item, set())
        if cov_best:
            for i in remaining:
                cov_i = cov_local.get(i, set())
                if cov_i:
                    overlap_sum[i] += len(cov_i & cov_best)

    return selected

#*─────────────────────────────────────────────────────────────────────────────

def submodular_K_B(problem,
                   f_scores: Dict[int, float],
                   cov_sets: Dict[int, Set[int]],
                   GreedyEvaluate: int,
                   sample_size: int = 20,
                   min_marginal: float = 0.05) -> int:
    """
    Estimate K_B from the average distinctiveness of singleton seeds.
    """
    n = problem.n
    keys = list(f_scores.keys())
    if len(keys) < 2:
        return max(1, GreedyEvaluate // n)

    # 1) Random sample of pairs
    sample = random.sample(keys, min(sample_size, len(keys)))
    ratios = []
    for i in sample:
        for j in sample:
            if i >= j:
                continue
            Ci, Cj = cov_sets[i], cov_sets[j]
            uni = len(Ci | Cj)
            if uni > 0:
                ratios.append(1.0 - len(Ci & Cj)/uni)

    # 2) Robust average (use median to mitigate outliers)
    if not ratios:
        avg_rho = 0.5
    else:
        avg_rho = float(np.median(ratios))
    avg_rho = max(avg_rho, min_marginal)  # never let it go to zero

    # 3) Compute theoretical K_B
    base_K = max(1, GreedyEvaluate // n)
    theoretical_K = int(base_K / avg_rho)

    # 4) Cap by compute budget
    comp_cap = max(5, GreedyEvaluate // (2 * n))
    return min(theoretical_K, comp_cap, n)

#*─────────────────────────────────────────────────────────────────────────────

def build_fast_coverage_sets(problem):
    """
    Build approximate coverage sets using network topology instead of
    expensive FS evaluations. Based on 2-hop reachability.
    """
    n = problem.n
    weightMatrix = problem.weightMatrix
    cov_sets = {}
    
    print("Building fast coverage sets...")
    
    for i in range(n):
        # Start with the node itself
        cover = {i}
        
        # Add direct neighbors (1-hop)
        direct_neighbors = np.where(weightMatrix[i, :] > 0)[1]
        cover.update(direct_neighbors)
        
        # Add neighbors of neighbors (2-hop) with probability weighting
        for neighbor in direct_neighbors:
            second_hop = np.where(weightMatrix[neighbor, :] > 0)[1]
            # Only add if connection strength is reasonable
            for second in second_hop:
                if (weightMatrix[i, neighbor] * weightMatrix[neighbor, second] > 0.01):
                    cover.add(second)
        
        cov_sets[i] = cover
    
    # Compute universe size
    universe = set().union(*cov_sets.values()) if cov_sets else set()
    universe_size = len(universe)
    
    print(f"Coverage sets built: avg size = {np.mean([len(s) for s in cov_sets.values()]):.1f}")
    return cov_sets, universe_size

#*─────────────────────────────────────────────────────────────────────────────
def build_rr_sets(problem, theta=1000):
    """
    Build θ reverse reachable sets for Independent Cascade.
    Much more efficient than Monte Carlo for influence estimation.
    """
    n = problem.n
    weightMatrix = problem.weightMatrix
    
    print(f"Building {theta} RR sets for influence estimation...")
    
    RR = []
    for _ in tqdm(range(theta), desc="RR sets"):
        # Start from random node
        v = np.random.randint(n)
        R = {v}
        queue = [v]
        
        while queue:
            u = queue.pop()
            # Consider all potential in-neighbors of u
            for w in range(n):
                if w != u and w not in R:
                    # Check if edge w->u exists and activates
                    if hasattr(weightMatrix, 'A'):
                        prob = weightMatrix[w, u]
                    else:
                        prob = weightMatrix[w, u]
                    
                    if prob > 0 and np.random.rand() < prob:
                        R.add(w)
                        queue.append(w)
        
        RR.append(R)
        
    
    print(f"Built {len(RR)} RR sets")
    return RR

#*─────────────────────────────────────────────────────────────────────────────
def estimate_influence_rr(RR, n):
    """
    Estimate influence for all nodes using RR sets.
    Returns dict mapping node → estimated influence.
    """
    theta = len(RR)
    counts = np.zeros(n, dtype=int)
    
    # Count how many RR sets each node appears in
    for R in RR:
        for node in R:
            counts[node] += 1
    
    # Convert to influence estimates
    f_scores = {}
    for i in range(n):
        f_scores[i] = (n / theta) * counts[i]
    
    return f_scores

#*─────────────────────────────────────────────────────────────────────────────
def get_f_scores_via_rr_sets(problem, budget, theta=1000, max_candidates=20):
    """
    Get f_scores using RR sets method - much faster than Monte Carlo.
    Only return scores for feasible and promising candidates.
    """
    print("Computing f_scores via RR sets method...")
    
    # Build RR sets once
    RR = build_rr_sets(problem, theta)
    
    # Get influence estimates for all nodes
    all_f_scores = estimate_influence_rr(RR, problem.n)
    
    # Filter to feasible nodes and rank by promise
    candidates = []
    for i in range(problem.n):
        v = unit(i, problem.n)
        c = problem.CS(v)
        if c <= budget:
            estimated_f = all_f_scores[i]
            cost_efficiency = estimated_f / c if c > 0 else estimated_f
            candidates.append((cost_efficiency, i, estimated_f, c))
    
    # Sort by cost-effectiveness and take top candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    top_candidates = candidates[:max_candidates]
    
    # Return f_scores only for selected candidates
    f_scores = {}
    for _, i, estimated_f, c in top_candidates:
        f_scores[i] = estimated_f
        print(f"  Node {i}: estimated_f={estimated_f:.2f}, c={c:.2f}, ratio={estimated_f/c:.3f}")
    
    print(f"Selected {len(f_scores)} candidates via RR sets")
    return f_scores
#*─────────────────────────────────────────────────────────────────────────────
def choose_theta(n, k, eps=0.1, delta=None, C=2.0):
    """
    Compute a practical RIS size theta based on theoretical guarantees.
      n     : number of nodes
      k     : target seed set size
      eps   : desired epsilon (e.g. 0.1 for 90% of optimum)
      delta : failure probability (default 1/n)
      C     : tuning constant (2 or 3)
    """
    if delta is None:
        delta = 1.0 / n
    
    # Approximate ln(n choose k)
    ln_nCk = k * math.log(n * math.e / max(k, 1))
    term = ln_nCk + math.log(2.0 / delta)
    theta = C * n * term / (eps**2)
    
    return int(math.ceil(theta))

#*─────────────────────────────────────────────────────────────────────────────
# --- CPU quota aware ---
def effective_cpus() -> int:
    """
    Returns the number of CPUs available to *this process*.
    Respects Linux cpuset/cgroup affinity limits (common in containers).
    """
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

#*─────────────────────────────────────────────────────────────────────────────
def choose_max_procs(num_tasks: int,
                     cap: Optional[int] = None,
                     physical_core_hint: Optional[int] = None) -> int:
    """
    Heuristic for max worker processes.

    - Never exceed number of tasks.
    - Never exceed CPUs available to the container.
    - Optionally clamp to a 'physical cores' hint (e.g. 64 on 128-thread EPYC).
    - Optionally clamp to a user cap.
    """
    cpus = effective_cpus()
    w = min(num_tasks, cpus)

    if physical_core_hint is not None:
        w = min(w, physical_core_hint)

    if cap is not None:
        w = min(w, cap)

    return max(1, w)

#*─────────────────────────────────────────────────────────────────────────────
def EPOAdapt(res_file, times, T, GreedyEvaluate, problem):
    start_wall = time.time()
    start_cpu  = time.process_time()

    n      = problem.n
    budget = problem.budget
    lambda_penalty      = 1.0              # tunable overlap penalty

    # 1) build singleton scores and coverage sets
    f_scores: Dict[int, float] = {}
    cov_sets:  Dict[int, Set[int]] = {}
    
    # Influence Maximization 
    if hasattr(problem, "weightMatrix"):
        
        cov_sets, universe_size = build_fast_coverage_sets(problem=problem)
    
        # 1) compute the cheapest singleton cost
        costs = [problem.CS(unit(i,n)) for i in range(n)]
        c_min = min(costs)
        # 2) upper‐bound on how many seeds you could pack
        k_max = max(1, min(problem.n, problem.budget // c_min))
        # 3) choose θ for that k_max
        theta = choose_theta(n, k_max, eps=0.5)
        # never let it go to zero or negative
        theta = max(1_000, theta)     
        
        # Get f_scores via RR sets instead of expensive FS calls
        f_scores = get_f_scores_via_rr_sets(problem, budget, theta=theta, max_candidates=20)

        if not f_scores:
            print("No valid f_scores from RR sets")
            return
    # This is the case for Maximum Coverage 
    else:
        for i in range(n):
            v = unit(i, n)
            c = problem.CS(v)
            if c > budget:
                continue
            f_scores[i] = problem.FS(v)
            # build coverage set: all j where adding j increases FS
            cover = {i}
            base = f_scores[i]
            for j in range(n):
                if j == i: 
                    continue
                w = v.copy(); w[0, j] = 1
                if problem.FS(w) - base > 0:
                    cover.add(j)
            cov_sets[i] = cover

        # Compute the universe size once:
        universe = set().union(*cov_sets.values())
        universe_size = len(universe)
    
    

    # adaptive K_B via submodular estimate
    K_B = submodular_K_B(problem,
                         f_scores,
                         cov_sets,
                         GreedyEvaluate)
    print(f"→ adaptive K_B = {K_B} based on avg marginal ≈ {GreedyEvaluate//n/K_B:.2f}")


    # 2) pick K_B diverse seeds
    selected_ids = penalized_greedy_diverse(cov_sets,
                                            f_scores,
                                            K_B,
                                            lambda_penalty=lambda_penalty)

    # 3) build the exact same list_single_f as before, but only for these IDs
    list_single_f = []
    for i in selected_ids:
        v = unit(i, n)
        list_single_f.append((i,
                              f_scores[i],
                              problem.CS(v)))

    print(f"→ GreedyEvaluate = {GreedyEvaluate}, n = {n}")
    print(f"→ K_B = {K_B}; launching {len(list_single_f)} diverse subRuns\n")

    # 4) Choose max_procs properly (task- and quota-aware)
    # If you're on a 64-core / 128-thread EPYC host, a good physical-core hint is 64.
    num_tasks = len(list_single_f)
    max_procs = choose_max_procs(num_tasks, cap=None, physical_core_hint=64)
    print(f"→ Using max_procs = {max_procs} (tasks={num_tasks}, effective_cpus={effective_cpus()})")

    best_solution = np.zeros((1, n), dtype='int8')
    
    f_best        = problem.FS(best_solution)
    c_best        = problem.CS(best_solution)

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        future_to_run = {}
        # Now pass them along into each sub_POMC call:
        for run_idx, (idx, value, cost) in enumerate(list_single_f, start=1):
            fut = executor.submit(
                process_subPOMC_gr,
                res_file, times, T, GreedyEvaluate,
                problem, idx, value, cost,
                cov_sets,        
                universe_size    
            )
            future_to_run[fut] = run_idx

        for fut in as_completed(future_to_run):
            run_idx = future_to_run[fut]
            print(f"\n>>> Subproblem {run_idx}/{len(list_single_f)} completed <<<")
            result = fut.result()
            if result is not None:
                solution_id, solution, f_value, c_cost = result
                if f_value > f_best:
                    f_best, c_best = f_value, c_cost
                    best_solution = solution.copy()

                    end_cpu  = time.process_time()
                    end_wall = time.time()
                    with open(f"{res_file}/result_{times}.txt", "a") as log:
                        log.write(
                            f"value = {f_best}  single_item = {solution_id}"
                            f"  cpu_time_used = {(end_cpu-start_cpu):.3f}"
                            f"  wall_time_used = {(end_wall-start_wall):.3f}"
                            f"  cost = {c_best}  budget = {budget}\n"
                        )
                        for item in Position(best_solution):
                            log.write(f"{item} ")
                        log.write("\n")

#─────────────────────────────────────────────────────────────────────────────
#*------------------------------------------------------------------------------------------------------------------------- 
def unit(j, n):
    """Helper function to create unit vector"""
    u = np.zeros((1, n), dtype='int8')
    u[0, j] = 1
    return u
#*------------------------------------------------------------------------------------------------------------------------- 

def greedy_plus_residual(s0, n, budget, func_f, func_c, exclude_index=None, problem=None):
    """
    Greedy seeds to be added to POMC's pareto set.
    """

    s = s0.copy()
    rem = set(range(n)) - set(np.flatnonzero(s0[0] == 1).tolist())
    if exclude_index is not None:
        rem.discard(exclude_index)

    # Start with the exact current cost.
    current_c = func_c(s)

    while True:
        base_f = func_f(s)
        base_c = current_c

        best_j, best_ratio = None, 0.0

        for j in rem:
            # ---- cost feasibility ----
            if problem is not None:
                dc = problem.cost[j]        # real per-item cost 
                new_c = base_c + dc
                if new_c > budget:
                    continue
            else:
                # fallback: original behavior
                s[0, j] = 1
                new_c = func_c(s)
                s[0, j] = 0
                if new_c > budget:
                    continue
                dc = new_c - base_c

            # ---- objective gain ----
            s[0, j] = 1
            new_f = func_f(s)
            s[0, j] = 0

            df = new_f - base_f
            ratio = float("inf") if dc == 0 else df / dc

            if ratio > best_ratio:
                best_ratio, best_j = ratio, j

        if best_j is None or best_ratio <= 0:
            break

        s[0, best_j] = 1
        rem.remove(best_j)

        # update running cost exactly
        if problem is not None:
            current_c = base_c + problem.cost[best_j]
        else:
            current_c = func_c(s)

    return s, func_f(s)

#*-------------------------------------------------------------------------------------------------------------------------   
  
def sub_POMC_gr(res_file, times, T, GreedyEvaluate, n, index, new_budget, new_func_f, new_func_c, value, cost,
             cov_sets, universe_size, problem): 
    
    start_wall = time.time()
    start_cpu = time.process_time()
     
    budget = new_budget
     
    # 1) Start with zero solution (residual space)
    zero = np.zeros((1, n), dtype='int8')
    population = [zero]
    fitness    = [(0, 0)]

    # 2) Top‑K singletons by raw gain (K=20)
    K = 20
    scores = []
    for i in range(n):
        if i == index: continue
        s = zero.copy(); s[0, i] = 1
        c_i = new_func_c(s)
        if 0 < c_i <= budget:
            f_i = new_func_f(s)
            if f_i > 0:
                scores.append((f_i, i, s))

    scores.sort(reverse=True, key=lambda x: x[0])

    for f_i, i, s in scores[:K]:
        population.append(s.copy())
        fitness.append((f_i, new_func_c(s)))
        # greedy extension from {i}
        s_g, f_g = greedy_plus_residual(s, n, budget, new_func_f, new_func_c, exclude_index=index, problem=problem)
        if f_g > f_i:
            population.append(s_g)
            fitness.append((f_g, new_func_c(s_g)))

    # 3) Inject the pure‐greedy knapsack seed on the empty residual
    V_full = [1]*n
    V_full[index] = 0
    s_full, f_full = greedy_plus_residual(zero, n, budget, new_func_f, new_func_c, exclude_index=index, problem=problem)
    c_full = new_func_c(s_full)
    if 0 < c_full <= budget:
        population.append(s_full)
        fitness.append((f_full, c_full))

    # 4) Now stack into arrays and continue
    population = np.vstack(population)
    fitness    = np.vstack(fitness)
    popSize    = population.shape[0]
    print(f"SubPOMC {index}: Initialized with {popSize} seeds ({K} singletons + greedy)") 

    # Rest of the POMC algorithm 
    iter = 0 
    best_times = 0 
    best_value = 0 
    nn = GreedyEvaluate 
    totalTime = T * GreedyEvaluate
    
    pbar = tqdm(range(totalTime), position=0, leave=True, desc=f"SubPOMC {index}")
    current_progress = 0
    
    while current_progress < totalTime:
        if iter >= nn:
            iter = 0
            resultIndex = -1
            maxValue = float("-inf")
            for p in range(0, popSize):
                if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
                    maxValue = fitness[p, 0]
                    resultIndex = p

            sub_file = res_file + '/times' + str(times)
            
            os.makedirs(sub_file, exist_ok=True)

            if fitness[resultIndex, 0] == best_value:
                best_times += 1
            else:
                best_times = 0

            if best_times >= 15: # was 10
                break
            else:
                best_value = fitness[resultIndex, 0]

            end_cpu = time.process_time()
            end_wall = time.time()
            cpu_time_used = end_cpu - start_cpu
            wall_time_used = end_wall - start_wall

            log = open(sub_file + '/subPOMC_index_' + str(index) + '.txt', 'a')
            log.write("value = " + str(fitness[resultIndex, 0]) + " total_value = " + str(fitness[resultIndex, 0] + value) +
                     " cpu_time_used =" + str(round(cpu_time_used, 3)) + " wall_time_used =" + str(round(wall_time_used, 3)) +
                     " cost = " + str(fitness[resultIndex, 1]) + " total_cost = " + str(fitness[resultIndex, 1] + cost) +
                     " budget = " + str(budget) + " total_budget = " + str(budget + cost) + " population = " + str(popSize))
            log.write("\n")
            for item in Position(population[resultIndex, :]):
                log.write(str(item))
                log.write(" ")
            log.write("\n")
            log.close()

        # Fixed random selection
        if popSize > 0:
            s = population[randint(0, popSize - 1), :]
        else:
            s = population[0, :]
            
        #*----------------------------------------------------------------------
        diversity = population_diversity(population,
                                        cov_sets,
                                        universe_size)

        offSpring = adaptive_mutation(
                        n, s, iter, totalTime, diversity,
                        alpha=3.0, beta=1.0,
                        burst_prob=0.05, burst_mult=5.0, max_rate=0.5,
                        costs=np.asarray(problem.cost, dtype=float),
                        budget=float(budget),
                        kappa=0.5
                    )

        offSpring = enforce_seed(offSpring, index)

        #*-------------------------------------------------------------------
        
        offSpringFit = np.asmatrix(np.zeros([1, 2]))
        offSpringFit[0, 1] = new_func_c(offSpring)
        if offSpringFit[0, 1] == 0 or offSpringFit[0, 1] > budget:
            iter += 1
            current_progress += 1
            pbar.update(1)
            continue
            
        offSpringFit[0, 0] = new_func_f(offSpring)
        
        hasBetter = False
        for i in range(0, popSize):
            if (fitness[i, 0] > offSpringFit[0, 0] and fitness[i, 1] <= offSpringFit[0, 1]) or \
               (fitness[i, 0] >= offSpringFit[0, 0] and fitness[i, 1] < offSpringFit[0, 1]):
                hasBetter = True
                break
                
        if hasBetter == False:
            Q = []
            for j in range(0, popSize):
                if offSpringFit[0, 0] >= fitness[j, 0] and offSpringFit[0, 1] <= fitness[j, 1]:
                    continue
                else:
                    Q.append(j)
            fitness = np.vstack((offSpringFit, fitness[Q, :]))
            population = np.vstack((offSpring, population[Q, :]))
        
        popSize = np.shape(fitness)[0]
        iter += 1
        current_progress += 1
        pbar.update(1)

    # Final result selection
    resultIndex = -1
    maxValue = float("-inf")
    for p in range(0, popSize):
        if fitness[p, 1] <= budget and fitness[p, 0] > maxValue:
            maxValue = fitness[p, 0]
            resultIndex = p

    sub_file = res_file + '/times' + str(times)

    os.makedirs(sub_file, exist_ok=True)


    end_cpu = time.process_time()
    end_wall = time.time()
    cpu_time_used = end_cpu - start_cpu
    wall_time_used = end_wall - start_wall

    log = open(sub_file + '/subPOMC_index_' + str(index) + '.txt', 'a')
    log.write("value = " + str(fitness[resultIndex, 0]) + " total_value = " + str(fitness[resultIndex, 0] + value) +
             " cpu_time_used =" + str(round(cpu_time_used, 3)) + " wall_time_used =" + str(round(wall_time_used, 3)) +
             " cost = " + str(fitness[resultIndex, 1]) + " total_cost = " + str(fitness[resultIndex, 1] + cost) +
             " budget = " + str(budget) + " total_budget = " + str(budget + cost) + " population = " + str(popSize))
    log.write("\n")
    for item in Position(population[resultIndex, :]):
        log.write(str(item))
        log.write(" ")
    log.write("\n")
    log.close()
    pbar.close()

    return population[resultIndex, :], fitness[resultIndex, 0], fitness[resultIndex, 1]

#*-------------------------------------------------------------------------------------------------------------------------------