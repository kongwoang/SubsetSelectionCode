from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Sequence

from ..algorithms import get_algorithm_runner, list_algorithms, normalize_algorithm_name
from ..common.types import AlgorithmConfig, AlgorithmResult
from ..io_utils.graph_readers import read_mc_neighbors
from ..io_utils.result_writer import ResultWriter, build_result_dir
from ..problems.max_cover import MaxCoverProblem
from .local_config import MC_ALGO_GRID, MC_DEFAULTS
from .pathing import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, resolve_input_file
from .summary import append_run_summary


def _parse_csv(raw: str, cast):
    values = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def _parse_algorithms(raw: str) -> list[str]:
    cleaned = raw.strip().lower()
    if cleaned in {"all", "*"}:
        return list_algorithms()
    return [normalize_algorithm_name(item) for item in _parse_csv(raw, str)]


def _normalize_max_workers(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "null"}:
            return None
        return int(cleaned)
    return int(value)


def _algo_param_dicts(raw: dict, mode: str = "product") -> list[dict]:
    if not raw:
        return [{}]
    normalized: dict[str, list] = {}
    for key in sorted(raw.keys()):
        value = raw[key]
        if isinstance(value, (list, tuple)):
            normalized[key] = list(value)
        else:
            normalized[key] = [value]

    if mode == "one_at_a_time":
        base = {key: values[0] for key, values in normalized.items()}
        combos = [dict(base)]
        for key, values in normalized.items():
            for value in values[1:]:
                candidate = dict(base)
                candidate[key] = value
                combos.append(candidate)
        return combos

    # default: full Cartesian product
    keys = sorted(normalized.keys())
    value_lists = [normalized[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def _build_algo_grid(
    algorithm: str,
    cli_iterations: str,
    cli_probs: str,
    cli_epsilons: str,
    cli_max_workers: str,
) -> list[tuple[int, float, float, int | None, dict]]:
    fallback = dict(MC_ALGO_GRID.get("*", {}))
    override = dict(MC_ALGO_GRID.get(algorithm, {}))
    merged = {**fallback, **override}

    iterations = _parse_csv(cli_iterations, int) if cli_iterations.strip() else merged.get("iterations", [MC_DEFAULTS["iterations"]])
    probs = _parse_csv(cli_probs, float) if cli_probs.strip() else merged.get("probs", [MC_DEFAULTS["prob"]])
    epsilons = _parse_csv(cli_epsilons, float) if cli_epsilons.strip() else merged.get("epsilons", [MC_DEFAULTS["epsilon"]])
    max_workers = (
        [_normalize_max_workers(v) for v in _parse_csv(cli_max_workers, str)]
        if cli_max_workers.strip()
        else [_normalize_max_workers(v) for v in merged.get("max_workers", [MC_DEFAULTS["max_workers"]])]
    )
    algo_param_dicts = _algo_param_dicts(
        merged.get("algo_params", {}),
        mode=str(merged.get("algo_params_mode", "product")),
    )

    return [
        (int(iteration), float(prob), float(epsilon), _normalize_max_workers(worker), dict(algo_params))
        for iteration, prob, epsilon, worker, algo_params in product(
            iterations,
            probs,
            epsilons,
            max_workers,
            algo_param_dicts,
        )
    ]


def _is_better(candidate: AlgorithmResult, best: AlgorithmResult | None) -> bool:
    if best is None:
        return True
    if candidate.value != best.value:
        return candidate.value > best.value
    return candidate.cost < best.cost


def _run_mc_single(
    *,
    problem: MaxCoverProblem,
    result_root: Path,
    adjacency_file: str,
    q: int,
    n: int,
    budget: float,
    canonical_algo: str,
    iterations: int,
    prob: float,
    epsilon: float,
    trial_id: int,
    max_workers: int | None,
    disable_progress: bool,
    data_dir: Path,
    algo_params: dict,
) -> AlgorithmResult:
    runner = get_algorithm_runner(canonical_algo)
    summary_params = {
        "algorithm": canonical_algo,
        "adjacency_file": adjacency_file,
        "q": q,
        "n": n,
        "budget": budget,
        "iterations": iterations,
        "prob": prob,
        "epsilon": epsilon,
        "max_workers": max_workers,
        "algo_params": algo_params,
        "data_dir": str(data_dir),
        "result_root": str(result_root),
    }

    k_b = problem.max_subset_size()
    greedy_evaluate = problem.n * k_b
    result_dir = build_result_dir(
        result_root=result_root,
        problem_name="mc",
        dataset_file=adjacency_file,
        algorithm_name=canonical_algo,
        budget=budget,
        q=q,
    )

    writer = ResultWriter(result_dir=result_dir, trial_id=trial_id)
    config = AlgorithmConfig(
        trial_id=trial_id,
        T=iterations,
        greedy_evaluate=greedy_evaluate,
        epsilon=epsilon,
        prob=prob,
        logger=writer,
        result_dir=result_dir,
        max_workers=max_workers,
        enable_progress_bar=not disable_progress,
        algo_params=algo_params,
    )

    try:
        result = runner(problem, config)
    except Exception as exc:
        append_run_summary(
            result_root=result_root,
            problem="mc",
            status="error",
            trial_id=trial_id,
            params=summary_params,
            error=str(exc),
        )
        raise

    append_run_summary(
        result_root=result_root,
        problem="mc",
        status="ok",
        trial_id=trial_id,
        params=summary_params,
        result=result,
    )
    return result


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run Maximum Coverage experiments",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("--data-dir", type=Path, default=MC_DEFAULTS.get("data_dir", DEFAULT_DATA_DIR))
    parser.add_argument("--result-root", type=Path, default=MC_DEFAULTS.get("result_root", DEFAULT_RESULTS_DIR))
    parser.add_argument("--adjacency-file", type=str, default=MC_DEFAULTS["adjacency_file"])
    parser.add_argument("--q", type=int, default=MC_DEFAULTS["q"])
    parser.add_argument("--n", type=int, default=MC_DEFAULTS["n"])
    parser.add_argument("--budget", type=float, default=MC_DEFAULTS["budget"])
    parser.add_argument(
        "--iterations",
        type=int,
        default=MC_DEFAULTS["iterations"],
        help="T multiplier used by evolutionary algorithms",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=MC_DEFAULTS["algorithm"],
        help=f"Algorithm name. Supported canonical names: {', '.join(list_algorithms())}",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=MC_DEFAULTS["prob"],
        help="stochastic branch probability in sto_evo_smc",
    )
    parser.add_argument("--epsilon", type=float, default=MC_DEFAULTS["epsilon"], help="epsilon parameter in sto_evo_smc")
    parser.add_argument("--trial-id", type=int, default=MC_DEFAULTS["trial_id"])
    parser.add_argument("--max-workers", type=int, default=MC_DEFAULTS["max_workers"])
    parser.add_argument("--disable-progress", action="store_true", default=MC_DEFAULTS["disable_progress"])
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction, default=MC_DEFAULTS["grid_search"])
    parser.add_argument("--stop-on-error", action="store_true", default=MC_DEFAULTS["stop_on_error"])
    parser.add_argument(
        "--grid-iterations",
        type=str,
        default="",
        help="Comma-separated iterations for grid mode (empty -> per-algorithm defaults in local_config)",
    )
    parser.add_argument(
        "--grid-probs",
        type=str,
        default="",
        help="Comma-separated prob values for grid mode (empty -> per-algorithm defaults in local_config)",
    )
    parser.add_argument(
        "--grid-epsilons",
        type=str,
        default="",
        help="Comma-separated epsilon values for grid mode (empty -> per-algorithm defaults in local_config)",
    )
    parser.add_argument(
        "--grid-max-workers",
        type=str,
        default="",
        help="Comma-separated max_workers values; use 'none' for default behavior",
    )
    parser.add_argument(
        "--algo-params-json",
        type=str,
        default="",
        help="JSON object for algorithm-specific params in single-run mode",
    )
    return parser


def run_mc(args: argparse.Namespace) -> AlgorithmResult:
    data_dir = Path(args.data_dir)
    result_root = Path(args.result_root)

    adjacency_path = resolve_input_file(args.adjacency_file, data_dir)

    data = read_mc_neighbors(str(adjacency_path), args.n)
    problem = MaxCoverProblem(data=data, budget=args.budget, n=args.n, q=args.q)

    if not args.grid_search:
        canonical_algo = normalize_algorithm_name(args.algorithm)
        algo_params = {}
        if args.algo_params_json.strip():
            loaded = json.loads(args.algo_params_json)
            if not isinstance(loaded, dict):
                raise ValueError("--algo-params-json must be a JSON object")
            algo_params = loaded
        return _run_mc_single(
            problem=problem,
            result_root=result_root,
            adjacency_file=adjacency_path.name,
            q=args.q,
            n=args.n,
            budget=args.budget,
            canonical_algo=canonical_algo,
            iterations=args.iterations,
            prob=args.prob,
            epsilon=args.epsilon,
            trial_id=args.trial_id,
            max_workers=args.max_workers,
            disable_progress=args.disable_progress,
            data_dir=data_dir,
            algo_params=algo_params,
        )

    algorithms = _parse_algorithms(args.algorithm)
    jobs: list[tuple[int, float, float, int | None, dict, str]] = []
    for algo in algorithms:
        for iteration, prob, epsilon, workers, algo_params in _build_algo_grid(
            algo,
            args.grid_iterations,
            args.grid_probs,
            args.grid_epsilons,
            args.grid_max_workers,
        ):
            jobs.append((iteration, prob, epsilon, workers, algo_params, algo))

    total = len(jobs)
    print(f"[MC-GRID] planned={total} algorithms={len(algorithms)}")
    best_result: AlgorithmResult | None = None

    for idx, (iteration, prob, epsilon, workers, algo_params, algo) in enumerate(jobs, start=1):
        trial_id = args.trial_id + idx - 1
        started = time.perf_counter()
        label = (
            f"trial={trial_id} algo={algo} iter={iteration} prob={prob} eps={epsilon} "
            f"workers={workers} algo_params={algo_params}"
        )
        print(f"[MC-GRID][{idx}/{total}] START {label}")
        try:
            result = _run_mc_single(
                problem=problem,
                result_root=result_root,
                adjacency_file=adjacency_path.name,
                q=args.q,
                n=args.n,
                budget=args.budget,
                canonical_algo=algo,
                iterations=iteration,
                prob=prob,
                epsilon=epsilon,
                trial_id=trial_id,
                max_workers=args.max_workers if args.max_workers is not None else workers,
                disable_progress=args.disable_progress,
                data_dir=data_dir,
                algo_params=algo_params,
            )
            elapsed = time.perf_counter() - started
            print(f"[MC-GRID][{idx}/{total}] OK elapsed={elapsed:.2f}s value={result.value} cost={result.cost}")
            if _is_better(result, best_result):
                best_result = result
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            print(f"[MC-GRID][{idx}/{total}] ERROR elapsed={elapsed:.2f}s {label}: {exc}")
            if args.stop_on_error:
                raise

    if best_result is None:
        raise RuntimeError("All MC grid jobs failed")
    return best_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_mc(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
