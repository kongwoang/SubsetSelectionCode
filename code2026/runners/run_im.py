from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ..algorithms import get_algorithm_runner, list_algorithms, normalize_algorithm_name
from ..common.types import AlgorithmConfig, AlgorithmResult
from ..io_utils.graph_readers import read_im_edge_matrix, read_outdegree_eps
from ..io_utils.result_writer import ResultWriter, build_result_dir
from ..problems.influence_maximization import InfluenceMaximizationProblem
from .local_config import IM_DEFAULTS
from .pathing import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, resolve_input_file
from .summary import append_run_summary


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run Influence Maximization experiments",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("--data-dir", type=Path, default=IM_DEFAULTS.get("data_dir", DEFAULT_DATA_DIR))
    parser.add_argument("--result-root", type=Path, default=IM_DEFAULTS.get("result_root", DEFAULT_RESULTS_DIR))
    parser.add_argument("--adjacency-file", type=str, default=IM_DEFAULTS["adjacency_file"])
    parser.add_argument("--outdegree-file", type=str, default=IM_DEFAULTS["outdegree_file"])
    parser.add_argument("--probability", type=float, default=IM_DEFAULTS["probability"])
    parser.add_argument("--budget", type=float, default=IM_DEFAULTS["budget"])
    parser.add_argument(
        "--iterations",
        type=int,
        default=IM_DEFAULTS["iterations"],
        help="T multiplier used by evolutionary algorithms",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=IM_DEFAULTS["algorithm"],
        help=f"Algorithm name. Supported canonical names: {', '.join(list_algorithms())}",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=IM_DEFAULTS["prob"],
        help="stochastic branch probability in sto_evo_smc",
    )
    parser.add_argument("--epsilon", type=float, default=IM_DEFAULTS["epsilon"], help="epsilon parameter in sto_evo_smc")
    parser.add_argument("--trial-id", type=int, default=IM_DEFAULTS["trial_id"])
    parser.add_argument("--max-workers", type=int, default=IM_DEFAULTS["max_workers"])
    parser.add_argument("--disable-progress", action="store_true", default=IM_DEFAULTS["disable_progress"])
    return parser


def run_im(args: argparse.Namespace) -> AlgorithmResult:
    data_dir = Path(args.data_dir)
    result_root = Path(args.result_root)

    adjacency_path = resolve_input_file(args.adjacency_file, data_dir)
    outdegree_path = resolve_input_file(args.outdegree_file, data_dir)

    canonical_algo = normalize_algorithm_name(args.algorithm)
    runner = get_algorithm_runner(canonical_algo)
    summary_params = {
        "algorithm": canonical_algo,
        "adjacency_file": args.adjacency_file,
        "outdegree_file": args.outdegree_file,
        "probability": args.probability,
        "budget": args.budget,
        "iterations": args.iterations,
        "prob": args.prob,
        "epsilon": args.epsilon,
        "data_dir": str(data_dir),
        "result_root": str(result_root),
    }

    weight_matrix = read_im_edge_matrix(args.probability, str(adjacency_path))
    node_num = int(np.shape(weight_matrix)[0])
    eps_values = read_outdegree_eps(str(outdegree_path), node_num)

    problem = InfluenceMaximizationProblem(weight_matrix=weight_matrix, budget=args.budget, eps_values=eps_values)
    k_b = problem.max_subset_size()
    greedy_evaluate = problem.n * k_b

    result_dir = build_result_dir(
        result_root=result_root,
        problem_name="im",
        dataset_file=adjacency_path.name,
        algorithm_name=canonical_algo,
        budget=args.budget,
    )

    if args.algorithm.lower() == "evo_smc":
        epsilon = 1e-10
        prob = 0.0
    else:
        epsilon = args.epsilon
        prob = args.prob

    writer = ResultWriter(result_dir=result_dir, trial_id=args.trial_id)
    config = AlgorithmConfig(
        trial_id=args.trial_id,
        T=args.iterations,
        greedy_evaluate=greedy_evaluate,
        epsilon=epsilon,
        prob=prob,
        logger=writer,
        result_dir=result_dir,
        max_workers=args.max_workers,
        enable_progress_bar=not args.disable_progress,
    )

    try:
        result = runner(problem, config)
    except Exception as exc:
        append_run_summary(
            result_root=result_root,
            problem="im",
            status="error",
            trial_id=args.trial_id,
            params=summary_params,
            error=str(exc),
        )
        raise

    append_run_summary(
        result_root=result_root,
        problem="im",
        status="ok",
        trial_id=args.trial_id,
        params=summary_params,
        result=result,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_im(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
