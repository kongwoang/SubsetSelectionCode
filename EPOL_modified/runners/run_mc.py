from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from algorithms import get_algorithm_runner
from common.types import AlgorithmConfig
from io_utils.graph_readers import read_mc_neighbors
from io_utils.result_writer import ResultWriter, build_result_dir_for_mc
from problems.max_cover import MaxCoverProblem


def _resolve_data_path(filename: str) -> str:
    local = Path("outdegree") / filename
    if local.exists():
        return str(local)

    project_local = Path(__file__).resolve().parents[1] / "outdegree" / filename
    if project_local.exists():
        return str(project_local)

    legacy = Path(__file__).resolve().parents[2] / "EPOL" / "outdegree" / filename
    return str(legacy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-adjacency_file", type=str, default="congress.edgelist-new.txt")
    parser.add_argument("-q", type=int, default=5)
    parser.add_argument("-n", type=int, default=475)
    parser.add_argument("-T", type=int, default=20)
    parser.add_argument("-budget", type=float, default=500)
    parser.add_argument("-times", type=int, default=3)
    parser.add_argument("-algo", type=str, default="PPOMC")
    parser.add_argument("-prob", type=float, default=0.5, help="sto_evo_smc_p")
    parser.add_argument("-epsilon", type=float, default=0.1, help="sto_evo_smc_epsilon")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    adjacency_file = _resolve_data_path(args.adjacency_file)
    print(adjacency_file, args.budget, args.algo, args.T)

    data = read_mc_neighbors(adjacency_file, args.n)
    problem = MaxCoverProblem(data=data, budget=args.budget, n=args.n, q=args.q)

    k_b = problem.max_subset_size()
    greedy_evaluate = problem.n * k_b
    print(k_b)

    result_dir = build_result_dir_for_mc(
        adjacency_file=args.adjacency_file,
        q=args.q,
        algo=args.algo,
        budget=args.budget,
        epsilon=args.epsilon,
        prob=args.prob,
    )

    try:
        algorithm_runner = get_algorithm_runner(args.algo)
    except ValueError:
        print("no suitable algo")
        return

    if args.algo == "EVO_SMC":
        epsilon = 1e-10
        prob = 0.0
    else:
        epsilon = args.epsilon
        prob = args.prob

    writer = ResultWriter(result_dir=result_dir, trial_id=args.times)
    config = AlgorithmConfig(
        trial_id=args.times,
        T=args.T,
        greedy_evaluate=greedy_evaluate,
        epsilon=epsilon,
        prob=prob,
        logger=writer,
        result_dir=result_dir,
    )

    algorithm_runner(problem, config)


if __name__ == "__main__":
    main(parse_args())
