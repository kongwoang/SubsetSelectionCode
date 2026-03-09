from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from algorithms import get_algorithm_runner
from common.types import AlgorithmConfig
from io_utils.graph_readers import read_im_edge_matrix, read_outdegree_eps
from io_utils.result_writer import ResultWriter, build_result_dir_for_im
from problems.influence_maximization import InfluenceMaximizationProblem


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
    parser.add_argument("-adjacency_file", type=str, default="graph200-01.txt")
    parser.add_argument("-outdegree_file", type=str, default="graph200_eps.txt")
    parser.add_argument("-probability", type=float, default=0.1)
    parser.add_argument("-T", type=int, default=20)
    parser.add_argument("-budget", type=float, default=100)
    parser.add_argument("-algo", type=str, default="PPOMC")
    parser.add_argument("-prob", type=float, default=0.5, help="sto_evo_smc_p")
    parser.add_argument("-epsilon", type=float, default=0.1, help="sto_evo_smc_epsilon")
    parser.add_argument("-times", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    adjacency_file = _resolve_data_path(args.adjacency_file)
    outdegree_file = _resolve_data_path(args.outdegree_file)

    print(adjacency_file, args.budget, args.algo, args.T)

    weight_matrix = read_im_edge_matrix(args.probability, adjacency_file)
    node_num = int(np.shape(weight_matrix)[0])
    eps_values = read_outdegree_eps(outdegree_file, node_num)

    problem = InfluenceMaximizationProblem(weight_matrix=weight_matrix, budget=args.budget, eps_values=eps_values)

    k_b = problem.max_subset_size()
    print(k_b)

    greedy_evaluate = problem.n * k_b

    result_dir = build_result_dir_for_im(
        adjacency_file=args.adjacency_file,
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
