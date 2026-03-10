from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Callable, Sequence

from ..algorithms import list_algorithms
from .local_config import IM_DEFAULTS, MC_DEFAULTS
from .run_im import run_im
from .run_mc import run_mc
from .summary import append_run_summary

# Edit this block directly to define your grid-search defaults.
GRID_CONFIG = {
    "problem": "both",  # im | mc | both
    "algorithms": "all",  # "all" or comma-separated names
    "trials": 1,
    "trial_start": 0,
    "max_workers": None,
    "disable_progress": False,
    "stop_on_error": False,
    "dry_run": False,
    "data_dir": None,  # None -> use local_config defaults
    "result_root": None,  # None -> use local_config defaults
    "im": {
        # Keep one default configuration for IM.
        "adjacency_files": [IM_DEFAULTS["adjacency_file"]],
        "outdegree_files": [IM_DEFAULTS["outdegree_file"]],
        "probabilities": [IM_DEFAULTS["probability"]],
        "budgets": [IM_DEFAULTS["budget"]],
        "iterations": [IM_DEFAULTS["iterations"]],
        "probs": [IM_DEFAULTS["prob"]],
        "epsilons": [IM_DEFAULTS["epsilon"]],
    },
    "mc": {
        # Keep one default configuration for MC.
        "adjacency_files": [MC_DEFAULTS["adjacency_file"]],
        "qs": [MC_DEFAULTS["q"]],
        "ns": [MC_DEFAULTS["n"]],
        "budgets": [MC_DEFAULTS["budget"]],
        "iterations": [MC_DEFAULTS["iterations"]],
        "probs": [MC_DEFAULTS["prob"]],
        "epsilons": [MC_DEFAULTS["epsilon"]],
    },
}


def _csv_default(items: Sequence[object]) -> str:
    return ",".join(str(item) for item in items)


def _parse_csv(raw: str, cast: Callable[[str], object]) -> list:
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
    return [str(item) for item in _parse_csv(raw, str)]


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Run grid search for IM/MC experiments",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument("--problem", choices=["im", "mc", "both"], default=GRID_CONFIG["problem"])
    parser.add_argument(
        "--algorithms",
        type=str,
        default=str(GRID_CONFIG["algorithms"]),
        help="Comma-separated list or 'all'",
    )
    parser.add_argument("--trials", type=int, default=int(GRID_CONFIG["trials"]), help="Number of trial_id repetitions")
    parser.add_argument("--trial-start", type=int, default=int(GRID_CONFIG["trial_start"]), help="Starting trial_id")
    parser.add_argument("--max-workers", type=int, default=GRID_CONFIG["max_workers"])
    parser.add_argument("--disable-progress", action="store_true", default=bool(GRID_CONFIG["disable_progress"]))
    parser.add_argument("--stop-on-error", action="store_true", default=bool(GRID_CONFIG["stop_on_error"]))
    parser.add_argument("--dry-run", action="store_true", default=bool(GRID_CONFIG["dry_run"]))

    parser.add_argument("--data-dir", type=Path, default=GRID_CONFIG["data_dir"], help="Override data dir for both IM and MC")
    parser.add_argument("--result-root", type=Path, default=GRID_CONFIG["result_root"], help="Override result root for both IM and MC")

    parser.add_argument("--im-adjacency-files", type=str, default=_csv_default(GRID_CONFIG["im"]["adjacency_files"]))
    parser.add_argument("--im-outdegree-files", type=str, default=_csv_default(GRID_CONFIG["im"]["outdegree_files"]))
    parser.add_argument("--im-probabilities", type=str, default=_csv_default(GRID_CONFIG["im"]["probabilities"]))
    parser.add_argument("--im-budgets", type=str, default=_csv_default(GRID_CONFIG["im"]["budgets"]))
    parser.add_argument("--im-iterations", type=str, default=_csv_default(GRID_CONFIG["im"]["iterations"]))
    parser.add_argument("--im-probs", type=str, default=_csv_default(GRID_CONFIG["im"]["probs"]))
    parser.add_argument("--im-epsilons", type=str, default=_csv_default(GRID_CONFIG["im"]["epsilons"]))

    parser.add_argument("--mc-adjacency-files", type=str, default=_csv_default(GRID_CONFIG["mc"]["adjacency_files"]))
    parser.add_argument("--mc-qs", type=str, default=_csv_default(GRID_CONFIG["mc"]["qs"]))
    parser.add_argument("--mc-ns", type=str, default=_csv_default(GRID_CONFIG["mc"]["ns"]))
    parser.add_argument("--mc-budgets", type=str, default=_csv_default(GRID_CONFIG["mc"]["budgets"]))
    parser.add_argument("--mc-iterations", type=str, default=_csv_default(GRID_CONFIG["mc"]["iterations"]))
    parser.add_argument("--mc-probs", type=str, default=_csv_default(GRID_CONFIG["mc"]["probs"]))
    parser.add_argument("--mc-epsilons", type=str, default=_csv_default(GRID_CONFIG["mc"]["epsilons"]))
    return parser


def _run_im_grid(args: argparse.Namespace, algorithms: Sequence[str]) -> tuple[int, int]:
    adjacency_files = _parse_csv(args.im_adjacency_files, str)
    outdegree_files = _parse_csv(args.im_outdegree_files, str)
    probabilities = _parse_csv(args.im_probabilities, float)
    budgets = _parse_csv(args.im_budgets, float)
    iterations = _parse_csv(args.im_iterations, int)
    probs = _parse_csv(args.im_probs, float)
    epsilons = _parse_csv(args.im_epsilons, float)

    base_data_dir = args.data_dir if args.data_dir is not None else IM_DEFAULTS["data_dir"]
    base_result_root = args.result_root if args.result_root is not None else IM_DEFAULTS["result_root"]
    planned = (
        args.trials
        * len(adjacency_files)
        * len(outdegree_files)
        * len(probabilities)
        * len(budgets)
        * len(iterations)
        * len(algorithms)
        * len(probs)
        * len(epsilons)
    )
    print(
        f"[IM] planned={planned} "
        f"(trials={args.trials}, files={len(adjacency_files)}, outdegrees={len(outdegree_files)}, "
        f"algorithms={len(algorithms)})"
    )
    total = 0
    failed = 0

    for trial_offset in range(args.trials):
        trial_id = args.trial_start + trial_offset
        for combo in product(
            adjacency_files,
            outdegree_files,
            probabilities,
            budgets,
            iterations,
            algorithms,
            probs,
            epsilons,
        ):
            (
                adjacency_file,
                outdegree_file,
                probability,
                budget,
                iteration,
                algorithm,
                prob,
                epsilon,
            ) = combo
            total += 1
            started = time.perf_counter()
            print(
                f"[IM][{total}/{planned}] START "
                f"trial={trial_id} algo={algorithm} adj={adjacency_file} out={outdegree_file} "
                f"p={probability} budget={budget} iter={iteration} prob={prob} eps={epsilon}"
            )
            run_args = argparse.Namespace(
                data_dir=base_data_dir,
                result_root=base_result_root,
                adjacency_file=adjacency_file,
                outdegree_file=outdegree_file,
                probability=probability,
                budget=budget,
                iterations=iteration,
                algorithm=algorithm,
                prob=prob,
                epsilon=epsilon,
                trial_id=trial_id,
                max_workers=args.max_workers,
                disable_progress=args.disable_progress,
            )
            if args.dry_run:
                print(f"[DRY][IM] trial={trial_id} algo={algorithm} budget={budget} p={probability} file={adjacency_file}")
                append_run_summary(
                    result_root=Path(base_result_root),
                    problem="im",
                    status="dry_run",
                    trial_id=trial_id,
                    params={
                        "algorithm": algorithm,
                        "adjacency_file": adjacency_file,
                        "outdegree_file": outdegree_file,
                        "probability": probability,
                        "budget": budget,
                        "iterations": iteration,
                        "prob": prob,
                        "epsilon": epsilon,
                    },
                )
                elapsed = time.perf_counter() - started
                print(f"[IM][{total}/{planned}] DRY_DONE elapsed={elapsed:.2f}s")
                continue
            try:
                run_im(run_args)
                elapsed = time.perf_counter() - started
                print(f"[IM][{total}/{planned}] OK elapsed={elapsed:.2f}s")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                elapsed = time.perf_counter() - started
                print(
                    f"[IM][{total}/{planned}] ERROR elapsed={elapsed:.2f}s "
                    f"trial={trial_id} algo={algorithm} file={adjacency_file}: {exc}"
                )
                if args.stop_on_error:
                    raise
    return total, failed


def _run_mc_grid(args: argparse.Namespace, algorithms: Sequence[str]) -> tuple[int, int]:
    adjacency_files = _parse_csv(args.mc_adjacency_files, str)
    qs = _parse_csv(args.mc_qs, int)
    ns = _parse_csv(args.mc_ns, int)
    budgets = _parse_csv(args.mc_budgets, float)
    iterations = _parse_csv(args.mc_iterations, int)
    probs = _parse_csv(args.mc_probs, float)
    epsilons = _parse_csv(args.mc_epsilons, float)

    base_data_dir = args.data_dir if args.data_dir is not None else MC_DEFAULTS["data_dir"]
    base_result_root = args.result_root if args.result_root is not None else MC_DEFAULTS["result_root"]
    planned = (
        args.trials
        * len(adjacency_files)
        * len(qs)
        * len(ns)
        * len(budgets)
        * len(iterations)
        * len(algorithms)
        * len(probs)
        * len(epsilons)
    )
    print(
        f"[MC] planned={planned} "
        f"(trials={args.trials}, files={len(adjacency_files)}, algorithms={len(algorithms)})"
    )
    total = 0
    failed = 0

    for trial_offset in range(args.trials):
        trial_id = args.trial_start + trial_offset
        for combo in product(
            adjacency_files,
            qs,
            ns,
            budgets,
            iterations,
            algorithms,
            probs,
            epsilons,
        ):
            adjacency_file, q, n, budget, iteration, algorithm, prob, epsilon = combo
            total += 1
            started = time.perf_counter()
            print(
                f"[MC][{total}/{planned}] START "
                f"trial={trial_id} algo={algorithm} adj={adjacency_file} q={q} n={n} "
                f"budget={budget} iter={iteration} prob={prob} eps={epsilon}"
            )
            run_args = argparse.Namespace(
                data_dir=base_data_dir,
                result_root=base_result_root,
                adjacency_file=adjacency_file,
                q=q,
                n=n,
                budget=budget,
                iterations=iteration,
                algorithm=algorithm,
                prob=prob,
                epsilon=epsilon,
                trial_id=trial_id,
                max_workers=args.max_workers,
                disable_progress=args.disable_progress,
            )
            if args.dry_run:
                print(f"[DRY][MC] trial={trial_id} algo={algorithm} budget={budget} q={q} n={n} file={adjacency_file}")
                append_run_summary(
                    result_root=Path(base_result_root),
                    problem="mc",
                    status="dry_run",
                    trial_id=trial_id,
                    params={
                        "algorithm": algorithm,
                        "adjacency_file": adjacency_file,
                        "q": q,
                        "n": n,
                        "budget": budget,
                        "iterations": iteration,
                        "prob": prob,
                        "epsilon": epsilon,
                    },
                )
                elapsed = time.perf_counter() - started
                print(f"[MC][{total}/{planned}] DRY_DONE elapsed={elapsed:.2f}s")
                continue
            try:
                run_mc(run_args)
                elapsed = time.perf_counter() - started
                print(f"[MC][{total}/{planned}] OK elapsed={elapsed:.2f}s")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                elapsed = time.perf_counter() - started
                print(
                    f"[MC][{total}/{planned}] ERROR elapsed={elapsed:.2f}s "
                    f"trial={trial_id} algo={algorithm} file={adjacency_file}: {exc}"
                )
                if args.stop_on_error:
                    raise
    return total, failed


def run_all(args: argparse.Namespace) -> dict[str, int]:
    algorithms = _parse_algorithms(args.algorithms)
    if not algorithms:
        raise ValueError("No algorithms selected. Use --algorithms with at least one item.")

    total = 0
    failed = 0

    if args.problem in {"im", "both"}:
        im_total, im_failed = _run_im_grid(args, algorithms)
        total += im_total
        failed += im_failed

    if args.problem in {"mc", "both"}:
        mc_total, mc_failed = _run_mc_grid(args, algorithms)
        total += mc_total
        failed += mc_failed

    succeeded = total - failed
    print(f"[SUMMARY] total={total} succeeded={succeeded} failed={failed}")
    return {"total": total, "succeeded": succeeded, "failed": failed}
