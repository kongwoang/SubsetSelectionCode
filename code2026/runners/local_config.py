from __future__ import annotations

from pathlib import Path

from .pathing import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR


IM_DEFAULTS = {
    "data_dir": DEFAULT_DATA_DIR,
    "result_root": DEFAULT_RESULTS_DIR,
    "adjacency_file": "graph200-01.txt",
    "outdegree_file": "graph200_eps.txt",
    "probability": 0.1,
    "budget": 100.0,
    "iterations": 20,
    "algorithm": "epoadapt",
    "prob": 0.5,
    "epsilon": 0.1,
    "trial_id": 0,
    "max_workers": None,
    "disable_progress": False,
    "grid_search": False,
    "stop_on_error": False,
}


MC_DEFAULTS = {
    "data_dir": DEFAULT_DATA_DIR,
    "result_root": DEFAULT_RESULTS_DIR,
    "adjacency_file": "congress.edgelist-new.txt",
    "q": 5,
    "n": 475,
    "budget": 500.0,
    "iterations": 20,
    "algorithm": "p_pomc",
    "prob": 0.5,
    "epsilon": 0.1,
    "trial_id": 0,
    "max_workers": None,
    "disable_progress": False,
    "grid_search": False,
    "stop_on_error": False,
}


# Per-algorithm hyper-parameter grid for IM runner.
# Keys are canonical algorithm names; "*" is fallback.
IM_ALGO_GRID = {
    "*": {
        "iterations": [IM_DEFAULTS["iterations"]],
        "probs": [IM_DEFAULTS["prob"]],
        "epsilons": [IM_DEFAULTS["epsilon"]],
        "max_workers": [IM_DEFAULTS["max_workers"]],
        "algo_params": {},
    },
    "sto_evo_smc": {
        "probs": [0.3, 0.5, 0.7],
        "epsilons": [0.05, 0.1, 0.2],
    },
    "epoadapt": {
        "max_workers": [6],
        "algo_params_mode": "product",
        "algo_params": {
            "lambda_penalty": [0.5, 1.0, 2.0],
            "alpha": [2.0, 3.0, 4.0],
            "beta": [0.5, 1.0, 1.5],
            "burst_prob": [0.0, 0.05],
            "kappa": [0.3, 0.5, 0.8],
            "theta_eps": [0.3, 0.5],
            "rr_max_candidates": [20],
            "top_k": [20],
            "sub_patience": [15],
        },
    },
}


# Per-algorithm hyper-parameter grid for MC runner.
# Keys are canonical algorithm names; "*" is fallback.
MC_ALGO_GRID = {
    "*": {
        "iterations": [MC_DEFAULTS["iterations"]],
        "probs": [MC_DEFAULTS["prob"]],
        "epsilons": [MC_DEFAULTS["epsilon"]],
        "max_workers": [MC_DEFAULTS["max_workers"]],
        "algo_params": {},
    },
    "sto_evo_smc": {
        "probs": [0.3, 0.5, 0.7],
        "epsilons": [0.05, 0.1, 0.2],
    },
    "epoadapt": {
        "max_workers": [6],
        "algo_params_mode": "product",
        "algo_params": {
            "lambda_penalty": [0.5, 1.0, 2.0],
            "alpha": [2.0, 3.0, 4.0],
            "beta": [0.5, 1.0, 1.5],
            "burst_prob": [0.0, 0.05],
            "kappa": [0.3, 0.5, 0.8],
            "theta_eps": [0.3, 0.5],
            "rr_max_candidates": [20],
            "top_k": [20],
            "sub_patience": [15],
        },
    },
}


def update_paths(data_dir: str | Path | None = None, result_root: str | Path | None = None) -> None:
    """Optional helper if you want to tweak defaults in interactive use."""
    if data_dir is not None:
        IM_DEFAULTS["data_dir"] = Path(data_dir)
        MC_DEFAULTS["data_dir"] = Path(data_dir)
    if result_root is not None:
        IM_DEFAULTS["result_root"] = Path(result_root)
        MC_DEFAULTS["result_root"] = Path(result_root)
