from __future__ import annotations

from pathlib import Path

from EPOL_modified.runners.pathing import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR


IM_DEFAULTS = {
    "data_dir": DEFAULT_DATA_DIR,
    "result_root": DEFAULT_RESULTS_DIR,
    "adjacency_file": "graph200-01.txt",
    "outdegree_file": "graph200_eps.txt",
    "probability": 0.1,
    "budget": 100.0,
    "iterations": 20,
    "algorithm": "p_pomc",
    "prob": 0.5,
    "epsilon": 0.1,
    "trial_id": 0,
    "max_workers": None,
    "disable_progress": False,
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
}


def update_paths(data_dir: str | Path | None = None, result_root: str | Path | None = None) -> None:
    """Optional helper if you want to tweak defaults in interactive use."""
    if data_dir is not None:
        IM_DEFAULTS["data_dir"] = Path(data_dir)
        MC_DEFAULTS["data_dir"] = Path(data_dir)
    if result_root is not None:
        IM_DEFAULTS["result_root"] = Path(result_root)
        MC_DEFAULTS["result_root"] = Path(result_root)
