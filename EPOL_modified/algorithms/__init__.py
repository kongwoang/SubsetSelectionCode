from __future__ import annotations

from typing import Callable, Dict

from algorithms.eamc import run_eamc
from algorithms.epomc import run_epomc
from algorithms.fpomc import run_fpomc
from algorithms.gga import run_gga
from algorithms.greedy_max import run_greedy_max
from algorithms.one_guess_greedy_plus import run_one_guess_greedy_plus
from algorithms.p_pomc import run_p_pomc
from algorithms.pomc import run_pomc
from algorithms.sto_evo_smc import run_sto_evo_smc

ALGORITHM_REGISTRY: Dict[str, Callable] = {
    "GGA": run_gga,
    "greedy_max": run_greedy_max,
    "one_guess_greedy_plus": run_one_guess_greedy_plus,
    "POMC": run_pomc,
    "EAMC": run_eamc,
    "FPOMC": run_fpomc,
    "EVO_SMC": run_sto_evo_smc,
    "sto_EVO_SMC": run_sto_evo_smc,
    "EPOMC": run_epomc,
    "PPOMC": run_p_pomc,
}


def get_algorithm_runner(name: str) -> Callable:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unsupported algorithm: {name}")
    return ALGORITHM_REGISTRY[name]


__all__ = [
    "ALGORITHM_REGISTRY",
    "get_algorithm_runner",
    "run_gga",
    "run_greedy_max",
    "run_one_guess_greedy_plus",
    "run_pomc",
    "run_eamc",
    "run_fpomc",
    "run_sto_evo_smc",
    "run_epomc",
    "run_p_pomc",
]
