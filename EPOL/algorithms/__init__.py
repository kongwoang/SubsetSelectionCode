from __future__ import annotations

from typing import Callable, Dict

from .eamc import run_eamc
from .epomc import run_epomc
from .epoadapt import run_epoadapt
from .fpomc import run_fpomc
from .gga import run_gga
from .greedy_max import run_greedy_max
from .newalgo import run_newalgo
from .one_guess_greedy_plus import run_one_guess_greedy_plus
from .p_pomc import run_p_pomc
from .pomc import run_pomc
from .sto_evo_smc import run_sto_evo_smc

ALGORITHM_REGISTRY: Dict[str, Callable] = {
    "gga": run_gga,
    "greedy_max": run_greedy_max,
    "newalgo": run_newalgo,
    "one_guess_greedy_plus": run_one_guess_greedy_plus,
    "pomc": run_pomc,
    "eamc": run_eamc,
    "epoadapt": run_epoadapt,
    "fpomc": run_fpomc,
    "sto_evo_smc": run_sto_evo_smc,
    "epomc": run_epomc,
    "p_pomc": run_p_pomc,
}

ALGORITHM_ALIASES: Dict[str, str] = {
    "gga": "gga",
    "greedy_max": "greedy_max",
    "newalgo": "newalgo",
    "new_algo": "newalgo",
    "one_guess_greedy_plus": "one_guess_greedy_plus",
    "pomc": "pomc",
    "eamc": "eamc",
    "epoadapt": "epoadapt",
    "epo_adapt": "epoadapt",
    "fpomc": "fpomc",
    "sto_evo_smc": "sto_evo_smc",
    "evo_smc": "sto_evo_smc",
    "epomc": "epomc",
    "p_pomc": "p_pomc",
    "ppomc": "p_pomc",
    "GGA": "gga",
    "POMC": "pomc",
    "EAMC": "eamc",
    "EPOADAPT": "epoadapt",
    "EPOAdapt": "epoadapt",
    "NEWALGO": "newalgo",
    "FPOMC": "fpomc",
    "EVO_SMC": "sto_evo_smc",
    "sto_EVO_SMC": "sto_evo_smc",
    "EPOMC": "epomc",
    "PPOMC": "p_pomc",
}


def normalize_algorithm_name(name: str) -> str:
    if name in ALGORITHM_ALIASES:
        return ALGORITHM_ALIASES[name]

    lowered = name.lower()
    if lowered in ALGORITHM_ALIASES:
        return ALGORITHM_ALIASES[lowered]

    raise ValueError(f"Unsupported algorithm: {name}")


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY.keys())


def get_algorithm_runner(name: str) -> Callable:
    canonical = normalize_algorithm_name(name)
    return ALGORITHM_REGISTRY[canonical]


__all__ = [
    "ALGORITHM_REGISTRY",
    "ALGORITHM_ALIASES",
    "get_algorithm_runner",
    "normalize_algorithm_name",
    "list_algorithms",
    "run_gga",
    "run_greedy_max",
    "run_newalgo",
    "run_one_guess_greedy_plus",
    "run_pomc",
    "run_eamc",
    "run_epoadapt",
    "run_fpomc",
    "run_sto_evo_smc",
    "run_epomc",
    "run_p_pomc",
]
