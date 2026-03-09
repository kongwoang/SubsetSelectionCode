from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AlgorithmResult:
    solution: np.matrix
    value: float
    cost: float
    cpu_time_used: float
    wall_time_used: float
    population: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    trial_id: int = 0
    T: int = 20
    greedy_evaluate: int = 0
    epsilon: float = 1e-10
    prob: float = 0.0
    logger: Any = None
    result_dir: Optional[str] = None
    max_workers: Optional[int] = None
    enable_progress_bar: bool = True
    checkpoint_patience: int = 10
