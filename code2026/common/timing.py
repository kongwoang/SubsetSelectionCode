from __future__ import annotations

import time
from typing import Tuple


def start_timing() -> Tuple[float, float]:
    return time.process_time(), time.time()


def elapsed_time(start_cpu: float, start_wall: float) -> Tuple[float, float]:
    cpu_time_used = time.process_time() - start_cpu
    wall_time_used = time.time() - start_wall
    return cpu_time_used, wall_time_used
