from __future__ import annotations

import numpy as np

from ..common.timing import elapsed_time, start_timing
from ..common.types import AlgorithmConfig, AlgorithmResult


def run_newalgo(problem, config: AlgorithmConfig) -> AlgorithmResult:
    """Placeholder algorithm for future implementation."""
    start_cpu, start_wall = start_timing()

    n = problem.n
    solution = np.mat(np.zeros((1, n)), "int8")

    # TODO: implement actual search logic.
    value = float(problem.FS(solution))
    cost = float(problem.CS(solution))

    cpu_time_used, wall_time_used = elapsed_time(start_cpu, start_wall)
    result = AlgorithmResult(
        solution=solution,
        value=value,
        cost=cost,
        cpu_time_used=cpu_time_used,
        wall_time_used=wall_time_used,
    )

    if config.logger is not None:
        message = (
            f"value = {result.value} cpu_time_used = {round(cpu_time_used, 3)}"
            f" wall_time_used = {round(wall_time_used, 3)} cost = {result.cost}"
            f" budget = {problem.budget}"
        )
        config.logger.write_main(message, result.solution)

    return result
