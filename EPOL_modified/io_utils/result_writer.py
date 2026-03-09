from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from common.solution import position
from common.types import AlgorithmResult


class ResultWriter:
    def __init__(self, result_dir: str, trial_id: int):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.trial_id = trial_id
        self.result_file = self.result_dir / f"result_{trial_id}.txt"

    def write_main(self, message: str, solution: np.matrix) -> None:
        self._append_solution_log(self.result_file, message, solution)

    def write_subpomc(self, index: int, message: str, solution: np.matrix) -> None:
        sub_dir = self.result_dir / f"times{self.trial_id}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_file = sub_dir / f"subPOMC_index_{index}.txt"
        self._append_solution_log(sub_file, message, solution)

    def write_final_result(self, result: AlgorithmResult, budget: float, extra: Optional[str] = None) -> None:
        pop_text = ""
        if result.population is not None:
            pop_text = f" population = {result.population}"
        extra_text = f" {extra}" if extra else ""
        message = (
            f"value = {result.value} cpu_time_used ="
            f"{round(result.cpu_time_used, 3)} wall_time_used ="
            f"{round(result.wall_time_used, 3)} cost = {result.cost}"
            f" budget = {budget}{pop_text}{extra_text}"
        )
        self.write_main(message, result.solution)

    @staticmethod
    def _append_solution_log(path: Path, message: str, solution: np.matrix) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")
            for item in position(solution):
                handle.write(str(item))
                handle.write(" ")
            handle.write("\n")


def build_result_dir_for_im(adjacency_file: str, algo: str, budget: float, epsilon: float, prob: float) -> str:
    if algo == "sto_EVO_SMC":
        return f"Result_01/{adjacency_file}_{algo}_{epsilon}_{prob}_{budget}"
    return f"Result_01/{adjacency_file}_{algo}_{budget}"


def build_result_dir_for_mc(adjacency_file: str, q: int, algo: str, budget: float, epsilon: float, prob: float) -> str:
    if algo == "sto_EVO_SMC":
        return f"Result-q={q}/{adjacency_file}_{algo}_{epsilon}_{prob}_{budget}"
    return f"Result-q={q}/{adjacency_file}_{algo}_{budget}"
