from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.types import AlgorithmResult


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _summary_file(result_root: Path, problem: str) -> Path:
    return result_root / f"run_all_{problem}_summary.jsonl"


def append_run_summary(
    *,
    result_root: Path,
    problem: str,
    status: str,
    trial_id: int,
    params: dict[str, Any],
    result: AlgorithmResult | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "problem": problem,
        "status": status,
        "trial_id": trial_id,
        "params": _to_jsonable(params),
    }
    if result is not None:
        payload["result"] = {
            "value": float(result.value),
            "cost": float(result.cost),
            "cpu_time_used": float(result.cpu_time_used),
            "wall_time_used": float(result.wall_time_used),
            "population": result.population,
            "extra": _to_jsonable(result.extra),
        }
    if error is not None:
        payload["error"] = error

    target = _summary_file(result_root, problem)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")
