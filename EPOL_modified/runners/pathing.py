from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "outdegree"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"


def resolve_input_file(file_or_path: str, data_dir: Path) -> Path:
    path = Path(file_or_path)
    if path.is_file():
        return path

    candidate = data_dir / file_or_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Input file not found: {file_or_path} (searched in {data_dir})")
