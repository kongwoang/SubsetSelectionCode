from __future__ import annotations

import numpy as np

# NumPy 2.x removed np.mat; keep legacy code paths working.
if not hasattr(np, "mat"):
    np.mat = np.asmatrix  # type: ignore[attr-defined]
