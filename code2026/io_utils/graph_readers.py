from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np


def read_im_edge_matrix(probability: float, file_path: str) -> np.matrix:
    path = Path(file_path)
    max_node = 0
    with path.open("r", encoding="utf-8") as data_file:
        for line in data_file:
            items = line.split()
            if not items:
                continue
            start = int(items[0])
            end = int(items[1])
            max_node = max(max_node, start, end)

    matrix = np.mat(np.zeros([max_node, max_node]))
    with path.open("r", encoding="utf-8") as data_file:
        for line in data_file:
            items = line.split()
            if not items:
                continue
            matrix[int(items[0]) - 1, int(items[1]) - 1] = probability
    return matrix


def read_outdegree_eps(file_path: str, n: int) -> List[float]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as data_file:
        first_line = data_file.readline().split()
    eps = [0.0] * n
    for i in range(min(n, len(first_line))):
        eps[i] = float(first_line[i])
    return eps


def read_mc_neighbors(file_path: str, n: int) -> List[List[int]]:
    path = Path(file_path)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    grouped = defaultdict(list)

    with path.open("r", encoding="utf-8") as data_file:
        for raw in data_file:
            parts = raw.split()
            if len(parts) < 2:
                continue
            source = int(parts[0]) - 1
            target = int(parts[1]) - 1
            if 0 <= source < n:
                grouped[source].append(target)

    for node in range(n):
        neighbors[node] = grouped.get(node, [])
    return neighbors
