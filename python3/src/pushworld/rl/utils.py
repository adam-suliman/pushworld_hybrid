# Copyright 2026
#
# Shared utilities for PushWorld RL scripts and training.

import csv
import json
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: object) -> None:
    with open(path, "w") as file:
        json.dump(_json_ready(data), file, indent=2, sort_keys=True)


def read_json(path: str) -> dict:
    with open(path) as file:
        return json.load(file)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class CSVLogger:
    """Small CSV logger that writes one row at a time."""

    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self._path = path
        self._fieldnames = list(fieldnames)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: Dict[str, object]) -> None:
        self._writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _json_ready(data: object) -> object:
    if is_dataclass(data):
        return asdict(data)
    if isinstance(data, dict):
        return {key: _json_ready(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_ready(value) for value in data]
    return data
