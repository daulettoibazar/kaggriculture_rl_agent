"""Shared fixtures. Puts the repo root on ``sys.path`` so ``src.*`` imports work."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a CUDA device")
    config.addinivalue_line("markers", "kaggri: needs the kaggri extension module")
    config.addinivalue_line("markers", "slow: takes more than ~20 s")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def has_kaggri() -> bool:
    try:
        import kaggri  # noqa: F401

        return True
    except Exception:
        return False
