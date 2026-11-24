import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402
from bnpl import data, pipeline as pl  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return data.load_raw()


@pytest.fixture(scope="session")
def df():
    return data.load_clean()


@pytest.fixture(scope="session")
def splits(df):
    return pl.split(df)
