import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pressure_screen import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "config/screen.yaml").read_text())


@pytest.fixture(scope="session")
def signal_tables():
    return synthetic.make(signal=True, n=300, seed=3)


@pytest.fixture(scope="session")
def noise_tables():
    return synthetic.make(signal=False, n=300, seed=3)
