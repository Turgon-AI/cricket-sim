"""Shared fixtures. `tiny_build` runs the real CLI end to end on the committed fixture (tests/fixtures/tiny):
verify -> build -> model -> export, in a fresh interpreter pointed at a temporary data root, exactly as CI does."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tiny"


def run_cli(data: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CRICKET_SIM_DATA": str(data), "CRICKET_SIM_SNAPSHOTS": str(FIXTURE)}
    return subprocess.run([sys.executable, "-m", "cricket_sim.cli", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=900)


def build_fixture(data: Path) -> dict:
    shutil.copytree(FIXTURE / "bronze", data / "bronze")
    # the fixture's ODIs span 2011-2023, so let the model predict from the second season on with tiny history
    model = ["model", "--allow-dirty", "--first-season", "2000", "--min-train-matches", "1"]
    for args in (["verify"], ["build", "--allow-dirty"], model, ["export", "--allow-dirty", "--skip-full-check"]):
        result = run_cli(data, *args)
        assert result.returncode == 0, f"cricket {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
    return json.loads((data / "gold" / "run.json").read_text())


@pytest.fixture(scope="session")
def tiny_build(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("tiny") / "data"
    build_fixture(data)
    return data


@pytest.fixture(scope="session")
def tiny_model_rows(tiny_build):
    """The real modelling table (models/dataset.py model_rows) of the fixture's gold layer."""
    import duckdb

    from cricket_sim.models import dataset

    con = duckdb.connect()
    dataset.load(con, gold=tiny_build / "gold")
    return con.execute("SELECT * FROM model_rows").df()
