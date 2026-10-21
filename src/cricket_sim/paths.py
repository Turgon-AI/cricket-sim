"""Where everything lives. Two environment variables let the same code run against another data tree
(used by the end-to-end test and CI, which build a tiny committed fixture instead of the real data):

  CRICKET_SIM_DATA       data root holding bronze/, staging/, silver/, gold/, models/, export/ (default: ./data)
  CRICKET_SIM_SNAPSHOTS  folder holding snapshots.lock.yml and snapshots.pins.yml (default: ./config)

Everything else in config/ (scope, sources, overrides, schemas) is always the project's own.
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = Path(os.environ.get("CRICKET_SIM_DATA") or ROOT / "data").resolve()
BRONZE = DATA / "bronze"
STAGING = DATA / "staging"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
MODELS = DATA / "models"
SNAPSHOTS = Path(os.environ.get("CRICKET_SIM_SNAPSHOTS") or CONFIG).resolve()
LOCK_FILE = SNAPSHOTS / "snapshots.lock.yml"
PINS_FILE = SNAPSHOTS / "snapshots.pins.yml"
# A fixture's snapshot folder marks itself as a SUBSET of the real snapshots. Checks that are only true of
# complete data (e.g. "every venue override still matches some Cricsheet venue") are skipped for it.
SUBSET = (SNAPSHOTS / "SUBSET").exists()


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


def snapshot_dir(source: str, snapshot_id: str) -> Path:
    return BRONZE / source / snapshot_id
