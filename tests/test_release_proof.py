"""What a release must prove about itself (data-engineering review, 2026-10-27).

- a build cannot record itself as clean when git cannot report the tree's state
- export refuses a release unless scripts/full_check.sh passed on exactly this build and model"""

import json
import subprocess

import pytest

from cricket_sim import build, integrity


# ---- git state ---------------------------------------------------------------------------------------------

def test_missing_git_is_recorded_as_dirty(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    state = build._git_state()
    assert state["dirty"] is True and state["commit"] is None


def test_failing_git_is_recorded_as_dirty(monkeypatch):
    # e.g. a copied tree that is not a repository: git runs but exits non-zero with empty stdout
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "not a git repo"))
    state = build._git_state()
    assert state["dirty"] is True and state["commit"] is None


# ---- full-check record -------------------------------------------------------------------------------------

def _runs(build_fp="b" * 64, model_fp="m" * 64):
    gold = {"build_fingerprint": build_fp, "code": {"commit": "c" * 40, "dirty": False, "source_sha256": "s" * 64}}
    return gold, {"model_fingerprint": model_fp}


def _write_runs(data, gold, model):
    (data / "gold").mkdir(parents=True, exist_ok=True)
    (data / "models").mkdir(parents=True, exist_ok=True)
    (data / "gold/run.json").write_text(json.dumps(gold))
    (data / "models/run.json").write_text(json.dumps(model))


def test_release_export_needs_a_passed_full_check(tmp_path):
    gold, model = _runs()
    _write_runs(tmp_path, gold, model)
    with pytest.raises(SystemExit, match="no passed full-data check"):
        integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)
    integrity.verify_full_check(tmp_path, gold, model, allow_dirty=True)        # tests and dirty exports skip it

    integrity.record_full_check(tmp_path)
    integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)
    record = json.loads((tmp_path / integrity.FULL_CHECK).read_text())
    assert record["passed"] and record["build_fingerprint"] == gold["build_fingerprint"]


@pytest.mark.parametrize("stale", ["build", "model", "source"])
def test_a_full_check_of_another_build_does_not_count(tmp_path, stale):
    gold, model = _runs()
    _write_runs(tmp_path, gold, model)
    integrity.record_full_check(tmp_path)
    if stale == "build":
        gold = {**gold, "build_fingerprint": "x" * 64}
    elif stale == "model":
        model = {"model_fingerprint": "x" * 64}
    else:
        gold = {**gold, "code": {**gold["code"], "source_sha256": "x" * 64}}
    with pytest.raises(SystemExit, match="no passed full-data check"):
        integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)


def test_the_tiny_build_has_no_full_check_record(tiny_build):
    # CI builds the 10-match sample only; nothing may claim the full check ran on it
    assert not (tiny_build / integrity.FULL_CHECK).exists()
