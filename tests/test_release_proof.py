"""What a release must prove about itself (data-engineering review, 2026-10-27).

- a build cannot record itself as clean when git cannot report the tree's state"""

import subprocess

from cricket_sim import build


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
