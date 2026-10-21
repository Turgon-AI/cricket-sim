import pytest

from cricket_sim import bronze, paths


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BRONZE", tmp_path / "bronze")
    monkeypatch.setattr(bronze, "BRONZE", tmp_path / "bronze")
    monkeypatch.setattr(bronze, "LOCK_FILE", tmp_path / "lock.yml")
    monkeypatch.setattr(bronze, "PINS_FILE", tmp_path / "pins.yml")
    folder = paths.snapshot_dir("src", "s1")
    folder.mkdir(parents=True)
    (folder / "a.txt").write_text("hello")
    bronze.write_manifest("src", "s1", {}, [{"name": "a.txt", "stored_as": "a.txt"}])
    bronze.lock_source("src", "s1")
    return folder


def test_intact_snapshot_verifies(fake_repo):
    assert bronze.verify("src", "s1") == []


def test_tampered_file_is_caught(fake_repo):
    (fake_repo / "a.txt").write_text("HELLO")
    assert any("checksum mismatch" in p for p in bronze.verify("src", "s1"))


def test_missing_file_is_caught(fake_repo):
    (fake_repo / "a.txt").unlink()
    assert any("missing a.txt" in p for p in bronze.verify("src", "s1"))


def test_unlisted_file_is_caught(fake_repo):
    (fake_repo / "._a.txt").write_text("macOS resource fork")
    assert any("unlisted file" in p for p in bronze.verify("src", "s1"))


def test_replaced_manifest_is_caught(fake_repo):
    # a whole snapshot swapped in under the same id still self-verifies; the pin must catch it
    (fake_repo / "a.txt").write_text("other")
    bronze.write_manifest("src", "s1", {"note": "swapped"}, [{"name": "a.txt", "stored_as": "a.txt"}])
    assert any("differs from the one that was locked" in p for p in bronze.verify("src", "s1"))


def test_missing_pin_for_locked_snapshot_is_caught(fake_repo):
    bronze.PINS_FILE.write_text("{}\n")
    assert any("no manifest hash pin" in p for p in bronze.verify("src", "s1"))
