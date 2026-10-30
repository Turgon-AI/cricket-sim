import json

from cricket_sim import export as export_module


def test_failed_export_keeps_previous_complete_directory(tmp_path, monkeypatch):
    data = tmp_path / "data"
    gold = tmp_path / "gold"
    gold.mkdir()
    fingerprint = "abcdef1234567890"
    (gold / "run.json").write_text(json.dumps({"build_fingerprint": fingerprint}))
    current = data / "export" / fingerprint[:12]
    current.mkdir(parents=True)
    (current / "manifest.json").write_text("complete")
    monkeypatch.setattr(export_module, "DATA", data)
    monkeypatch.setattr(export_module, "GOLD", gold)

    def fail(_allow_dirty, stage, _skip_full_check=False):
        stage.mkdir(parents=True)
        (stage / "manifest.json").write_text("partial")
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(export_module, "_write_export", fail)
    try:
        export_module.export()
    except RuntimeError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("expected the simulated export failure")

    assert (current / "manifest.json").read_text() == "complete"
    assert not list(current.parent.glob(current.name + ".tmp-*"))
