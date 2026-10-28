"""Validate recorded artifact bytes and dependency identity before downstream consumption."""

import hashlib
import json
from pathlib import Path


def source_provenance(root: Path) -> dict:
    """Hash producing source, contracts and configuration, including uncommitted changes."""
    files = sorted({*root.glob("src/**/*.py"), *root.glob("contracts/**/*.yaml"),
                    *root.glob("config/*.yml"), root / "pyproject.toml", root / "uv.lock"})
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "input_file_sha256": hashes}


def verify_artifacts(data: Path, run: dict, layers: tuple[str, ...]) -> None:
    """Reject missing, extra, unreadable, changed or incompletely recorded artifact files.

    Version 2 fingerprints include SHA256 as well as row/schema hashes. Older runs must
    be rebuilt once; they cannot prove the byte-integrity promised by current exports.
    """
    import duckdb
    from .build import fingerprint

    expected = {name: value for name, value in run.get("tables", {}).items()
                if name.split("/", 1)[0] in layers}
    actual = {f"{layer}/{p.stem}": p for layer in layers for p in (data/layer).glob("*.parquet")}
    if not expected or set(actual) != set(expected):
        raise SystemExit("artifact inventory mismatch: missing, extra or unrecorded tables; rebuild first")
    con = duckdb.connect()
    try:
        for name in sorted(expected):
            if "sha256" not in expected[name]:
                raise SystemExit(f"{name}: legacy artifact fingerprint lacks SHA256; rebuild first")
            try:
                current = fingerprint(con, actual[name])
            except Exception as exc:
                raise SystemExit(f"{name}: artifact cannot be read; rebuild first") from exc
            if current != expected[name]:
                raise SystemExit(f"{name}: artifact fingerprint mismatch; rebuild first")
    finally:
        con.close()


FULL_CHECK = "gold/full_check.json"


def full_check_stamp(gold_run: dict, model_run: dict) -> dict:
    """What a passed full-data check vouches for: this exact build, model and source."""
    return {"build_fingerprint": gold_run["build_fingerprint"], "model_fingerprint": model_run["model_fingerprint"],
            "commit": gold_run["code"]["commit"], "dirty": gold_run["code"]["dirty"],
            "source_sha256": gold_run["code"]["source_sha256"]}


def record_full_check(data: Path) -> None:
    """Called by scripts/full_check.sh after every check passed: write gold/full_check.json for the live build.

    It sits in gold/, so the next build (which publishes a fresh gold/) drops it, and export refuses until the
    full check is run again."""
    import os
    from datetime import datetime, timezone
    from .transaction import writer_lock

    with writer_lock(data):
        gold_run = json.loads((data / "gold/run.json").read_text())
        model_run = json.loads((data / "models/run.json").read_text())
        record = {"passed": True, **full_check_stamp(gold_run, model_run),
                  "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmp = data / (FULL_CHECK + ".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, data / FULL_CHECK)
    print(f"recorded the passed full check for build {record['build_fingerprint'][:16]} in {data / FULL_CHECK}")


def verify_full_check(data: Path, gold_run: dict, model_run: dict, allow_dirty: bool) -> None:
    """A release export needs proof that scripts/full_check.sh passed on exactly this build and model.
    CI only builds the 10-match sample, so without this nothing shows the real data was checked."""
    path = data / FULL_CHECK
    record = json.loads(path.read_text()) if path.exists() else None
    if record and record.get("passed") and all(record.get(k) == v
                                               for k, v in full_check_stamp(gold_run, model_run).items()):
        return
    if not allow_dirty:
        raise SystemExit(f"no passed full-data check recorded for this build and model ({FULL_CHECK}); "
                         "run scripts/full_check.sh, which exports at its end")


def verify_release_inputs(data: Path, gold_run: dict, allow_dirty: bool) -> dict:
    """Verify both generations before any export files are staged."""
    path = data / "models/run.json"
    if not path.exists():
        raise SystemExit("models/run.json missing; run cricket model before export")
    model_run = json.loads(path.read_text())
    if model_run.get("build_fingerprint") != gold_run.get("build_fingerprint"):
        raise SystemExit("model/gold build fingerprint mismatch; rebuild models before export")
    if not allow_dirty and any(r.get("code", {}).get("dirty", True) for r in (gold_run, model_run)):
        raise SystemExit("refusing to export data or models made from uncommitted code")
    verify_artifacts(data, gold_run, ("staging", "silver", "gold"))
    verify_artifacts(data, model_run, ("models",))
    headline = data / "gold/headline.json"
    if not headline.exists() or hashlib.sha256(headline.read_bytes()).hexdigest() != gold_run.get("headline_sha256"):
        raise SystemExit("gold headline fingerprint mismatch; rebuild first")
    outputs = {"tables": gold_run["tables"], "headline_sha256": gold_run["headline_sha256"]}
    if hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest() != gold_run.get("build_fingerprint"):
        raise SystemExit("gold run fingerprint does not match recorded artifacts")
    if hashlib.sha256(json.dumps(model_run["tables"], sort_keys=True).encode()).hexdigest() != model_run.get("model_fingerprint"):
        raise SystemExit("model run fingerprint does not match recorded artifacts")
    verify_full_check(data, gold_run, model_run, allow_dirty)
    return model_run
