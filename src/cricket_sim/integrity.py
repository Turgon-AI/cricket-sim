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
    return model_run
