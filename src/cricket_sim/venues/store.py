"""Bronze hygiene for the network-fetched context sources (venues_geo, weather; also used by wikidata_players).

* Every stored response is a self-describing record {"request": {url, params}, "fetched_at_utc", "response"}
  (older snapshots stored the bare response and kept the request in the manifest; readers accept both).
* Writes are atomic: compress to <name>.tmp, then os.replace. A resumed folder deletes leftover *.tmp files.
* Failures are recorded in failures.json; finalize refuses to seal a snapshot with failures unless allow_gaps.
* Readers use only files listed in _snapshot.json, and refuse a snapshot folder holding any file the manifest
  does not list (a sync-tool leftover or a macOS ._ file is an error, not something to skip quietly).
* Snapshot ids are <UTC date>-<plan sha8>, so a new plan (or a re-fetch on another day) never collides with a
  sealed snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import zstandard

from .. import bronze
from ..paths import snapshot_dir

MANIFEST = "_snapshot.json"
FAILURES = "failures.json"
PLAN = "plan.json"


class MissingRequest(RuntimeError):
    """A request the build needs is neither in the snapshot nor recorded as a failure."""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_snapshot_id(plan: dict) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{sha256_bytes(canonical_json(plan))[:8]}"


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def store_zst_atomic(path: Path, obj) -> str:
    """Compress canonical JSON of obj to path atomically; returns sha256 of the uncompressed bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_name(path.name + ".raw.tmp")
    tmp = path.with_name(path.name + ".tmp")
    raw.write_bytes(canonical_json(obj))
    try:
        digest = bronze.compress_to_zst(raw, tmp)
        os.replace(tmp, path)
    finally:
        raw.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return digest


def read_zst_json(path: Path):
    return json.loads(bronze.read_zst(path))  # bronze.read_zst closes its handle


def unwrap(obj, entry: dict | None = None) -> tuple[dict, object]:
    """-> (request, response) for both the record format and the older bare-response format."""
    if isinstance(obj, dict) and set(obj) >= {"request", "response"} and "fetched_at_utc" in obj:
        return obj["request"], obj["response"]
    entry = entry or {}
    return {"url": entry.get("url"), "params": entry.get("params")}, obj


def strict_problems(source: str, snapshot_id: str, folder: Path | None = None) -> list[str]:
    """bronze.verify (manifest hashes, unlisted files, pinned manifest sha) for standard folders; the same
    file checks for an explicit folder (tests)."""
    if folder is None or folder == snapshot_dir(source, snapshot_id):
        return bronze.verify(source, snapshot_id)
    manifest_path = folder / MANIFEST
    if not manifest_path.exists():
        return [f"{source}/{snapshot_id}: no {MANIFEST} (snapshot missing or never finalized)"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    listed = {MANIFEST}
    for entry in manifest["files"]:
        listed.add(entry["stored_as"])
        path = folder / entry["stored_as"]
        if not path.exists():
            problems.append(f"{source}/{snapshot_id}: missing {entry['stored_as']}")
        elif bronze.sha256(path) != entry["stored_sha256"]:
            problems.append(f"{source}/{snapshot_id}: checksum mismatch on {entry['stored_as']}")
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and rel not in listed:
            problems.append(f"{source}/{snapshot_id}: file on disk not listed in {MANIFEST}: {rel}")
    return problems


class Snapshot:
    """Read-only view of one sealed snapshot. Only manifest-listed files are ever opened."""

    def __init__(self, source: str, snapshot_id: str, folder: Path | None = None, verify: bool = True):
        self.source, self.snapshot_id = source, snapshot_id
        self.folder = folder or snapshot_dir(source, snapshot_id)
        if verify:
            problems = strict_problems(source, snapshot_id, self.folder)
            if problems:
                raise SystemExit("bronze does not verify:\n" + "\n".join(problems[:30]))
        self.manifest_sha256 = bronze.sha256(self.folder / MANIFEST)
        self.manifest = json.loads((self.folder / MANIFEST).read_text())
        self.entries = {e["stored_as"]: e for e in self.manifest["files"]}
        self.failures = {f["stored_as"]: f for f in self._json(FAILURES, [])}
        self.plan = self._json(PLAN, None)

    def _json(self, name: str, default):
        return json.loads((self.folder / name).read_text()) if name in self.entries else default

    def has(self, stored_as: str) -> bool:
        return stored_as in self.entries

    def record(self, stored_as: str) -> tuple[dict, object]:
        entry = self.entries[stored_as]
        return unwrap(read_zst_json(self.folder / stored_as), entry)

    def response(self, stored_as: str):
        return self.record(stored_as)[1]

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.entries if n.startswith(prefix) and n.endswith(".zst"))


class Writer:
    """Fills an open snapshot folder. Resumable: completed files are cache, *.tmp leftovers are discarded."""

    def __init__(self, source: str, snapshot_id: str):
        self.source, self.snapshot_id = source, snapshot_id
        self.folder = snapshot_dir(source, snapshot_id)
        self.failures: dict[str, dict] = {}
        self.calls = 0

    @property
    def sealed(self) -> bool:
        return (self.folder / MANIFEST).exists()

    def open(self) -> "Writer":
        self.folder.mkdir(parents=True, exist_ok=True)
        for tmp in self.folder.rglob("*.tmp"):
            tmp.unlink()
        old = self.folder / FAILURES
        if old.exists():  # a previous run's failures are retried, not inherited
            old.unlink()
        return self

    def has(self, stored_as: str) -> bool:
        return (self.folder / stored_as).exists()

    def response(self, stored_as: str):
        return unwrap(read_zst_json(self.folder / stored_as))[1]

    def store(self, stored_as: str, request: dict, response, reused_from: str | None = None,
              fetched_at: str | None = None) -> None:
        record = {"request": request, "fetched_at_utc": fetched_at or now_utc(), "response": response}
        if reused_from:
            record["reused_from"] = reused_from
        store_zst_atomic(self.folder / stored_as, record)
        self.failures.pop(stored_as, None)

    def fail(self, stored_as: str, request: dict, error: str) -> None:
        self.failures[stored_as] = {"stored_as": stored_as, "request": request, "error": error[:500]}

    def write_plan(self, plan: dict) -> None:
        write_json_atomic(self.folder / PLAN, plan)

    def finalize(self, upstream: dict, cfg: dict, allow_gaps: bool = False) -> None:
        failures = sorted(self.failures.values(), key=lambda f: f["stored_as"])
        write_json_atomic(self.folder / FAILURES, failures)
        if failures and not allow_gaps:
            raise SystemExit(f"{self.source}/{self.snapshot_id}: {len(failures)} request(s) failed (see {FAILURES}); "
                             "re-run to retry, or pass allow_gaps=True to seal the snapshot with these gaps recorded")
        files = []
        for path in sorted(self.folder.rglob("*")):
            rel = path.relative_to(self.folder).as_posix()
            if not path.is_file() or rel == MANIFEST:
                continue
            if path.suffix == ".tmp":
                raise SystemExit(f"unexpected temporary file {rel} at finalize")
            entry = {"name": rel.removesuffix(".zst"), "stored_as": rel, "compressed": rel.endswith(".zst")}
            if entry["compressed"]:
                h = hashlib.sha256()
                with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
                    while block := r.read(1 << 20):
                        h.update(block)
                entry["raw_sha256"] = h.hexdigest()
                request, _ = unwrap(read_zst_json(path))
                entry["url"], entry["params"] = request.get("url"), request.get("params")
            files.append(entry)
        upstream = {**upstream, "n_failures": len(failures), "allow_gaps": bool(allow_gaps and failures),
                    "licence": cfg["licence"], "attribution": cfg["attribution"], "redistributable": cfg["redistributable"]}
        bronze.write_manifest(self.source, self.snapshot_id, upstream, files)
        problems = strict_problems(self.source, self.snapshot_id, self.folder)
        if problems:
            raise SystemExit("snapshot does not verify right after finalize:\n" + "\n".join(problems[:20]))
