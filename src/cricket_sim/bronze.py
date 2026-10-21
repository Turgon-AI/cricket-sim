"""Bronze snapshots: immutable, checksummed copies of upstream files.

A snapshot is data/bronze/<source>/<snapshot_id>/ plus _snapshot.json. Builds only ever
read the snapshot pinned in config/snapshots.lock.yml, never "latest".
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import zstandard

from .paths import BRONZE, LOCK_FILE, PINS_FILE, snapshot_dir

ZSTD_LEVEL = 10  # fixed so the same input always produces the same bytes
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def compress_to_zst(src: Path, dest: Path) -> str:
    """Compress src to dest.zst and return the sha256 of the *uncompressed* bytes."""
    h = hashlib.sha256()
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    with open(src, "rb") as fin, open(dest, "wb") as fout:
        with cctx.stream_writer(fout, closefd=False) as writer:
            while block := fin.read(CHUNK):
                h.update(block)
                writer.write(block)
    return h.hexdigest()


class ZstError(Exception):
    """A .zst file that is cut off, corrupt, or does not match what its manifest recorded."""


def _iter_zst(path: Path):
    """Yield the decompressed chunks of every frame. stream_reader().read() returns a cut-off stream without
    an error, so each frame is followed with decompressobj() and its eof flag instead."""
    dobj = None
    seen = False
    try:
        with open(path, "rb") as f:
            while block := f.read(CHUNK):
                while block:
                    seen = True
                    if dobj is None:
                        dobj = zstandard.ZstdDecompressor().decompressobj()
                    out = dobj.decompress(block)
                    if out:
                        yield out
                    if dobj.eof:
                        block, dobj = dobj.unused_data, None
                    else:
                        block = b""
    except zstandard.ZstdError as exc:
        raise ZstError(f"{path.name}: corrupt zstd data ({exc})") from exc
    if not seen or dobj is not None:
        raise ZstError(f"{path.name}: zstd stream is cut off (truncated); the last frame is incomplete")


def _manifest_entry(path: Path) -> dict | None:
    """The manifest entry for a stored file, found from the _snapshot.json in its snapshot folder."""
    for folder in list(path.parents)[:4]:
        manifest = folder / "_snapshot.json"
        if manifest.exists():
            rel = path.relative_to(folder).as_posix()
            try:
                files = json.loads(manifest.read_text())["files"]
            except (ValueError, KeyError):
                return None
            return next((e for e in files if e.get("stored_as") == rel), None)
    return None


def check_zst(path: Path, entry: dict | None = None, keep: bool = False) -> bytes:
    """Decompress the whole file. Raise ZstError if it is cut off or corrupt, or if its decompressed size or
    sha256 differ from the manifest entry (raw_bytes, raw_sha256), which are checked when the entry has them."""
    entry = _manifest_entry(path) if entry is None else entry
    digest, size, parts = hashlib.sha256(), 0, []
    for chunk in _iter_zst(path):
        digest.update(chunk)
        size += len(chunk)
        if keep:
            parts.append(chunk)
    if entry and "raw_bytes" in entry and entry["raw_bytes"] != size:
        raise ZstError(f"{path.name}: decompresses to {size} bytes, manifest raw_bytes says {entry['raw_bytes']}")
    if entry and "raw_sha256" in entry and entry["raw_sha256"] != digest.hexdigest():
        raise ZstError(f"{path.name}: decompressed sha256 differs from manifest raw_sha256")
    return b"".join(parts)


def read_zst(path: Path, entry: dict | None = None) -> bytes:
    """Decompress a .zst file written by compress_to_zst (streamed frames carry no size header).

    Stops the build (SystemExit) on a cut-off or corrupt stream, or one that disagrees with its manifest entry."""
    try:
        return check_zst(Path(path), entry, keep=True)
    except ZstError as exc:
        raise SystemExit(f"bronze: {exc}") from exc


def write_manifest(source: str, snapshot_id: str, upstream: dict, files: list[dict]) -> Path:
    folder = snapshot_dir(source, snapshot_id)
    for entry in files:
        stored = folder / entry["stored_as"]
        entry["stored_bytes"] = stored.stat().st_size
        entry["stored_sha256"] = sha256(stored)
    manifest = {
        "source": source,
        "snapshot_id": snapshot_id,
        "ingested_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upstream": upstream,
        "files": sorted(files, key=lambda e: e["stored_as"]),
    }
    path = folder / "_snapshot.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def read_manifest(source: str, snapshot_id: str) -> dict:
    return json.loads((snapshot_dir(source, snapshot_id) / "_snapshot.json").read_text())


def verify(source: str, snapshot_id: str) -> list[str]:
    """Return a list of problems; empty means the snapshot is intact."""
    folder = snapshot_dir(source, snapshot_id)
    if not (folder / "_snapshot.json").exists():
        return [f"{source}/{snapshot_id}: no _snapshot.json (snapshot missing locally?)"]
    problems = []
    listed = set()
    for entry in read_manifest(source, snapshot_id)["files"]:
        listed.add(entry["stored_as"])
        path = folder / entry["stored_as"]
        if not path.exists():
            problems.append(f"{source}/{snapshot_id}: missing {entry['stored_as']}")
        elif sha256(path) != entry["stored_sha256"]:
            problems.append(f"{source}/{snapshot_id}: checksum mismatch on {entry['stored_as']}")
        elif entry["stored_as"].endswith(".zst"):
            # the stored hash only proves the bytes are the ones recorded; a cut-off file can be recorded too
            try:
                check_zst(path, entry)
            except ZstError as exc:
                problems.append(f"{source}/{snapshot_id}: {exc}")
    # anything on disk that the manifest does not list could be read by a glob and trusted silently
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and rel != "_snapshot.json" and rel not in listed:
            problems.append(f"{source}/{snapshot_id}: unlisted file on disk: {rel}")
    pins = read_pins()
    if read_lock().get(source) == snapshot_id:
        if source not in pins:
            problems.append(f"{source}/{snapshot_id}: no manifest hash pin for locked snapshot")
        elif pins[source] != manifest_sha(source, snapshot_id):
            problems.append(f"{source}/{snapshot_id}: _snapshot.json differs from the one that was locked")
    return problems


def manifest_sha(source: str, snapshot_id: str) -> str:
    return sha256(snapshot_dir(source, snapshot_id) / "_snapshot.json")


def read_pins() -> dict:
    """sha256 of each locked snapshot's _snapshot.json: the lock pins content, not just a folder name."""
    if not PINS_FILE.exists():
        return {}
    return yaml.safe_load(PINS_FILE.read_text()) or {}


def read_lock() -> dict:
    if not LOCK_FILE.exists():
        return {}
    return yaml.safe_load(LOCK_FILE.read_text()) or {}


# every source a build reads; a lock that lacks one is cut off or the wrong file
REQUIRED_SOURCES = ("cricsheet", "ari", "hawkeye", "cricsheet_missing", "wikidata_players",
                    "venues_geo", "weather", "wiki_fixtures")


def _strict_yaml(path: Path, what: str) -> tuple[dict, list[str]]:
    """Load a lock or pins file, reporting a missing, unreadable or empty file instead of returning {}."""
    if not path.exists():
        return {}, [f"{what} file missing: {path}"]
    try:
        value = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return {}, [f"{what} file cannot be parsed (cut off or damaged?): {path}"]
    if not isinstance(value, dict) or not value:
        return {}, [f"{what} file is empty: {path}"]
    return value, []


def verify_all() -> list[str]:
    """Problems with the lock, the pins and every locked snapshot. Empty means a build can trust bronze.

    read_lock() and read_pins() return {} for a missing file; that is right for a first ingest but must not
    read as "nothing to check" here."""
    lock, problems = _strict_yaml(LOCK_FILE, "lock")
    _, pin_problems = _strict_yaml(PINS_FILE, "pins")
    problems += pin_problems
    if problems and not lock:
        return problems
    absent = [s for s in REQUIRED_SOURCES if s not in lock]
    if absent:
        problems.append(f"lock file lists {len(lock)} sources but {len(absent)} of {len(REQUIRED_SOURCES)} "
                        f"required sources are missing: {absent}")
    if any("cannot be parsed" in p for p in problems):
        return problems
    return problems + [p for s, sid in lock.items() for p in verify(s, sid)]


def write_lock(lock: dict) -> None:
    header = "# Snapshots that builds read. Change deliberately (cricket lock <source> <id>), then commit.\n"
    LOCK_FILE.write_text(header + yaml.safe_dump(dict(sorted(lock.items())), sort_keys=True))


def lock_source(source: str, snapshot_id: str) -> None:
    if verify(source, snapshot_id):
        raise SystemExit(f"Refusing to lock {source}/{snapshot_id}: it does not verify")
    lock = read_lock()
    lock[source] = snapshot_id
    write_lock(lock)
    pins = read_pins()
    pins[source] = manifest_sha(source, snapshot_id)
    PINS_FILE.write_text("# sha256 of each locked snapshot's _snapshot.json (written by `cricket lock`).\n"
                         + yaml.safe_dump(dict(sorted(pins.items())), sort_keys=True))


def discard_partial(folder: Path) -> None:
    if folder.exists() and not (folder / "_snapshot.json").exists():
        shutil.rmtree(folder)


def list_snapshots(source: str) -> list[str]:
    base = BRONZE / source
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "_snapshot.json").exists())
