"""Fetch upstream sources into new bronze snapshots. Never modifies an existing snapshot."""

import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from . import bronze
from .paths import load_yaml, snapshot_dir

HEADERS = {"User-Agent": "cricket-sim research (non-commercial)"}


def _download(url: str, dest: Path) -> None:
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)


def _start(source: str, snapshot_id: str) -> Path | None:
    if snapshot_id in bronze.list_snapshots(source):
        print(f"{source}/{snapshot_id} already exists; nothing to do")
        return None
    folder = snapshot_dir(source, snapshot_id)
    bronze.discard_partial(folder)
    folder.mkdir(parents=True)
    return folder


def _finish(source: str, snapshot_id: str, upstream: dict, files: list[dict]) -> str:
    bronze.write_manifest(source, snapshot_id, upstream, files)
    if source not in bronze.read_lock():
        bronze.lock_source(source, snapshot_id)
        print(f"locked {source} -> {snapshot_id} (first snapshot)")
    else:
        print(f"new snapshot {source}/{snapshot_id}; run `cricket lock {source} {snapshot_id}` to build from it")
    return snapshot_id


def ingest_cricsheet() -> str:
    """Snapshot id = upstream date + hash of the downloaded bytes, so a same-day update or a
    register-only change is never skipped as 'already exists'."""
    cfg = load_yaml("sources.yml")["cricsheet"]
    head = requests.head(cfg["files"]["all_json.zip"], headers=HEADERS, timeout=60)
    head.raise_for_status()
    last_modified = parsedate_to_datetime(head.headers["Last-Modified"])
    with tempfile.TemporaryDirectory(dir=bronze.BRONZE.parent) as tmp:
        combined = hashlib.sha256()
        for name, url in cfg["files"].items():
            _download(url, Path(tmp) / name)
            combined.update(bronze.sha256(Path(tmp) / name).encode())
        snapshot_id = f"{last_modified:%Y-%m-%d}-{combined.hexdigest()[:8]}"
        folder = _start("cricsheet", snapshot_id)
        if folder is None:
            return snapshot_id
        files = []
        for name, url in cfg["files"].items():
            shutil.move(Path(tmp) / name, folder / name)
            files.append({"name": name, "url": url, "stored_as": name, "compressed": False})
    upstream = {"all_json_last_modified": last_modified.isoformat()}
    return _finish("cricsheet", snapshot_id, upstream, files)


def ingest_ari() -> str:
    import kagglehub

    cfg = load_yaml("sources.yml")["ari"]
    snapshot_id = f"v{cfg['kaggle_version']}"
    folder = _start("ari", snapshot_id)
    if folder is None:
        return snapshot_id
    handle = f"{cfg['kaggle_handle']}/versions/{cfg['kaggle_version']}"
    files = []
    for name in cfg["files"]:
        cached = Path(kagglehub.dataset_download(handle, path=name))
        stored_as = name + ".zst" if name.endswith(".csv") else name
        if stored_as.endswith(".zst"):
            raw_sha = bronze.compress_to_zst(cached, folder / stored_as)
        else:
            shutil.copyfile(cached, folder / stored_as)
            raw_sha = bronze.sha256(cached)
        files.append({"name": name, "stored_as": stored_as, "compressed": stored_as.endswith(".zst"),
                      "raw_bytes": cached.stat().st_size, "raw_sha256": raw_sha})
        cached.unlink()  # keep disk usage down; the compressed copy is the record
        print(f"  ari: {name} stored")
    _check_against_ari_manifest(folder, files)
    return _finish("ari", snapshot_id, {"kaggle_handle": handle}, files)


def _check_against_ari_manifest(folder: Path, files: list[dict]) -> None:
    import json

    published = {f["name"]: f for f in json.loads((folder / "manifest.json").read_text())["files"]}
    for entry in files:
        expected = published.get(entry["name"])
        if expected and expected["sha256"] != entry["raw_sha256"]:
            raise SystemExit(f"ari: {entry['name']} does not match the author's published checksum")


def ingest_hawkeye() -> str:
    cfg = load_yaml("sources.yml")["hawkeye"]
    snapshot_id = cfg["commit"][:7]
    folder = _start("hawkeye", snapshot_id)
    if folder is None:
        return snapshot_id
    files = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in cfg["files"]:
            url = f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['commit']}/{name}"
            raw = Path(tmp) / name
            _download(url, raw)
            raw_sha = bronze.compress_to_zst(raw, folder / f"{name}.zst")
            files.append({"name": name, "url": url, "stored_as": f"{name}.zst", "compressed": True,
                          "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha})
    return _finish("hawkeye", snapshot_id, {"repo": cfg["repo"], "commit": cfg["commit"]}, files)


def ingest_cricsheet_missing() -> str:
    cfg = load_yaml("sources.yml")["cricsheet_missing"]
    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = _start("cricsheet_missing", snapshot_id)
    if folder is None:
        return snapshot_id
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "missing.html"
        _download(cfg["url"], raw)
        raw_sha = bronze.compress_to_zst(raw, folder / "missing.html.zst")
        files = [{"name": "missing.html", "url": cfg["url"], "stored_as": "missing.html.zst", "compressed": True,
                  "raw_bytes": raw.stat().st_size, "raw_sha256": raw_sha}]
    return _finish("cricsheet_missing", snapshot_id, {"url": cfg["url"]}, files)


INGESTERS = {"cricsheet": ingest_cricsheet, "ari": ingest_ari, "hawkeye": ingest_hawkeye,
             "cricsheet_missing": ingest_cricsheet_missing}


def _context_ingesters():
    from .context_players import ingest_wikidata_players
    from .context_venues import ingest_venues_and_weather
    from .wiki_fixtures import ingest_wiki_fixtures
    return {"wikidata_players": ingest_wikidata_players, "venues_and_weather": ingest_venues_and_weather,
            "wiki_fixtures": ingest_wiki_fixtures}


INGESTERS.update(_context_ingesters())
