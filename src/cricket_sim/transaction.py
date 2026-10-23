"""Off-path generation and recoverable publication for local pipeline artifacts.

Writers share one advisory lock. Publication journals every destination before moving
it; interrupted commits roll back on the next pipeline command. External readers must
avoid the short publication window or acquire the same lock; multi-directory rename is
not a filesystem primitive.

Retention: every successful publish keeps the previous generation of its kind (build,
model, export) in .history/ and its stage in .pending/, so one undo is always possible,
and prunes anything older of that kind. Failed stages (no journal) and rollback
quarantines (.aborted/) are kept until `cricket prune`, which also takes --keep N.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _name(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or name in ('', '.'):
        raise ValueError(f'invalid publication destination: {name}')
    return name


def _rollback(data: Path, record: dict) -> None:
    backup = data / record['backup']
    stage = data / record['stage']
    aborted = data / '.aborted' / record['id']
    for name in reversed(record['destinations']):
        live, prior = data / name, backup / name
        if prior.exists():
            if live.exists():
                quarantine = aborted / name
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                os.replace(live, quarantine)
            live.parent.mkdir(parents=True, exist_ok=True)
            os.replace(prior, live)
        elif not record['existed'][name] and live.exists() and not (stage/name).exists():
            quarantine = aborted / name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(live, quarantine)


def pending_publications(data: Path) -> list[dict]:
    """Journals still `prepared`: a publish that was killed, or one that is moving directories right now."""
    records = [json.loads(path.read_text()) for path in sorted((data/'.transactions').glob('*.json'))]
    return [record for record in records if record['state'] == 'prepared']


def recover(data: Path) -> None:
    """Idempotently undo every publication without a durable committed marker."""
    for path in sorted((data/'.transactions').glob('*.json')):
        record = json.loads(path.read_text())
        if record['state'] == 'prepared':
            _rollback(data, record)
            record['state'] = 'rolled_back'
            _save(path, record)
    # Keep any <layer>.prev backup directories found here instead of deleting them.
    previous = [(data/name, data/(name+'.prev')) for name in ('staging', 'silver', 'gold')]
    if any(prior.exists() for _, prior in previous):
        quarantine = data/'.aborted'/('legacy-'+uuid.uuid4().hex)
        for live, prior in previous:
            if prior.exists():
                if live.exists():
                    quarantine.mkdir(parents=True, exist_ok=True)
                    os.replace(live, quarantine/live.name)
                os.replace(prior, live)


@contextmanager
def writer_lock(data: Path):
    data.mkdir(parents=True, exist_ok=True)
    with (data/'.pipeline.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit('another pipeline writer is active; retry after it finishes') from exc
        try:
            recover(data)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def new_stage(data: Path, kind: str) -> Path:
    path = data/'.pending'/(kind+'-'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


def publish(data: Path, stage: Path, names: list[str], prepare=None) -> None:
    """Caller holds writer_lock. Keep prior output recoverable through every move.

    prepare(name) may materialize a late-bound artifact, such as warehouse views,
    after its dependent tables are in their final locations but before commit.
    """
    names = [_name(n) for n in names]
    if len(names) != len(set(names)):
        raise ValueError('duplicate publication destination')
    identity = uuid.uuid4().hex
    backup = data/'.history'/identity
    record = {'id': identity, 'state': 'prepared', 'stage': str(stage.relative_to(data)),
              'backup': str(backup.relative_to(data)), 'destinations': names,
              'existed': {name: (data/name).exists() for name in names}}
    journal = data/'.transactions'/(identity+'.json')
    _save(journal, record)
    try:
        for name in names:
            if prepare is not None:
                prepare(name)
            generated, live, prior = stage/name, data/name, backup/name
            if not generated.exists():
                raise FileNotFoundError(f'publication stage missing {name}')
            live.parent.mkdir(parents=True, exist_ok=True)
            if live.exists():
                prior.parent.mkdir(parents=True, exist_ok=True)
                os.replace(live, prior)
            os.replace(generated, live)
        record['state'] = 'committed'
        _save(journal, record)
        _prune_kind(data, _kind(record), keep=1)
    except BaseException:
        _rollback(data, record)
        record['state'] = 'rolled_back'
        _save(journal, record)
        raise


def _kind(record: dict) -> str:
    """'build', 'model' or 'export': the stage name is <kind>-<uuid> under .pending/."""
    return Path(record['stage']).name.rsplit('-', 1)[0]


def _journals(data: Path) -> list[dict]:
    """Every journal, oldest first (by its own write time)."""
    paths = sorted((data/'.transactions').glob('*.json'), key=lambda p: (p.stat().st_mtime, p.name))
    return [{**json.loads(p.read_text()), '_path': p} for p in paths]


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def _remove(path: Path, freed: dict, dry_run: bool) -> None:
    """Delete one retained directory or journal file (never following a symlink out of it)."""
    if path.is_symlink() or not path.exists():
        return
    freed[str(path)] = _dir_bytes(path) if path.is_dir() else path.stat().st_size
    if not dry_run:
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def _prune_kind(data: Path, kind: str, keep: int, dry_run: bool = False, freed: dict | None = None) -> dict:
    """Caller holds writer_lock. Drop every finished generation of `kind` except the newest `keep` committed
    ones: the .history backup (the generation that publish replaced), the .pending stage and the journal.
    A rolled-back journal left nothing worth keeping, so it always goes. Prepared journals are never touched."""
    freed = {} if freed is None else freed
    finished = [r for r in _journals(data) if r['state'] != 'prepared' and _kind(r) == kind]
    committed = [r for r in finished if r['state'] == 'committed']
    protected = {r['id'] for r in committed[len(committed) - keep:]} if keep > 0 else set()
    for record in (r for r in finished if r['id'] not in protected):
        _remove(data/record['backup'], freed, dry_run)
        _remove(data/record['stage'], freed, dry_run)
        _remove(record['_path'], freed, dry_run)
    return freed


def prune(data: Path, keep: int = 1, dry_run: bool = False) -> dict:
    """`cricket prune`: free the disk that retained generations use. Keeps the newest `keep` finished
    generations of each kind (0 keeps none), then removes failed stages (a .pending/ directory no journal
    names) and rollback quarantines (.aborted/). Returns {path: bytes} of what was (or would be) removed."""
    with writer_lock(data):
        freed = {}
        for kind in sorted({_kind(r) for r in _journals(data)}):
            _prune_kind(data, kind, keep, dry_run, freed)
        referenced = {data/r['stage'] for r in _journals(data)}
        for stage in sorted((data/'.pending').glob('*')) if (data/'.pending').exists() else []:
            if stage not in referenced:
                _remove(stage, freed, dry_run)
        _remove(data/'.aborted', freed, dry_run)
        return freed


def run_worker(stage: Path, job: str, settings: dict) -> None:
    """Use a fresh interpreter so all imported data paths point to the stage."""
    env = {**os.environ, 'CRICKET_SIM_DATA': str(stage), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'cricket_sim.worker', job, json.dumps(settings)], env=env)
    if result.returncode:
        raise SystemExit(f'{job} generation failed (exit {result.returncode}); previous outputs retained at their live paths')
