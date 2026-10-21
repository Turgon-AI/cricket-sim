"""Off-path generation and recoverable publication for local pipeline artifacts.

Writers share one advisory lock. Publication journals every destination before moving
it; interrupted commits roll back on the next pipeline command. Original generations
and failed stages are retained. External readers must avoid the short publication
window or acquire the same lock; multi-directory rename is not a filesystem primitive.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
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
    except BaseException:
        _rollback(data, record)
        record['state'] = 'rolled_back'
        _save(journal, record)
        raise


def run_worker(stage: Path, job: str, settings: dict) -> None:
    """Use a fresh interpreter so all imported data paths point to the stage."""
    env = {**os.environ, 'CRICKET_SIM_DATA': str(stage), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-m', 'cricket_sim.worker', job, json.dumps(settings)], env=env)
    if result.returncode:
        raise SystemExit(f'{job} generation failed (exit {result.returncode}); previous outputs retained at their live paths')
