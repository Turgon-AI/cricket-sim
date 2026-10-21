"""Off-machine copy of bronze via rclone. Any rclone remote works (config/remote.yml)."""

import subprocess

from . import bronze
from .paths import BRONZE, load_yaml


def _target() -> str:
    cfg = load_yaml("remote.yml")
    remotes = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True).stdout.split()
    if f"{cfg['rclone_remote']}:" not in remotes:
        raise SystemExit(
            f"rclone remote '{cfg['rclone_remote']}' is not set up; create it with `rclone config`")
    return f"{cfg['rclone_remote']}:{cfg['remote_path']}"


def backup() -> None:
    target = _target()
    # copy, never sync: a snapshot deleted locally must survive remotely
    subprocess.run(["rclone", "copy", str(BRONZE), target, "--checksum", "--progress"], check=True)
    subprocess.run(["rclone", "check", str(BRONZE), target, "--one-way"], check=True)


def restore() -> None:
    target = _target()
    for source, snapshot_id in bronze.read_lock().items():
        if not bronze.verify(source, snapshot_id):
            continue
        dest = BRONZE / source / snapshot_id
        subprocess.run(["rclone", "copy", f"{target}/{source}/{snapshot_id}", str(dest), "--progress"], check=True)
        problems = bronze.verify(source, snapshot_id)
        print(f"{source}/{snapshot_id}: " + ("restored" if not problems else "; ".join(problems)))
