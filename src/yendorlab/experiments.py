"""SQLite run history; historical M0 rows remain readable."""

import json
import hashlib
import platform
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def source_fingerprint():
    """Hash runnable source and configs, including currently untracked files."""
    paths = []
    for directory in (Path("src/yendorlab"), Path("configs")):
        if directory.is_dir():
            paths.extend(p for p in directory.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts)
    patch = Path("native/patches/0001-yendorlab-stream.patch")
    if patch.is_file():
        paths.append(patch)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def record(data_dir: Path, experiment: str, config: dict, result: dict):
    data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(data_dir / "experiments.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, experiment_id TEXT, "
               "started_utc TEXT, config_json TEXT, result_json TEXT)")
    run_id = str(uuid.uuid4())
    config = {**config, "yendorlab_revision": revision(),
              "brogue_revision": "1ba4240b7a928ddf0ffb772717bf1d433cd63804",
              "hardware": platform.platform(), "protocol": 2,
              "observation_schema": 2, "action_schema": 2}
    # The Windows checkout is CRLF; WSL Git needs the same normalization.
    unstaged = subprocess.run(["git", "-c", "core.autocrlf=true", "diff", "--quiet"],
                              capture_output=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               capture_output=True, text=True)
    config["worktree_dirty"] = (bool(unstaged.returncode or staged.returncode or
                                     untracked.stdout.strip()) if untracked.returncode == 0 else None)
    config["source_tree_sha256"] = source_fingerprint()
    patch = Path("native/patches/0001-yendorlab-stream.patch")
    config["native_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest() if patch.is_file() else None
    db.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
               (run_id, experiment, datetime.now(timezone.utc).isoformat(),
                json.dumps(config), json.dumps(result)))
    db.commit()
    db.close()
    return run_id


def list_runs(data_dir: Path):
    db = sqlite3.connect(data_dir / "experiments.sqlite3")
    rows = db.execute("SELECT run_id, experiment_id, started_utc FROM runs ORDER BY started_utc DESC").fetchall()
    db.close()
    return rows


def show_run(data_dir: Path, run_id: str):
    db = sqlite3.connect(data_dir / "experiments.sqlite3")
    row = db.execute("SELECT experiment_id, started_utc, config_json, result_json FROM runs WHERE run_id=?",
                     (run_id,)).fetchone()
    db.close()
    if row is None:
        raise KeyError(run_id)
    return {"experiment": row[0], "started_utc": row[1],
            "config": json.loads(row[2]), "result": json.loads(row[3])}
