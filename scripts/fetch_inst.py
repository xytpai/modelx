#!/usr/bin/env python3
"""Fetch separately versioned instruction data; install verified copies, not symlinks.

Python standard library only. Git is needed for network access, not --source/--verify.
Downloaded files live under data/inst (gitignored), with their own MD5SUMS.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "git@github.com:xytpai/datax-inst.git"
STATE = ".datax-inst.json"
MAX_BYTES = 50_000_000


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(name: str) -> str:
    parts = name.split("/")
    if (not name or "\\" in name or ":" in name
            or any(p in {"", ".", ".."} for p in parts)
            or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ValueError(f"Unsafe manifest path: {name!r}")
    return str(PurePosixPath(name))


def checked_file(base: Path, name: str) -> Path:
    name = safe_name(name)
    path = base
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed: {path}")
    if not path.is_file():
        raise ValueError(f"Missing regular file: {path}")
    return path


def read_manifest(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ValueError(f"Symlink manifest: {path}")
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{32})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid MD5SUMS line: {line!r}")
        value, name = match.groups()
        name = safe_name(name)
        if name in entries:
            raise ValueError(f"Duplicate manifest path: {name}")
        entries[name] = value
    if not entries:
        raise ValueError("Empty checksum manifest")
    return entries


def manifest_text(entries: dict[str, str]) -> str:
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))


def installed_name(source_name: str) -> str:
    name = safe_name(source_name)
    if name.startswith("data/") and name.endswith(".jsonl"):
        return safe_name(name[5:])
    if name.startswith("LICENSES/"):
        return name
    raise ValueError(f"Unexpected data repository manifest entry: {name}")


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"Import lock exists: {path}. If a previous process crashed, confirm "
            "no import is running before removing this lock."
        ) from None
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def tree_files(directory: Path) -> set[str]:
    files = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in managed directory: {path}")
        if path.is_file():
            files.add(path.relative_to(directory).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Non-regular file in managed directory: {path}")
    return files


def verify_install(destination: Path) -> dict:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"Not an installed data directory: {destination}")
    state_path = checked_file(destination, STATE)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("version") != 1 or not isinstance(state.get("files"), dict):
        raise ValueError(f"Invalid import state: {state_path}")
    entries = state["files"]
    if not entries:
        raise ValueError("Empty installed manifest")
    for name, value in entries.items():
        safe_name(name)
        if name in {STATE, "MD5SUMS"} or not (
                name.endswith(".jsonl") or name.startswith("LICENSES/")):
            raise ValueError(f"Invalid installed path: {name}")
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{32}", value):
            raise ValueError(f"Invalid installed digest: {name}")
        path = checked_file(destination, name)
        if md5(path) != value:
            raise ValueError(f"Locally modified file; refusing overwrite: {path}")
    if tree_files(destination) != set(entries) | {"MD5SUMS", STATE}:
        raise ValueError("Unmanaged or missing files in destination; refusing overwrite")
    if read_manifest(destination / "MD5SUMS") != entries:
        raise ValueError("Installed MD5SUMS differs from import state")
    return state


def validate_jsonl(path: Path, ids: set[str]) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {
                    "id", "language", "meta", "messages"}:
                raise ValueError(f"Invalid datax record in {path}:{count + 1}")
            rid = record["id"]
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError(f"Empty/duplicate record ID in {path}:{count + 1}")
            if not isinstance(record["language"], str) or not record["language"]:
                raise ValueError(f"Invalid language: {rid}")
            meta = record["meta"]
            if not isinstance(meta, dict) or set(meta) != {"category", "topic"}:
                raise ValueError(f"Expected category/topic-only metadata: {rid}")
            if not all(isinstance(value, str) for value in meta.values()):
                raise ValueError(f"Invalid metadata strings: {rid}")
            messages = record["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Missing messages: {rid}")
            for msg in messages:
                if (not isinstance(msg, dict) or set(msg) != {"role", "content"}
                        or msg["role"] not in {"system", "user", "assistant"}
                        or not isinstance(msg["content"], str)
                        or not msg["content"].strip()):
                    raise ValueError(f"Invalid message: {rid}")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"Conversation has no final answer: {rid}")
            ids.add(rid)
            count += 1
    if not count:
        raise ValueError(f"Empty data shard: {path}")
    return count


def git(args: list[str], env: dict | None = None, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def fetch_repo(url: str, branch: str, cache: Path, env: dict) -> Path:
    if not branch or branch.startswith("-"):
        raise ValueError("Invalid branch")
    cache = cache.absolute()
    if cache.is_symlink():
        raise ValueError("Cache must not be a symlink")
    with exclusive_lock(cache.parent / (cache.name + ".fetch.lock")):
        if cache.exists():
            if not (cache / ".git").is_dir():
                raise ValueError(f"Not a Git cache: {cache}; use --source for a local dataset")
            if git(["status", "--porcelain", "--untracked-files=all"], env, cache):
                raise ValueError(f"Dirty cache, refusing to reset it: {cache}")
            if git(["remote", "get-url", "origin"], env, cache) != url:
                raise ValueError("Cache origin differs from --repo; choose another --cache-dir")
            if git(["branch", "--show-current"], env, cache) != branch:
                raise ValueError("Cache branch differs; choose another --cache-dir")
            git(["fetch", "--depth=1", "origin", branch], env, cache)
            current = git(["rev-parse", "HEAD"], env, cache)
            desired = git(["rev-parse", "FETCH_HEAD"], env, cache)
            if current != desired:
                # A shallow fetch may hide the ancestor; deepen without resetting local commits.
                git(["fetch", "--deepen=100", "origin", branch], env, cache)
                git(["merge", "--ff-only", "FETCH_HEAD"], env, cache)
                if git(["rev-parse", "HEAD"], env, cache) != git(["rev-parse", "FETCH_HEAD"], env, cache):
                    raise ValueError("Cache has local commits; use --source explicitly or another --cache-dir")
        else:
            staging = Path(tempfile.mkdtemp(prefix="inst-clone-", dir=cache.parent))
            try:
                git(["clone", "--depth=1", "--single-branch", "--branch", branch,
                     "--", url, str(staging / "repo")], env)
                os.rename(staging / "repo", cache)
            finally:
                shutil.rmtree(staging)
    return cache


def install(source: Path, destination: Path, pattern: str = "*.jsonl",
            repository: str = "local", revision: str = "local") -> dict:
    source = source.absolute()
    destination = destination.absolute()
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Source/destination must not be symlinks")
    resolved_source, resolved_dest = source.resolve(), destination.resolve()
    if (resolved_source == resolved_dest or resolved_source in resolved_dest.parents
            or resolved_dest in resolved_source.parents):
        raise ValueError("Source and destination must not overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(destination.parent / ("." + destination.name + ".import.lock")):
        previous = None
        if destination.exists():
            if not destination.is_dir():
                raise ValueError("Destination is not a directory")
            if any(destination.iterdir()):
                previous = verify_install(destination)
        entries = read_manifest(source / "MD5SUMS")
        available = {name for name in tree_files(source / "data")}
        listed = {name[5:] for name in entries if name.startswith("data/")}
        if available != listed:
            raise ValueError("Source data files and manifest differ")
        selected = {}
        for name, value in entries.items():
            local_name = installed_name(name)
            if name.startswith("data/") and not fnmatch.fnmatchcase(local_name, pattern):
                continue
            if local_name in selected:
                raise ValueError(f"Destination mapping collision: {local_name}")
            path = checked_file(source, name)
            if name.startswith("data/") and path.stat().st_size >= MAX_BYTES:
                raise ValueError(f"Data file must be smaller than 50 MB: {path}")
            if md5(path) != value:
                raise ValueError(f"Source checksum mismatch: {path}")
            selected[local_name] = (name, value)
        if not any(name.endswith(".jsonl") for name in selected):
            raise ValueError(f"No JSONL files match {pattern!r}")
        if not any(name.startswith("LICENSES/") for name in selected):
            raise ValueError("Source license notice missing")
        installed = {name: value for name, (_, value) in selected.items()}
        if previous is not None and previous["files"] == installed:
            refreshed = dict(previous, repository=repository, revision=revision, pattern=pattern)
            if refreshed != previous:
                fd, temporary = tempfile.mkstemp(prefix=".inst-state-", dir=destination.parent)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(json.dumps(refreshed, indent=2) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination / STATE)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            return dict(refreshed, unchanged=True, destination=str(destination))
        stage = Path(tempfile.mkdtemp(prefix="." + destination.name + ".stage-",
                                     dir=destination.parent))
        backup = None
        try:
            ids: set[str] = set()
            records = 0
            for local_name, (source_name, value) in sorted(selected.items()):
                original = checked_file(source, source_name)
                target = stage / local_name
                target.parent.mkdir(parents=True, exist_ok=True)
                # Copies isolate source checkout and installed files from each other's edits.
                shutil.copyfile(original, target)
                if md5(target) != value:
                    raise ValueError(f"Copy/source changed during import: {original}")
                if local_name.endswith(".jsonl"):
                    records += validate_jsonl(target, ids)
            (stage / "MD5SUMS").write_text(manifest_text(installed), encoding="utf-8")
            state = {"version": 1, "repository": repository, "revision": revision,
                     "pattern": pattern, "records": records, "files": installed}
            (stage / STATE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            verify_install(stage)
            # Recheck modifications made while the copies were being staged.
            if previous is not None:
                if verify_install(destination) != previous:
                    raise ValueError("Destination state changed during import")
            elif destination.exists() and any(destination.iterdir()):
                raise ValueError("Destination changed during import")
            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix="." + destination.name + ".old-",
                                              dir=destination.parent))
                backup.rmdir()
                os.rename(destination, backup)
            try:
                os.rename(stage, destination)
            except BaseException:
                if backup is not None:
                    os.rename(backup, destination)
                    backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup)
            return dict(state, unchanged=False, destination=str(destination))
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Git URL (default: GitHub SSH)")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/datax-inst",
                        help="Managed Git cache; updates are clean and fast-forward only")
    parser.add_argument("--source", type=Path, help="Offline local checkout; no Git commands or updates")
    parser.add_argument("--dest", type=Path, default=ROOT / "data/inst")
    parser.add_argument("--pattern", default="*.jsonl",
                        help="Relative data-file glob, e.g. 'tulu_3_sft_en_*.jsonl'")
    parser.add_argument("--ssh-key", type=Path, help="Existing SSH private key path; never copied")
    parser.add_argument("--known-hosts", type=Path,
                        help="Trusted SSH known_hosts file (default: next to --ssh-key, if present)")
    parser.add_argument("--verify", action="store_true", help="Only verify installed data, offline")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            state = verify_install(args.dest)
            print(f"Verified {state['records']:,} records, {len(state['files'])} files: {args.dest}")
            return
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        if args.ssh_key is not None:
            key = args.ssh_key.expanduser().resolve()
            if not key.is_file():
                raise ValueError(f"SSH key file not found: {key}")
            env["GIT_SSH_COMMAND"] = (
                "ssh -i " + shlex.quote(str(key))
                + " -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes")
            hosts = args.known_hosts or (key.parent / "known_hosts")
            if args.known_hosts is not None and not hosts.expanduser().is_file():
                raise ValueError(f"Known-hosts file not found: {hosts}")
            if hosts.expanduser().is_file():
                env["GIT_SSH_COMMAND"] += " -o " + shlex.quote(
                    "UserKnownHostsFile=" + str(hosts.expanduser().resolve()))
        elif args.known_hosts is not None:
            hosts = args.known_hosts.expanduser().resolve()
            if not hosts.is_file():
                raise ValueError(f"Known-hosts file not found: {hosts}")
            env["GIT_SSH_COMMAND"] = ("ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o "
                                      + shlex.quote("UserKnownHostsFile=" + str(hosts)))
        if args.source is not None:
            source = args.source.expanduser()
            repository, revision = str(source), "local-verified-manifest"
        else:
            args.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            source = fetch_repo(args.repo, args.branch, args.cache_dir, env)
            repository, revision = args.repo, git(["rev-parse", "HEAD"], env, source)
        result = install(source, args.dest, args.pattern, repository, revision)
        print(json.dumps({"destination": result["destination"], "records": result["records"],
                          "files": len(result["files"]), "unchanged": result["unchanged"],
                          "revision": result["revision"]}, indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
