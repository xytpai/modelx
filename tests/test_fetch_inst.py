"""Offline fixtures: safe, verified imports without network or huge datasets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_inst.py"
spec = importlib.util.spec_from_file_location("fetch_inst", SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def write_dataset(root, names=("en.jsonl", "other.jsonl")):
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "LICENSES").mkdir(exist_ok=True)
    entries = {}
    for index, name in enumerate(names):
        record = {"id": name, "language": "en", "meta": {"category": "test", "topic": "source"},
                  "messages": [{"role": "user", "content": f"question {index}"},
                               {"role": "assistant", "content": "answer"}]}
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n")
        entries[f"data/{name}"] = fetch.md5(path)
    path = root / "LICENSES/source.txt"
    path.write_text("Test license notice\n")
    entries["LICENSES/source.txt"] = fetch.md5(path)
    (root / "MD5SUMS").write_text(fetch.manifest_text(entries))
    return root


def refresh_manifest(source):
    entries = {}
    for directory in ["data", "LICENSES"]:
        for p in (source / directory).rglob("*"):
            if p.is_file():
                entries[p.relative_to(source).as_posix()] = fetch.md5(p)
    (source / "MD5SUMS").write_text(fetch.manifest_text(entries))


def test_import_metadata_messages_copy_checksums_idempotency_and_loader_discovery(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "data/inst"
    result = fetch.install(src, dest)
    assert result["records"] == 2
    assert result["unchanged"] is False
    assert fetch.verify_install(dest)["records"] == 2
    assert (dest / "en.jsonl").read_bytes() == (src / "data/en.jsonl").read_bytes()
    assert (dest / "en.jsonl").stat().st_ino != (src / "data/en.jsonl").stat().st_ino
    assert (dest / "LICENSES/source.txt").is_file()
    before = (dest / "en.jsonl").stat().st_mtime_ns
    assert fetch.install(src, dest)["unchanged"]
    assert (dest / "en.jsonl").stat().st_mtime_ns == before
    from dataloader.utils import find_jsonl_files
    assert len(find_jsonl_files(tmp_path / "data")) == 2
    assert (dest / "MD5SUMS").read_text().splitlines() == sorted(
        (dest / "MD5SUMS").read_text().splitlines(), key=lambda s: s.split("  ", 1)[1])


def test_pattern_retains_license(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    result = fetch.install(src, dest, "en*.jsonl")
    assert result["records"] == 1
    assert (dest / "LICENSES/source.txt").exists()
    assert not (dest / "other.jsonl").exists()


def test_modified_destination_and_unmanaged_files_not_overwritten(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    fetch.install(src, dest)
    target = dest / "en.jsonl"
    target.write_text("local user changes")
    with pytest.raises(ValueError, match="Locally modified"):
        fetch.install(src, dest)
    assert target.read_text() == "local user changes"
    target.write_bytes((src / "data/en.jsonl").read_bytes())
    (dest / "notes.txt").write_text("my notes")
    with pytest.raises(ValueError, match="Unmanaged"):
        fetch.install(src, dest)
    assert (dest / "notes.txt").read_text() == "my notes"


def test_source_corruption_keeps_existing_install(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    fetch.install(src, dest)
    previous = (dest / "MD5SUMS").read_bytes()
    (src / "data/en.jsonl").write_text("corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch.install(src, dest)
    assert (dest / "MD5SUMS").read_bytes() == previous
    fetch.verify_install(dest)


def test_manifest_valid_but_bad_json_not_published(tmp_path):
    src = write_dataset(tmp_path / "src")
    (src / "data/en.jsonl").write_text("not JSON\n")
    refresh_manifest(src)
    dest = tmp_path / "inst"
    with pytest.raises(ValueError):
        fetch.install(src, dest)
    assert not dest.exists()
    assert not list(tmp_path.glob(".inst.stage-*"))
    assert not (tmp_path / ".inst.import.lock").exists()


def test_update_add_remove_and_empty_directory(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    dest.mkdir()
    fetch.install(src, dest)
    (src / "data/other.jsonl").unlink()
    refresh_manifest(src)
    result = fetch.install(src, dest)
    assert result["records"] == 1
    assert not (dest / "other.jsonl").exists()
    assert fetch.verify_install(dest)["records"] == 1


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "data/../escape",
                                  "data//evil", "data/./evil", "data\\evil",
                                  "C:/evil", "data/file\nname", ""])
def test_bad_paths(name):
    with pytest.raises(ValueError):
        fetch.safe_name(name)


def test_duplicate_manifest_rejected(tmp_path):
    src = write_dataset(tmp_path / "src")
    line = (src / "MD5SUMS").read_text().splitlines()[0]
    with (src / "MD5SUMS").open("a") as f:
        f.write(line + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        fetch.install(src, tmp_path / "inst")


def test_symlink_source_and_destination_rejected(tmp_path):
    src = write_dataset(tmp_path / "src")
    shard = src / "data/en.jsonl"
    outside = tmp_path / "outside"
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        fetch.install(src, tmp_path / "inst")
    dest = tmp_path / "linked"
    dest.symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        fetch.install(src, dest)


def test_missing_license_and_no_matches(tmp_path):
    src = write_dataset(tmp_path / "src")
    with pytest.raises(ValueError, match="No JSONL"):
        fetch.install(src, tmp_path / "inst", "absent*.jsonl")
    (src / "LICENSES/source.txt").unlink()
    refresh_manifest(src)
    with pytest.raises(ValueError, match="license notice missing"):
        fetch.install(src, tmp_path / "inst")


def test_duplicate_ids_and_size_limit(tmp_path, monkeypatch):
    src = write_dataset(tmp_path / "src")
    monkeypatch.setattr(fetch, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="50 MB"):
        fetch.install(src, tmp_path / "inst")
    monkeypatch.setattr(fetch, "MAX_BYTES", 50_000_000)
    (src / "data/other.jsonl").write_bytes((src / "data/en.jsonl").read_bytes())
    refresh_manifest(src)
    with pytest.raises(ValueError, match="duplicate"):
        fetch.install(src, tmp_path / "inst")


def test_local_command_and_verify_are_offline(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    subprocess.run(["python3", str(SCRIPT), "--source", str(src), "--dest", str(dest)],
                   check=True, capture_output=True, text=True, cwd=tmp_path)
    result = subprocess.run(["python3", str(SCRIPT), "--verify", "--dest", str(dest)],
                            check=True, capture_output=True, text=True, cwd=tmp_path)
    assert "Verified 2 records" in result.stdout


def test_locks_and_overlapping_directories(tmp_path):
    src = write_dataset(tmp_path / "src")
    lock = tmp_path / ".inst.import.lock"
    lock.write_text("other process")
    with pytest.raises(ValueError, match="lock exists"):
        fetch.install(src, tmp_path / "inst")
    with pytest.raises(ValueError, match="overlap"):
        fetch.install(src, src / "nested")


def test_source_unlisted_data_rejected(tmp_path):
    src = write_dataset(tmp_path / "src")
    (src / "data/new.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="files and manifest differ"):
        fetch.install(src, tmp_path / "inst")


def test_git_fetch_clone_update_and_dirty_cache_protection(tmp_path):
    import os
    src = write_dataset(tmp_path / "repo")
    def git(*args):
        return subprocess.run(["git", *args], cwd=src, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("add", ".")
    git("commit", "-m", "first")
    cache = tmp_path / "cache"
    env = os.environ.copy()
    url = src.as_uri()
    fetch.fetch_repo(url, "main", cache, env)
    assert (cache / "MD5SUMS").read_bytes() == (src / "MD5SUMS").read_bytes()
    (src / "data/other.jsonl").unlink()
    refresh_manifest(src)
    git("add", "-A")
    git("commit", "-m", "second")
    fetch.fetch_repo(url, "main", cache, env)
    assert not (cache / "data/other.jsonl").exists()
    (cache / "mine.txt").write_text("do not discard")
    with pytest.raises(ValueError, match="Dirty cache"):
        fetch.fetch_repo(url, "main", cache, env)
    assert (cache / "mine.txt").read_text() == "do not discard"


def test_invalid_new_version_preserves_previous_install(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    fetch.install(src, dest)
    before = {p.relative_to(dest).as_posix(): p.read_bytes()
              for p in dest.rglob("*") if p.is_file()}
    (src / "data/en.jsonl").write_text("not valid JSON")
    refresh_manifest(src)
    with pytest.raises(ValueError):
        fetch.install(src, dest)
    after = {p.relative_to(dest).as_posix(): p.read_bytes()
             for p in dest.rglob("*") if p.is_file()}
    assert after == before
    assert not list(tmp_path.glob(".inst.stage-*"))
    fetch.verify_install(dest)


def test_ssh_key_uses_trusted_sibling_known_hosts(tmp_path, monkeypatch, capsys):
    src = write_dataset(tmp_path / "src")
    key = tmp_path / "ssh dir/id_test"
    key.parent.mkdir()
    key.write_text("dummy fixture, never an actual private key")
    hosts = key.parent / "known_hosts"
    hosts.write_text("trusted test fixture")
    seen = {}
    def fake_fetch(url, branch, cache, env):
        seen.update(env)
        return src
    monkeypatch.setattr(fetch, "fetch_repo", fake_fetch)
    monkeypatch.setattr(fetch, "git", lambda *args: "test-revision")
    fetch.main(["--ssh-key", str(key), "--cache-dir", str(tmp_path / "cache"),
                "--dest", str(tmp_path / "inst")])
    import shlex
    args = shlex.split(seen["GIT_SSH_COMMAND"])
    assert str(key) in args
    assert "StrictHostKeyChecking=yes" in args
    assert "UserKnownHostsFile=" + str(hosts) in args
    assert "test-revision" in capsys.readouterr().out


def test_git_cache_never_discards_local_commits(tmp_path):
    import os
    src = write_dataset(tmp_path / "repo")
    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True,
                              capture_output=True, text=True).stdout.strip()
    git(src, "init", "-b", "main")
    git(src, "config", "user.email", "test@example.invalid")
    git(src, "config", "user.name", "Test")
    git(src, "add", ".")
    git(src, "commit", "-m", "initial")
    cache = tmp_path / "cache"
    env = os.environ.copy()
    fetch.fetch_repo(src.as_uri(), "main", cache, env)
    git(cache, "config", "user.email", "test@example.invalid")
    git(cache, "config", "user.name", "Test")
    (cache / "local.txt").write_text("do not discard")
    git(cache, "add", "local.txt")
    git(cache, "commit", "-m", "local")
    original_commit = git(cache, "rev-parse", "HEAD")
    (src / "remote.txt").write_text("remote update")
    git(src, "add", "remote.txt")
    git(src, "commit", "-m", "remote")
    with pytest.raises(ValueError):
        fetch.fetch_repo(src.as_uri(), "main", cache, env)
    assert git(cache, "rev-parse", "HEAD") == original_commit
    assert (cache / "local.txt").read_text() == "do not discard"


def test_git_cache_local_ahead_is_not_mistaken_for_remote(tmp_path):
    import os
    src = write_dataset(tmp_path / "repo")
    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True,
                              capture_output=True, text=True).stdout.strip()
    git(src, "init", "-b", "main")
    git(src, "config", "user.email", "test@example.invalid")
    git(src, "config", "user.name", "Test")
    git(src, "add", ".")
    git(src, "commit", "-m", "initial")
    cache = tmp_path / "cache"
    fetch.fetch_repo(src.as_uri(), "main", cache, os.environ.copy())
    git(cache, "config", "user.email", "test@example.invalid")
    git(cache, "config", "user.name", "Test")
    (cache / "local.txt").write_text("local")
    git(cache, "add", "local.txt")
    git(cache, "commit", "-m", "local-only")
    commit = git(cache, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="local commits"):
        fetch.fetch_repo(src.as_uri(), "main", cache, os.environ.copy())
    assert git(cache, "rev-parse", "HEAD") == commit


def test_identical_data_refreshes_revision_without_copying(tmp_path):
    src = write_dataset(tmp_path / "src")
    dest = tmp_path / "inst"
    fetch.install(src, dest, repository="local", revision="local")
    inode = (dest / "en.jsonl").stat().st_ino
    result = fetch.install(src, dest, repository="remote", revision="012345")
    assert result["unchanged"]
    assert (dest / "en.jsonl").stat().st_ino == inode
    state = fetch.verify_install(dest)
    assert state["revision"] == "012345" and state["repository"] == "remote"
