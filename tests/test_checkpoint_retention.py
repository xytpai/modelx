"""Configurable checkpoint retention, including transactional keep-one behavior."""
from pathlib import Path

import pytest
import torch

import train as training


def state(step):
    return {"weight": torch.full((2, 3), float(step))}


def history(step):
    return [(i, 1.0 / i) for i in range(1, step + 1)]


def save(directory, step, keep=3):
    return training.save_checkpoint(
        directory, state(step), {"step": step}, history(step),
        keep_checkpoints=keep,
    )


def snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("keep", [1, 2, 3, 5])
def test_retention_keeps_latest_files_and_preserves_full_progress(tmp_path, keep):
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    for step in range(1, 8):
        path = save(tmp_path, step, keep)
        assert path.name == f"model-{step:08d}.pt"
        paths = training.checkpoint_paths(tmp_path)
        assert len(paths) == min(step, keep)
        assert [int(p.stem[6:]) for p in paths] == list(range(max(1, step - keep + 1), step + 1))
        loaded_path, weights, metadata = training.load_checkpoint(tmp_path)
        assert loaded_path == path
        assert weights["weight"].dtype == torch.bfloat16
        assert torch.equal(weights["weight"], state(step)["weight"].bfloat16())
        assert metadata == {"progress": {"step": step}, "loss_history": history(step)}
        assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "notes.txt").read_text() == "not a checkpoint"


@pytest.mark.parametrize("keep", [1, 2, 5])
def test_publication_precedes_any_retention_deletion(tmp_path, monkeypatch, keep):
    previous = [save(tmp_path, step, keep) for step in range(1, keep + 1)]
    original = training.checkpoint_paths
    observed = []

    def inspect(directory):
        paths = original(directory)
        if len(paths) == keep + 1:
            assert all(p.exists() for p in previous)
            payload = torch.load(paths[-1], weights_only=True)
            assert payload["progress"]["step"] == keep + 1
            assert payload["loss_history"] == history(keep + 1)
            assert torch.equal(payload["weights"]["weight"], state(keep + 1)["weight"].bfloat16())
            observed.append(True)
        return paths

    monkeypatch.setattr(training, "checkpoint_paths", inspect)
    latest = save(tmp_path, keep + 1, keep)
    assert observed == [True]
    assert len(original(tmp_path)) == keep
    assert latest.exists() and not previous[0].exists()


@pytest.mark.parametrize("failure", ["write", "verification", "rename"])
def test_keep_one_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    previous = save(tmp_path, 1, 1)
    before = snapshot(tmp_path)
    if failure == "write":
        def fail(payload, file):
            file.write(b"partial checkpoint")
            raise OSError("simulated disk full")
        monkeypatch.setattr(training.torch, "save", fail)
    elif failure == "verification":
        real_load = training.torch.load

        def bad_load(path, **kwargs):
            payload = real_load(path, **kwargs)
            if Path(path).suffix == ".tmp":
                payload["progress"] = {"step": -1}
            return payload
        monkeypatch.setattr(training.torch, "load", bad_load)
    else:
        real_replace = Path.replace

        def fail_replace(path, target):
            if path.suffix == ".tmp":
                raise OSError("simulated publish failure")
            return real_replace(path, target)
        monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises((OSError, ValueError)):
        save(tmp_path, 2, 1)
    assert snapshot(tmp_path) == before
    assert training.load_checkpoint(tmp_path)[0] == previous
    assert training.load_checkpoint(tmp_path)[2]["progress"]["step"] == 1


def test_changing_retention_at_next_save_keeps_serials_monotonic(tmp_path):
    for step in range(1, 5):
        save(tmp_path, step, 3)
    assert [p.name for p in training.checkpoint_paths(tmp_path)] == [
        "model-00000002.pt", "model-00000003.pt", "model-00000004.pt",
    ]
    latest = save(tmp_path, 5, 1)
    assert training.checkpoint_paths(tmp_path) == [latest]
    assert latest.name == "model-00000005.pt"
    save(tmp_path, 6, 4)
    assert [p.name for p in training.checkpoint_paths(tmp_path)] == [
        "model-00000005.pt", "model-00000006.pt",
    ]
    assert training.load_checkpoint(tmp_path)[2]["loss_history"] == history(6)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "1", None])
def test_invalid_api_retention_fails_before_creating_or_deleting_files(tmp_path, bad):
    directory = tmp_path / "new"
    with pytest.raises(ValueError, match="integer >= 1"):
        save(directory, 1, bad)
    assert not directory.exists()
    with pytest.raises(ValueError, match="integer >= 1"):
        training.train(None, None, None, output=directory, max_steps=1, keep_checkpoints=bad)
    assert not directory.exists()


@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "invalid"])
def test_invalid_cli_retention_fails_before_gpu_or_model_setup(monkeypatch, bad):
    monkeypatch.setenv("WORLD_SIZE", "8")

    def forbid(*a, **kw):
        pytest.fail("validation must precede GPU/model setup")
    monkeypatch.setattr(training, "run", forbid)
    monkeypatch.setattr(training.torch.cuda, "set_device", forbid)
    with pytest.raises(SystemExit) as exc:
        training.main(["--steps", "1", "--keep-checkpoints", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize("options,expected", [([], 3), (["--keep-checkpoints", "1"], 1),
                                               (["--keep-checkpoints", "5"], 5)])
def test_cli_retention_default_and_override(monkeypatch, options, expected):
    received = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(training, "run", lambda cli: received.append(cli.keep_checkpoints))
    training.main(["--steps", "1", *options])
    assert received == [expected]
