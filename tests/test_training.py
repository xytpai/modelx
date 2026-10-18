from dataclasses import asdict
import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from dataloader import PretrainDataLoader
from model import MODELS, ModelArgs, Transformer, get_config
from model.config import parameter_count
import train as training
from train import checkpoint_paths, load_checkpoint



@pytest.fixture
def cpu_cli(monkeypatch):
    # Exercise main on CPU without changing its production CUDA-only setting.
    from types import SimpleNamespace
    monkeypatch.setattr(training, "torch", SimpleNamespace(
        **(vars(torch) | {"device": lambda *a, **kw: torch.device("cpu")})
    ))


@pytest.fixture
def tiny_args():
    return ModelArgs(
        name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8,
        vocab_size=16, moe_intermediate_size=24, num_experts=2,
        num_activated_experts=2, max_seq_len=32,
    )


@pytest.fixture
def local_tokenizer(tmp_path):
    vocab = {word: index for index, word in enumerate(
        ["<unk>", "<eos>", "User", "Assistant", ":", "Q", "A", "next"]
    )}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>",
    )
    path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(path)
    return tokenizer, path


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "data" / "qa.jsonl"
    path.parent.mkdir()
    records = [
        {"id": str(i), "language": "en", "meta": {"topic": "not training text"},
         "messages": [
             {"role": "user", "content": "Q"},
             {"role": "assistant", "content": "A"},
         ] + ([
             {"role": "user", "content": "next Q"},
             {"role": "assistant", "content": "next A"},
         ] if i == 2 else [])}
        for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_default_2b_config_parameter_count_without_allocating_weights():
    args = get_config("2B")
    assert args == get_config()
    assert args.n_layers == 20 and args.dim == 2048
    assert args.ffn_type == "dense"
    assert args.num_experts == args.num_activated_experts == 0
    assert parameter_count(args) == 1_587_633_152
    with torch.device("meta"):
        model = Transformer(args)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(args)
    assert get_config("2B", max_seq_len=128).max_seq_len == 128
    assert MODELS["2B"].max_seq_len == 4096


@pytest.mark.parametrize("name", list(MODELS))
def test_production_config_dimensions_are_aligned(name):
    args = get_config(name)
    hidden = args.intermediate_size if args.ffn_type == "dense" else args.moe_intermediate_size
    assert args.dim > 0 and args.dim % 512 == 0
    assert hidden > 0 and hidden % 512 == 0
    assert args.head_dim == 128
    assert args.vocab_size % 512 == 0
    assert args.dim == args.n_heads * args.head_dim
    assert args.n_heads % args.n_kv_heads == 0
    assert (args.n_kv_heads * args.head_dim) % 512 == 0



def saved_history(output):
    return load_checkpoint(output)[2]["loss_history"]


@pytest.mark.parametrize("limit,log_interval,save_interval,expected", [
    (2, 1, 1, 2), (4, 2, 3, 4), (3, 3, 2, 3),
    (5, 1, 100, 5), (1, 100, 1, 1),
])
def test_independent_intervals_bf16_and_embedded_history(
    tmp_path, tiny_args, local_tokenizer, data_file, capsys,
    limit, log_interval, save_interval, expected,
):
    import threading
    model = Transformer(tiny_args)
    before = model.output.weight.detach().clone()
    loader = PretrainDataLoader(local_tokenizer[0], data_file, batch_size=2, max_seq_len=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    output = tmp_path / "run"
    seen = []

    def check(module, inputs, result):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        assert result.dtype == torch.bfloat16
        seen.append(True)

    hook = model.register_forward_hook(check)
    count, loss = training.train(
        model, loader, optimizer, output=output, max_steps=limit,
        log_interval=log_interval, save_interval=save_interval,
    )
    hook.remove()
    assert count == expected and len(seen) == expected
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.output.weight, before)
    assert model.output.weight.dtype == model.cos.dtype == torch.float32
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    expected_logs = list(range(log_interval, expected + 1, log_interval))
    assert [r["step"] for r in printed] == expected_logs
    assert all("epoch" not in r and "batch" not in r for r in printed)
    paths = checkpoint_paths(output)
    saved_steps = list(range(save_interval, expected + 1, save_interval))[-3:]
    assert len(paths) == len(saved_steps)
    for path, step in zip(paths, saved_steps):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert set(payload) == {"weights", "progress", "loss_history"}
        assert {v.dtype for v in payload["weights"].values()} == {torch.bfloat16}
        assert payload["progress"]["step"] == step
        assert set(payload["progress"]) == {"step", "loss", "lr", "input_tokens", "loss_tokens"}
        assert payload["loss_history"] == [
            (r["step"], r["loss"]) for r in printed if r["step"] <= step
        ]
    assert not list(output.glob("*.jsonl")) and not list(output.glob("*.png"))
    assert not list(output.glob("*.json"))
    assert not any(t.name.startswith(("datax-worker", "reader-worker")) for t in threading.enumerate())


def test_main_resumes_embedded_history_tokens_and_steps(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, capsys, cpu_cli,
):
    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "run"
    # Include an old log file to ensure continuation neither reads nor edits it.
    output.mkdir()
    old_log = output / "log.jsonl"
    old_log.write_bytes(b"ignore this obsolete log")
    cli = [
        "--output-dir", str(output), "--data-dir", str(data_file),
        "--tokenizer", str(local_tokenizer[1]), "--num-workers", "1",
        "--log-interval", "2", "--save-interval", "2",
    ]
    training.main([*cli, "--steps", "2"])
    _, weights, saved = load_checkpoint(output)
    assert saved["progress"]["step"] == 2
    assert "epoch" not in saved["progress"] and "batch" not in saved["progress"]
    assert [step for step, _ in saved["loss_history"]] == [2]
    assert json.loads((output / "config.json").read_text()) == asdict(tiny_args)
    assert PreTrainedTokenizerFast.from_pretrained(output / "tokenizer").get_vocab() == local_tokenizer[0].get_vocab()
    actual_train = training.train
    inspected = []

    def inspect(model, loader, optimizer, **kwargs):
        assert not optimizer.state
        assert kwargs["progress"] == saved["progress"]
        assert kwargs["loss_history"] == saved["loss_history"]
        for name, value in model.state_dict().items():
            assert value.dtype == torch.float32
            assert torch.equal(value, weights[name].float())
        inspected.append(True)
        return actual_train(model, loader, optimizer, **kwargs)

    def forbid(*a, **kw):
        pytest.fail("resume must use saved config")

    monkeypatch.setattr(training, "train", inspect)
    monkeypatch.setattr(training, "get_config", forbid)
    training.main([*cli, "--steps", "8", "--tokenizer", "must-not-load"])
    assert inspected == [True]
    _, _, continued = load_checkpoint(output)
    history = continued["loss_history"]
    assert history[:1] == saved["loss_history"]
    assert [step for step, _ in history] == [2, 4, 6, 8]
    assert continued["progress"]["step"] == 8
    assert len(list(output.glob("*.pt"))) == 3
    assert old_log.read_bytes() == b"ignore this obsolete log"
    assert sorted(p.name for p in output.glob("*.json")) == ["config.json"]

    loader = PretrainDataLoader(local_tokenizer[0], data_file, max_seq_len=32, num_workers=1)
    inputs = saved["progress"]["input_tokens"]
    targets = saved["progress"]["loss_tokens"]
    steps = 2
    while steps < 8:
        loader.set_epoch(steps)
        for batch in loader:
            steps += 1
            inputs += int(batch["attention_mask"].sum())
            targets += int((batch["labels"][:, 1:] != -100).sum())
            if steps == 8:
                break
    assert continued["progress"]["input_tokens"] == inputs
    assert continued["progress"]["loss_tokens"] == targets

    monkeypatch.setattr(training, "train", actual_train)
    before = snapshot(output)
    training.main([*cli, "--steps", "8"])
    assert snapshot(output) == before
    training.main([*cli, "--steps", "12"])
    assert [step for step, _ in saved_history(output)] == [2, 4, 6, 8, 10, 12]
    assert load_checkpoint(output)[2]["progress"]["step"] == 12
    assert "fresh AdamW" in capsys.readouterr().out


def test_non_aligned_save_point_has_step_without_extra_history_point(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, capsys, cpu_cli,
):
    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cli = [
        "--output-dir", str(tmp_path / "run"), "--data-dir", str(data_file),
        "--tokenizer", str(local_tokenizer[1]), "--num-workers", "1",
        "--log-interval", "2", "--save-interval", "3",
    ]
    output = tmp_path / "run"
    training.main([*cli, "--steps", "8"])
    _, _, saved = load_checkpoint(output)
    assert saved["progress"]["step"] == 6
    assert [step for step, _ in saved["loss_history"]] == [2, 4, 6]
    first_stdout = capsys.readouterr().out
    assert '"step": 8' in first_stdout  # Printed, but not yet saved.
    training.main([*cli, "--steps", "10"])
    _, _, continued = load_checkpoint(output)
    assert continued["progress"]["step"] == 9
    assert "epoch" not in continued["progress"] and "batch" not in continued["progress"]
    assert continued["loss_history"][:3] == saved["loss_history"]
    assert [step for step, _ in continued["loss_history"]] == [2, 4, 6, 8]
    assert not list(output.glob("*.jsonl"))


def test_interruption_keeps_previous_history_and_resumes(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file,
):
    model = Transformer(tiny_args)
    loader = PretrainDataLoader(local_tokenizer[0], data_file, max_seq_len=32)
    actual = model.forward
    calls = 0

    def interrupt(tokens):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt
        return actual(tokens)

    monkeypatch.setattr(model, "forward", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train(model, loader, torch.optim.AdamW(model.parameters()),
                       output=tmp_path, max_steps=6, log_interval=1, save_interval=2)
    _, weights, saved = load_checkpoint(tmp_path)
    assert saved["progress"]["step"] == 2
    assert [step for step, _ in saved["loss_history"]] == [1, 2]
    restored = Transformer(tiny_args)
    restored.load_state_dict(weights)
    training.train(restored, loader, torch.optim.AdamW(restored.parameters()),
                   output=tmp_path, max_steps=4, log_interval=1, save_interval=2,
                   progress=saved["progress"], loss_history=saved["loss_history"])
    assert [step for step, _ in saved_history(tmp_path)] == [1, 2, 3, 4]
    assert [step for step, _ in saved["loss_history"]] == [1, 2]  # Caller-owned history isn't mutated.


def test_corrupt_checkpoint_fallback_restores_its_own_loss_history(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, cpu_cli,
):
    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cli = ["--output-dir", str(tmp_path / "run"), "--data-dir", str(data_file),
           "--tokenizer", str(local_tokenizer[1]), "--num-workers", "1",
           "--log-interval", "1", "--save-interval", "1"]
    output = tmp_path / "run"
    training.main([*cli, "--steps", "4"])
    history = saved_history(output)[:3]
    checkpoint_paths(output)[-1].write_bytes(b"broken")
    with pytest.warns(RuntimeWarning, match="Skipping unusable"):
        training.main([*cli, "--steps", "5"])
    assert saved_history(output)[:3] == history
    assert [step for step, _ in saved_history(output)] == [1, 2, 3, 4, 5]
    assert len(list(output.glob("*.pt"))) == 3
    assert not list(output.glob("*.jsonl"))


def test_nonfinite_gradient_is_not_saved(tmp_path, tiny_args, local_tokenizer, data_file, monkeypatch):
    model = Transformer(tiny_args)
    loader = PretrainDataLoader(local_tokenizer[0], data_file, max_seq_len=32)
    monkeypatch.setattr(training, "token_loss", lambda logits, labels: logits.sum() * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        training.train(model, loader, torch.optim.AdamW(model.parameters()),
                       output=tmp_path, max_steps=1, log_interval=1, save_interval=1)
    assert not checkpoint_paths(tmp_path) and not list(tmp_path.glob("*.jsonl"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_gpu_forced_bf16_autocast(tmp_path, tiny_args, local_tokenizer, data_file):
    model = Transformer(tiny_args).cuda()
    loader = PretrainDataLoader(local_tokenizer[0], data_file, max_seq_len=32)
    dtypes = []
    hook = model.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    count, loss = training.train(model, loader, torch.optim.AdamW(model.parameters()),
                                 output=tmp_path, max_steps=1, log_interval=1, save_interval=1)
    hook.remove()
    assert count == 1 and dtypes == [torch.bfloat16]
    assert model.cos.dtype == model.sin.dtype == torch.float32
    assert all(v.dtype == torch.bfloat16 for v in load_checkpoint(tmp_path)[1].values())


def weights(value):
    return {"weight": torch.full((2, 3), float(value))}


def save(output, value):
    return training.save_checkpoint(output, weights(value), {"step": value, "loss": 4.0},
                                    [(step, 4.0) for step in range(1, value + 1)])


def snapshot(output):
    return {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_queue_new_weights_and_history_committed_before_oldest_deleted(tmp_path, monkeypatch):
    first = [save(tmp_path, step) for step in range(1, 4)]
    (tmp_path / "optimizer.pt").write_bytes(b"unrelated")
    original = training.checkpoint_paths
    seen = []

    def inspect(output):
        paths = original(output)
        if len(paths) == 4:
            assert all(path.exists() for path in first)
            payload = torch.load(paths[-1], weights_only=True)
            assert set(payload) == {"weights", "progress", "loss_history"}
            assert payload["progress"]["step"] == 4
            assert payload["loss_history"] == [(1, 4.0), (2, 4.0), (3, 4.0), (4, 4.0)]
            assert payload["weights"]["weight"].dtype == torch.bfloat16
            seen.append(True)
        return paths

    monkeypatch.setattr(training, "checkpoint_paths", inspect)
    newest = save(tmp_path, 4)
    assert seen == [True]
    assert checkpoint_paths(tmp_path) == [*first[1:], newest]
    assert (tmp_path / "optimizer.pt").read_bytes() == b"unrelated"
    assert not list(tmp_path.glob("*.jsonl")) and not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failure", ["write", "weights", "history"])
def test_failed_save_preserves_previous_queue(tmp_path, monkeypatch, failure):
    for step in range(1, 4):
        save(tmp_path, step)
    before = snapshot(tmp_path)
    if failure == "write":
        def fail(payload, handle):
            handle.write(b"incomplete")
            raise OSError("disk full")
        monkeypatch.setattr(training.torch, "save", fail)
    else:
        original = training.torch.load

        def wrong(*a, **kw):
            payload = original(*a, **kw)
            if failure == "weights":
                payload["weights"]["weight"] = weights(-1)["weight"]
            else:
                payload["loss_history"] = []
            return payload
        monkeypatch.setattr(training.torch, "load", wrong)
    with pytest.raises((OSError, ValueError)):
        save(tmp_path, 4)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("marker", [None, "anything", "datax-log-v1"])
def test_load_does_not_check_format_marker(tmp_path, marker):
    payload = {"weights": weights(1), "progress": {"step": 1}, "loss_history": [(1, 3.0)]}
    if marker is not None:
        payload["format"] = marker
    path = tmp_path / "model-00000001.pt"
    torch.save(payload, path)
    chosen, state, saved = load_checkpoint(tmp_path)
    assert chosen == path and saved == {"progress": {"step": 1}, "loss_history": [(1, 3.0)]}


def test_old_files_and_stale_temp_ignored_all_bad_raise(tmp_path):
    old_files = {
        "model.pt": b"old weights", "training_state.json": b"old progress",
        "log.jsonl": b"old log", "model-00000001.json": b"old sidecar",
        "model-00000001.tmp": b"partial",
    }
    for name, content in old_files.items():
        (tmp_path / name).write_bytes(content)
    assert checkpoint_paths(tmp_path) == []
    new = save(tmp_path, 1)
    assert (tmp_path / "log.jsonl").read_bytes() == b"old log"
    new.write_bytes(b"broken")
    before = snapshot(tmp_path)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="no usable checkpoint"):
        load_checkpoint(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("new_count,new_batch_size", [(1, 1), (5, 2)])
def test_resume_with_changed_data_size_and_batch_size(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, cpu_cli,
    new_count, new_batch_size,
):
    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    output = tmp_path / "variable"
    cli = [
        "--output-dir", str(output), "--data-dir", str(data_file),
        "--tokenizer", str(local_tokenizer[1]), "--num-workers", "1",
        "--log-interval", "1", "--save-interval", "1",
    ]
    training.main([*cli, "--steps", "2"])
    _, weights_before, saved = load_checkpoint(output)
    before = saved["progress"]
    rows = [json.loads(line) for line in data_file.read_text().splitlines()]
    rows = [dict(rows[0], id=str(i)) for i in range(new_count)]
    data_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    training.main([*cli, "--steps", "7", "--batch-size", str(new_batch_size)])
    _, _, resumed = load_checkpoint(output)
    assert resumed["progress"]["step"] == 7
    assert resumed["loss_history"][:2] == saved["loss_history"]
    assert [step for step, _ in resumed["loss_history"]] == list(range(1, 8))
    assert set(resumed["progress"]) == {"step", "loss", "lr", "input_tokens", "loss_tokens"}

    # Exactly five updates on the new data, not old-batch skipping.
    loader = PretrainDataLoader(local_tokenizer[0], data_file, batch_size=new_batch_size, max_seq_len=32)
    expected_inputs = before["input_tokens"]
    expected_targets = before["loss_tokens"]
    steps = 2
    while steps < 7:
        loader.set_epoch(steps)
        for batch in loader:
            steps += 1
            expected_inputs += int(batch["attention_mask"].sum())
            expected_targets += int((batch["labels"][:, 1:] != -100).sum())
            if steps == 7:
                break
    assert resumed["progress"]["input_tokens"] == expected_inputs
    assert resumed["progress"]["loss_tokens"] == expected_targets


def test_old_epoch_batch_info_is_ignored_when_resuming_new_data(
    tmp_path, tiny_args, local_tokenizer, data_file,
):
    model = Transformer(tiny_args)
    loader = PretrainDataLoader(local_tokenizer[0], data_file, max_seq_len=32)
    progress = {
        "step": 10, "loss": 4.0, "epoch": 999, "batch": 999999,
        "input_tokens": 100, "loss_tokens": 50,
    }
    count, _ = training.train(
        model, loader, torch.optim.AdamW(model.parameters()), output=tmp_path,
        max_steps=11, log_interval=1, save_interval=1,
        progress=progress, loss_history=[(10, 4.0)],
    )
    assert count == 11
    _, _, saved = load_checkpoint(tmp_path)
    assert "epoch" not in saved["progress"] and "batch" not in saved["progress"]
    assert [step for step, _ in saved["loss_history"]] == [10, 11]


def test_empty_dataset_fails_instead_of_looping_forever(
    tmp_path, tiny_args, local_tokenizer,
):
    data = tmp_path / "empty.jsonl"
    data.write_text("")
    model = Transformer(tiny_args)
    loader = PretrainDataLoader(local_tokenizer[0], data, max_seq_len=32)
    with pytest.raises(ValueError, match="no training samples"):
        training.train(model, loader, torch.optim.AdamW(model.parameters()),
                       output=tmp_path / "run", max_steps=5)
    assert not checkpoint_paths(tmp_path / "run")


def test_step_target_already_met_does_not_read_or_rewrite(
    tmp_path, tiny_args,
):
    class NoRead:
        def __iter__(self):
            pytest.fail("completed step target must not read data")
    model = Transformer(tiny_args)
    assert training.train(
        model, NoRead(), torch.optim.AdamW(model.parameters()), output=tmp_path,
        max_steps=5, progress={"step": 7, "loss": 3.0},
    ) == (7, 3.0)
    assert list(tmp_path.iterdir()) == []


def test_cli_uses_steps_not_epochs(capsys):
    with pytest.raises(SystemExit) as exc:
        training.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--steps" in help_text and "--epochs" not in help_text
    with pytest.raises(SystemExit) as exc:
        training.main([])
    assert exc.value.code == 2
    assert "--steps" in capsys.readouterr().err


@pytest.mark.parametrize("preset", ["2B", "7B"])
def test_dense_cli_train_resume_and_inference(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, cpu_cli, preset,
):
    """Dense config persists and resumes via existing training/inference entrypoints."""
    from dataclasses import replace
    import infer

    args = replace(tiny_args, name="tiny-dense", ffn_type="dense",
                   intermediate_size=48, num_experts=0, num_activated_experts=0)
    calls = []

    def choose(name, max_seq_len=None):
        calls.append(name)
        assert name == preset
        return args

    monkeypatch.setattr(training, "get_config", choose)
    output = tmp_path / "dense-run"
    cli = [
        "--model", preset, "--output-dir", str(output),
        "--data-dir", str(data_file), "--tokenizer", str(local_tokenizer[1]),
        "--num-workers", "1", "--log-interval", "1", "--save-interval", "1",
    ]
    training.main([*cli, "--steps", "2"])
    assert calls == [preset]
    saved_args = ModelArgs(**json.loads((output / "config.json").read_text()))
    assert saved_args == args
    _, state, _ = load_checkpoint(output)
    assert any("feed_forward.w1.weight" in name for name in state)
    assert not any(".experts." in name or ".gate." in name for name in state)
    training.main([*cli, "--steps", "3"])
    assert calls == [preset]  # Resume uses the saved dense config.
    assert load_checkpoint(output)[2]["progress"]["step"] == 3
    restored, tokenizer = infer.load_model(output, torch.device("cpu"))
    assert type(restored.layers[0].feed_forward).__name__ == "FeedForward"
    assert isinstance(infer.generate(restored, tokenizer, "Q", max_new_tokens=2), str)


@pytest.mark.parametrize("initial_keep,resumed_keep", [(1, 1), (3, 1), (1, 4)])
def test_cli_checkpoint_retention_survives_resume_and_can_change(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, cpu_cli,
    initial_keep, resumed_keep,
):
    import infer

    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    output = tmp_path / "retention-run"
    cli = [
        "--output-dir", str(output), "--data-dir", str(data_file),
        "--tokenizer", str(local_tokenizer[1]), "--num-workers", "1",
        "--log-interval", "1", "--save-interval", "1",
    ]
    training.main([*cli, "--steps", "4", "--keep-checkpoints", str(initial_keep)])
    assert len(checkpoint_paths(output)) == initial_keep
    assert load_checkpoint(output)[2]["progress"]["step"] == 4
    config_before = (output / "config.json").read_bytes()
    files_before = snapshot(output)

    # Changing the limit does not delete files merely upon resume with no new step.
    training.main([*cli, "--steps", "4", "--keep-checkpoints", str(resumed_keep)])
    assert snapshot(output) == files_before

    def forbid(*a, **kw):
        pytest.fail("resume must use saved config")
    monkeypatch.setattr(training, "get_config", forbid)
    training.main([*cli, "--steps", "8", "--keep-checkpoints", str(resumed_keep)])
    paths = checkpoint_paths(output)
    assert len(paths) == resumed_keep
    assert paths[-1].name == "model-00000008.pt"
    _, weights, saved = load_checkpoint(output)
    assert saved["progress"]["step"] == 8
    assert [step for step, _ in saved["loss_history"]] == list(range(1, 9))
    assert (output / "config.json").read_bytes() == config_before
    assert not list(output.glob("*.tmp"))
    restored, tok = infer.load_model(output, torch.device("cpu"))
    assert set(restored.state_dict()) == set(weights)
    assert isinstance(infer.generate(restored, tok, "Q", max_new_tokens=2), str)


def test_cli_default_data_directory_uses_new_loader_from_any_cwd(
    tmp_path, monkeypatch, tiny_args, local_tokenizer, data_file, cpu_cli,
):
    from dataloader.loader import PretrainDataLoader as Loader

    monkeypatch.setattr(training, "get_config", lambda *a, **kw: tiny_args)
    monkeypatch.setattr(training, "DATA_DIR", data_file.parent)
    monkeypatch.chdir(tmp_path)
    original_train = training.train
    seen = []

    def check(model, loader, optimizer, **kwargs):
        assert type(loader) is Loader
        assert loader.root == data_file.parent
        assert loader.batch_size == 2 and loader.num_workers == 1
        assert loader.max_seq_len == tiny_args.max_seq_len
        assert loader.rank == 0 and loader.world_size == 1
        seen.append(True)
        return original_train(model, loader, optimizer, **kwargs)

    monkeypatch.setattr(training, "train", check)
    output = tmp_path / "default-data-run"
    training.main([
        "--tokenizer", str(local_tokenizer[1]), "--output-dir", str(output),
        "--steps", "1", "--batch-size", "2", "--num-workers", "1",
        "--save-interval", "1", "--log-interval", "1",
    ])
    assert seen == [True]
    assert load_checkpoint(output)[2]["progress"]["step"] == 1
