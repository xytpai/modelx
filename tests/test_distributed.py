"""Real two-process DDP checks, launched by torchrun (never allocate a preset)."""

from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train as training
from model import ModelArgs, Transformer, token_loss


def launch(output, mode, backend="gloo", extra=()):
    env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
         str(Path(__file__).resolve()), mode, str(output), backend, *extra],
        capture_output=True, text=True, env=env, timeout=150,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


class RoutedModel(torch.nn.Module):
    """Different branches per rank plus an always-unused expert."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False) for _ in range(3)])

    def forward(self, tokens):
        expert = 0 if tokens[0, 0].item() == 2 else 1
        return self.experts[expert](self.embedding(tokens))


def batch_for(rank):
    tokens = torch.tensor([[2, 3, 4]] if rank == 0 else [[3, 4, 5, 6, 7]])
    labels = tokens.clone()
    labels[:, :2] = -100
    return {"input_ids": tokens, "labels": labels, "attention_mask": torch.ones_like(tokens)}


class UnevenLoader:
    def __init__(self, rank, empty):
        self.rank = rank
        self.num_batches = (0 if rank == 1 else 3) if empty else (3 if rank == 0 else 2)
        self.starts = []
        self.closed = 0

    def set_epoch(self, step):
        self.starts.append(step)

    def __iter__(self):
        try:
            for _ in range(self.num_batches):
                yield batch_for(self.rank)
        finally:
            self.closed += 1


def ddp_worker(output, backend, empty=False, dense=False, keep_checkpoints=3):
    rank = int(os.environ["RANK"])
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(5)
        if dense:
            args = ModelArgs(
                name="tiny-dense", dim=16, n_layers=2, n_heads=2,
                n_kv_heads=1, head_dim=8, vocab_size=16,
                ffn_type="dense", intermediate_size=48,
                num_experts=0, num_activated_experts=0, max_seq_len=16,
            )
            model = Transformer(args).to(device)
        else:
            model = RoutedModel().to(device)
        reference = deepcopy(model)
        unused_before = None if dense else model.experts[2].weight.detach().clone()
        wrapped = DDP(
            model, device_ids=[int(os.environ["LOCAL_RANK"])] if backend == "nccl" else None,
            find_unused_parameters=not dense, broadcast_buffers=False,
        )
        loader = UnevenLoader(rank, empty)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)

        original_save = training.save_checkpoint
        saved_steps = []

        def checked_save(path, state, progress, history, *, keep_checkpoints=3):
            assert rank == 0
            assert not any(key.startswith("module.") for key in state)
            saved_steps.append(progress["step"])
            return original_save(path, state, progress, history, keep_checkpoints=keep_checkpoints)

        training.save_checkpoint = checked_save
        if empty:
            try:
                training.train(wrapped, loader, optimizer, output=output, max_steps=5)
            except ValueError as exc:
                assert "at least one rank" in str(exc)
            else:
                raise AssertionError("empty rank must fail on both processes")
            assert loader.closed == 1
            return

        # Reference: one optimizer update over both ranks' valid target tokens.
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        losses = []
        for _ in range(5):
            ref_optimizer.zero_grad(set_to_none=True)
            combined = 0
            for source_rank in (0, 1):
                batch = batch_for(source_rank)
                tokens, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = token_loss(reference(tokens)[:, :-1], labels[:, 1:])
                combined = combined + loss * ((labels[:, 1:] != -100).sum().item() / 4)
            losses.append(combined.detach().item())
            combined.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
            ref_optimizer.step()

        steps, loss = training.train(
            wrapped, loader, optimizer, output=output, max_steps=5, log_interval=1, save_interval=1,
            keep_checkpoints=keep_checkpoints,
        )
        assert steps == 5 and loader.starts == [0, 2, 4] and loader.closed == 3
        assert abs(loss - losses[-1]) < 1e-5
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], atol=1e-6, rtol=1e-5)
            peers = [torch.empty_like(parameter) for _ in range(2)]
            dist.all_gather(peers, parameter)
            torch.testing.assert_close(peers[0], peers[1], atol=0, rtol=0)
        if not dense:
            assert torch.equal(model.experts[2].weight, unused_before)
        else:
            assert all(p.grad is not None for p in model.parameters())
        assert saved_steps == ([1, 2, 3, 4, 5] if rank == 0 else [])
        if rank == 0:
            _, weights, saved = training.load_checkpoint(output)
            assert saved["progress"]["input_tokens"] == 5 * 8
            assert saved["progress"]["loss_tokens"] == 5 * 4
            assert [step for step, _ in saved["loss_history"]] == [1, 2, 3, 4, 5]
            torch.testing.assert_close(torch.tensor([v for _, v in saved["loss_history"]]), torch.tensor(losses))
            assert len(training.checkpoint_paths(output)) == min(5, keep_checkpoints)
            assert all(value.dtype == torch.bfloat16 for value in weights.values())
    finally:
        dist.destroy_process_group()


def test_ddp_two_cpu_processes_token_weighting_unused_experts_and_queue(tmp_path):
    result = launch(tmp_path / "cpu", "core")
    logs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"step"')]
    assert [row["step"] for row in logs] == [1, 2, 3, 4, 5]


def test_dense_ddp_two_cpu_processes_without_unused_parameter_detection(tmp_path):
    result = launch(tmp_path / "dense", "dense")
    logs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"step"')]
    assert [row["step"] for row in logs] == [1, 2, 3, 4, 5]


def test_dense_ddp_keeps_one_readable_checkpoint_with_all_progress(tmp_path):
    output = tmp_path / "dense-keep-one"
    launch(output, "dense-keep-one")
    paths = training.checkpoint_paths(output)
    assert len(paths) == 1
    assert paths[0].name == "model-00000005.pt"
    _, _, saved = training.load_checkpoint(output)
    assert saved["progress"]["step"] == 5
    assert [step for step, _ in saved["loss_history"]] == [1, 2, 3, 4, 5]
    assert not list(output.glob("*.tmp"))


def test_ddp_empty_rank_fails_on_both_processes_without_hanging(tmp_path):
    launch(tmp_path / "empty", "empty")
    assert not training.checkpoint_paths(tmp_path / "empty")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_ddp_two_gpu_processes_token_weighting_and_queue(tmp_path):
    launch(tmp_path / "gpu", "core", "nccl")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_torchrun_cli_moe_resume_and_single_gpu_inference(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    import infer

    output = tmp_path / "model"
    output.mkdir()
    args = ModelArgs(
        name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8,
        vocab_size=16, moe_intermediate_size=24, num_experts=4,
        num_activated_experts=1, max_seq_len=32,
    )
    # Configuration is only a fixture for the tiny preset in the worker.
    # No weights exist: the first invocation must take main's fresh-training path.
    (output / "config.json").write_text(json.dumps(asdict(args)))
    raw = Tokenizer(models.WordLevel(
        {"<unk>": 0, "<eos>": 1, "User": 2, "Assistant": 3, ":": 4, "Q": 5, "A": 6},
        unk_token="<unk>",
    ))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(output / "tokenizer")
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({
        "id": str(i), "language": "en", "meta": {}, "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A " * (i + 1)},
        ],
    }) for i in range(5)) + "\n")
    first = launch(output, "cli", "nccl", [str(data), "3"])
    second = launch(output, "cli", "nccl", [str(data), "5"])
    for result, expected in ((first, [1, 2, 3]), (second, [4, 5])):
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"step"')]
        assert [row["step"] for row in rows] == expected
        assert result.stdout.count("Finished:") == 1
        assert "world_size=2" in result.stdout
    _, weights, saved = training.load_checkpoint(output)
    assert saved["progress"]["step"] == 5
    assert [step for step, _ in saved["loss_history"]] == [1, 2, 3, 4, 5]
    assert len(training.checkpoint_paths(output)) == 3
    assert not any(name.startswith("module.") for name in weights)
    assert not list(output.glob("*.jsonl"))
    # Same DDP checkpoint loads into the unwrapped single-device inference model.
    loaded, tok = infer.load_model(output, torch.device("cuda:0"))
    assert isinstance(infer.generate(loaded, tok, "Q", max_new_tokens=2), str)
    torch.cuda.synchronize()
    del loaded
    torch.cuda.empty_cache()



def test_book_and_qa_ddp_matches_global_token_weighted_reference(tmp_path):
    result = launch(tmp_path / "books-mixed", "books-mixed")
    logs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"step"')]
    assert len(logs) == 1 and logs[0]["loss_tokens"] == 5


def books_mixed_worker(output):
    """Same QA loader on both ranks: book continuation versus ordinary QA."""
    from dataloader import PretrainDataLoader

    class CharacterTokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 100 + 2 for c in text]

    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    try:
        output.mkdir(parents=True, exist_ok=True)
        data = output / "samples.jsonl"
        if rank == 0:
            data.write_text("\n".join(json.dumps(r) for r in [
                {"id": "book", "language": "en", "meta": {"title": "ignored"}, "text": "ABC"},
                {"id": "qa", "language": "en", "meta": {}, "messages": [
                    {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"},
                ]},
            ]) + "\n")
        dist.barrier()
        torch.manual_seed(5)
        args = ModelArgs(name="tiny-dense", dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
                         head_dim=8, vocab_size=128, ffn_type="dense", intermediate_size=24,
                         num_experts=0, num_activated_experts=0, max_seq_len=128)
        model = Transformer(args)
        reference = deepcopy(model)
        # Build the reference locally, without invoking distributed loader collectives.
        reference_loader = PretrainDataLoader(CharacterTokenizer(), data, max_seq_len=128)
        ref_batches = [
            reference_loader._collate([reference_loader._encode(
                reference_loader._to_sample(json.loads(line)),
            )])
            for line in data.read_text().splitlines()
        ]
        target_counts = [(b["labels"][:, 1:] != -100).sum().item() for b in ref_batches]
        assert target_counts == [3, 2]
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        ref_optimizer.zero_grad(set_to_none=True)
        expected = 0
        for batch, count in zip(ref_batches, target_counts):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = token_loss(reference(batch["input_ids"])[:, :-1], batch["labels"][:, 1:])
            expected = expected + loss * (count / sum(target_counts))
        expected.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
        ref_optimizer.step()
        wrapped = DDP(model, find_unused_parameters=False, broadcast_buffers=False)
        loader = PretrainDataLoader(
            CharacterTokenizer(), data, batch_size=1, max_seq_len=128, num_workers=2,
            rank=rank, world_size=2,
        )
        steps, value = training.train(
            wrapped, loader, torch.optim.SGD(wrapped.parameters(), lr=0.01),
            output=output, max_steps=1, save_interval=1, log_interval=1,
        )
        assert steps == 1 and abs(value - expected.item()) < 1e-5
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, dict(reference.named_parameters())[name], atol=1e-6, rtol=1e-5)
        if rank == 0:
            progress = training.load_checkpoint(output)[2]["progress"]
            assert progress["loss_tokens"] == 5
            assert progress["input_tokens"] == sum(int(b["attention_mask"].sum()) for b in ref_batches)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("count", [0, 1, 13])
def test_loader_summary_sums_all_ranks_including_empty_shards(tmp_path, count):
    result = launch(tmp_path / f"summary-{count}", "loader-summary", extra=[str(count)])
    summaries = [line for line in result.stdout.splitlines() if line.startswith("[Data] Total:")]
    header_tokens = len("User: Q\nAssistant: ")
    expected_tokens = sum(min(header_tokens + i + 1 + 1, 30) for i in range(count))
    assert summaries == [
        f"[Data] Total: samples={count}, tokens={expected_tokens} (after truncation, no padding)"
    ]
    assert "Encoding" not in result.stderr


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_loader_summary_nccl(tmp_path):
    result = launch(tmp_path / "summary-nccl", "loader-summary", "nccl", ["5"])
    assert result.stdout.count("[Data] Total: samples=5, tokens=115") == 1


def loader_summary_worker(output, backend, count):
    from dataloader import PretrainDataLoader

    class CharacterTokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    try:
        data = output / "samples.jsonl"
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            # Character tokens: prompt/headers, then answer and one EOS.
            records = [{"messages": [{"role": "user", "content": "Q"},
                                     {"role": "assistant", "content": "A" * (i + 1)}]}
                       for i in range(count)]
            data.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        dist.barrier()
        loader = PretrainDataLoader(CharacterTokenizer(), data, batch_size=2,
                                    max_seq_len=30, num_workers=1, rank=rank, world_size=2)
        batches = list(loader)
        indices = range(rank, count, 2)
        assert sum(b["input_ids"].shape[0] for b in batches) == len(indices)
        expected_tokens = sum(min(len("User: Q\nAssistant: ") + i + 2, 30) for i in indices)
        assert sum(int(b["attention_mask"].sum()) for b in batches) == expected_tokens
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mode, directory, backend = sys.argv[1:4]
    output = Path(directory)
    if mode == "loader-summary":
        loader_summary_worker(output, backend, int(sys.argv[4]))
    elif mode == "books-mixed":
        books_mixed_worker(output)
    elif mode == "cli":
        args = ModelArgs(**json.loads((output / "config.json").read_text()))
        training.get_config = lambda *a, **kw: args

        def tiny_model(args):
            model = Transformer(args)
            with torch.no_grad():
                model.layers[0].feed_forward.gate.weight.zero_()
            return model

        training.Transformer = tiny_model
        training.main([
            "--output-dir", str(output), "--data-dir", sys.argv[4],
            "--tokenizer", str(output / "tokenizer"),
            "--steps", sys.argv[5], "--batch-size", "1", "--num-workers", "1",
            "--log-interval", "1", "--save-interval", "1",
        ])
    else:
        ddp_worker(output, backend, empty=mode == "empty",
                   dense=mode in {"dense", "dense-keep-one"},
                   keep_checkpoints=1 if mode == "dense-keep-one" else 3)
