"""Task weights are simple integer record repetition, not sampling probabilities."""
from collections import Counter
import json
import math
import threading

import pytest
import torch

from dataloader import PretrainDataLoader
from dataloader.loader import parse_task_weights
import train


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def write_task(root, task, count):
    path = root/task/"one.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": f"{task}:{i:05d}"},
    ]}) + "\n" for i in range(count)))
    return path


def answers(loader):
    result = []
    for batch in loader:
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            text = "".join(chr(t) for t in ids[mask.bool()].tolist() if t)
            result.append(text.split("Assistant: ", 1)[1])
    return result


def test_parse_integer_repeat_counts():
    assert parse_task_weights(None) is None
    assert parse_task_weights(" cot=5, qa=2,code=0 ") == {"code": 0, "cot": 5, "qa": 2}
    assert parse_task_weights({"qa": 2, "cot": 5}) == {"cot": 5, "qa": 2}


@pytest.mark.parametrize("value", [
    "", "qa", "qa=", "=1", "qa=-1", "qa=1.5", "qa=2.0", "qa=nan", "qa=inf",
    "qa=0,cot=0", "qa=1,qa=2", "qa=1,", "qa=1,,cot=2",
    "../qa=1", "/qa=1", ".=1", "..=1", "qa/cot=1", r"qa\cot=1",
    {}, {"qa": True}, {"qa": None}, {"qa": 1.5}, {"qa": -1}, [], {1: 1},
])
def test_invalid_repeat_counts(value):
    with pytest.raises(ValueError):
        parse_task_weights(value)


def test_each_record_repeats_exactly_weight_times(tmp_path):
    write_task(tmp_path, "qa", 3)
    write_task(tmp_path, "code", 7)
    loader = PretrainDataLoader(Tokenizer(), tmp_path, task_weights="qa=2,code=3",
                               batch_size=2, num_workers=2)
    result = Counter(answers(loader))
    assert result == Counter({
        **{f"qa:{i:05d}": 2 for i in range(3)},
        **{f"code:{i:05d}": 3 for i in range(7)},
    })
    assert loader.num_batches == math.ceil(27/2)
    assert sum(v for k, v in result.items() if k.startswith("qa:")) == 6


def test_zero_and_unlisted_tasks_are_not_read(tmp_path):
    write_task(tmp_path, "qa", 3)
    for task in ["code", "cot"]:
        p = tmp_path/task/"bad.jsonl"
        p.parent.mkdir()
        p.write_text("{invalid\n")
    loader = PretrainDataLoader(Tokenizer(), tmp_path, task_weights="qa=2,code=0")
    assert Counter(answers(loader)) == Counter({f"qa:{i:05d}": 2 for i in range(3)})


def test_repeated_data_is_encoded_only_once_per_pass(tmp_path):
    write_task(tmp_path, "qa", 4)
    loader = PretrainDataLoader(Tokenizer(), tmp_path, task_weights="qa=5", num_workers=2)
    actual = loader._encode
    lock = threading.Lock()
    calls = []

    def encode(sample):
        with lock:
            calls.append(sample)
        return actual(sample)

    loader._encode = encode
    assert len(answers(loader)) == 20
    assert len(calls) == 4


def test_rank_shards_are_disjoint_and_cover_repeated_data(tmp_path):
    write_task(tmp_path, "qa", 7)
    write_task(tmp_path, "cot", 13)
    opts = dict(task_weights="qa=2,cot=3", batch_size=2, num_workers=1)
    full = Counter(answers(PretrainDataLoader(Tokenizer(), tmp_path, **opts)))
    shards = [Counter(answers(PretrainDataLoader(
        Tokenizer(), tmp_path, rank=r, world_size=3, **opts
    ))) for r in range(3)]
    assert sum(shards, Counter()) == full
    assert not set(shards[0]) & set(shards[1])


def test_workers_and_epoch_keep_original_deterministic_shuffle(tmp_path):
    write_task(tmp_path, "qa", 20)
    write_task(tmp_path, "cot", 11)
    opts = dict(task_weights="qa=2,cot=3", batch_size=3, seed=42)
    loader = PretrainDataLoader(Tokenizer(), tmp_path, num_workers=1, **opts)
    first = answers(loader)
    assert first == answers(PretrainDataLoader(Tokenizer(), tmp_path, num_workers=4, **opts))
    assert first == answers(loader)
    loader.set_epoch(10)
    other = answers(loader)
    assert first != other and Counter(first) == Counter(other)


def test_no_weights_preserves_original_counts(tmp_path):
    write_task(tmp_path, "qa", 3)
    write_task(tmp_path, "cot", 5)
    a = answers(PretrainDataLoader(Tokenizer(), tmp_path, batch_size=2))
    b = answers(PretrainDataLoader(Tokenizer(), tmp_path, batch_size=2, task_weights=None))
    assert a == b and len(a) == 8
    assert all(v == 1 for v in Counter(a).values())


def test_file_paths_and_missing_tasks_fail_clearly(tmp_path):
    path = write_task(tmp_path, "qa", 1)
    with pytest.raises(ValueError, match="directory containing task"):
        answers(PretrainDataLoader(Tokenizer(), path, task_weights="qa=1"))
    with pytest.raises(ValueError, match="does not exist"):
        answers(PretrainDataLoader(Tokenizer(), tmp_path, task_weights="missing=1"))
    (tmp_path/"empty").mkdir()
    with pytest.raises(ValueError, match="no enabled JSONL"):
        answers(PretrainDataLoader(Tokenizer(), tmp_path, task_weights="empty=1"))
    write_task(tmp_path, "agent", 2)
    with pytest.raises(ValueError, match="no enabled JSONL"):
        answers(PretrainDataLoader(Tokenizer(), tmp_path, task_weights="agent=1"))


def test_cli_passes_repeat_counts_to_run(monkeypatch):
    received = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(train, "run", lambda cli: received.append(cli.task_weights))
    train.main(["--steps", "1", "--task-weights", "qa=2,cot=5"])
    train.main(["--steps", "1"])
    assert received == [{"cot": 5, "qa": 2}, None]
    with pytest.raises(SystemExit):
        train.main(["--steps", "1", "--task-weights", "qa=0.5"])


def test_repetition_works_with_original_training_loop(tmp_path):
    from model import ModelArgs, Transformer

    class TinyTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 120 + 2 for c in text]

    write_task(tmp_path/"data", "qa", 2)
    write_task(tmp_path/"data", "cot", 3)
    loader = PretrainDataLoader(TinyTokenizer(), tmp_path/"data", task_weights="qa=2,cot=3",
                               batch_size=2, max_seq_len=128, num_workers=1)
    cfg = ModelArgs(name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
                    head_dim=8, vocab_size=128, ffn_type="dense", intermediate_size=24,
                    num_experts=0, num_activated_experts=0, max_seq_len=128)
    model = Transformer(cfg)
    step, loss = train.train(model, loader, torch.optim.SGD(model.parameters(), lr=.01),
                             output=tmp_path/"run", max_steps=9, log_interval=3, save_interval=9)
    assert step == 9 and math.isfinite(loss)
    assert train.load_checkpoint(tmp_path/"run")[2]["progress"]["step"] == 9
