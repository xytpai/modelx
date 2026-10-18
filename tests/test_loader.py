"""Tests for the single loader.py implementation and its training integration."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from dataloader import IGNORE_INDEX, PretrainDataLoader
from dataloader.loader import PretrainDataLoader as Loader
from dataloader.utils import find_jsonl_files, load_jsonl_files
from model import ModelArgs, Transformer
import train as training


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def qa(index=0):
    return {"id": str(index), "language": "zh", "meta": {"source": "DO_NOT_ENCODE"},
            "messages": [{"role": "user", "content": f"问题{index}"},
                         {"role": "assistant", "content": f"答案{index}"}]}


def text_record(text="ABC", **meta):
    return {"id": "DO_NOT_ENCODE", "language": "zh", "meta": meta, "text": text}


def write(path, *records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")
    return path


def rows(loader):
    return [tuple(ids[mask.bool()].tolist()) for batch in loader
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]


def test_public_api_and_training_use_new_implementation():
    assert PretrainDataLoader is Loader
    assert training.PretrainDataLoader is Loader
    assert training.IGNORE_INDEX == IGNORE_INDEX == -100


def test_find_files_recursive_line_count_and_path_order(tmp_path):
    b = write(tmp_path / "wiki" / "b.jsonl", text_record())
    a = write(tmp_path / "books" / "a.jsonl", text_record())
    c = write(tmp_path / "qa" / "nested" / "c.jsonl", qa(0), qa(1))
    (tmp_path / "ignore.txt").write_text("not JSONL")
    (tmp_path / "directory.jsonl").mkdir()
    assert find_jsonl_files(tmp_path) == [a, b, c]
    assert find_jsonl_files(a) == [a]
    assert find_jsonl_files(tmp_path / "directory.jsonl") == []
    with pytest.raises(FileNotFoundError):
        find_jsonl_files(tmp_path / "missing")
    with pytest.raises(ValueError, match="JSONL"):
        find_jsonl_files(tmp_path / "ignore.txt")


def test_paths_relative_to_cwd_and_home_expansion(tmp_path, monkeypatch):
    path = write(tmp_path / "data" / "one.jsonl", qa())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_jsonl_files("data/one.jsonl") == [Path("data/one.jsonl")]
    assert find_jsonl_files("~/data") == [path]


def test_parallel_reading_is_ordered_and_shards_records_across_files(tmp_path):
    a = write(tmp_path / "a.jsonl", *(qa(i) for i in range(3)))
    b = write(tmp_path / "b.jsonl", *(qa(i) for i in range(3, 9)))
    a.write_text("\n" + a.read_text() + "\n")
    serial = load_jsonl_files([a, b], num_workers=1)
    assert serial == [qa(i) for i in range(9)]
    assert load_jsonl_files([a, b], num_workers=4) == serial
    for rank in range(3):
        assert load_jsonl_files([a, b], 2, rank, 3) == serial[rank::3]


def test_conversation_order_and_only_assistant_content_is_supervised(tmp_path):
    loader = Loader(Tokenizer(), tmp_path)
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "developer", "content": "约束"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "调用"},
        {"role": "tool", "content": "结果"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    sample = loader._to_sample(dict(qa(), messages=messages))
    for i, message in enumerate(messages):
        assert sample[2 * i] == (("\n" if i else "") + message["role"].capitalize() + ": ", False)
        assert sample[2 * i + 1] == (message["content"], message["role"] == "assistant")
    ids, labels = loader._encode(sample)
    assert ids == Tokenizer().encode("".join(part for part, _ in sample)) + [0]
    assert [label for label in labels if label != IGNORE_INDEX] == Tokenizer().encode("调用A1A2") + [0]
    assert "DO_NOT_ENCODE" not in "".join(part for part, _ in sample)


@pytest.mark.parametrize("source", ["books", "wiki", "other"])
@pytest.mark.parametrize("text", ["AB", "这是原文开头。后面是真实原文。", "😊你好🌍世界", "文" * 1500])
def test_text_sources_share_neutral_prompt_and_exact_suffix(tmp_path, source, text):
    r = text_record(text, title=" 测试标题 ", chapters=[" 概述 ", "", "正文"],
                    source_url="DO_NOT_ENCODE")
    original = deepcopy(r)
    loader = Loader(Tokenizer(), tmp_path / source)
    sample = loader._to_sample(r)
    user, answer = sample[1][0], sample[3][0]
    assert user.startswith("根据以下原文开头续写后文，保持原文风格，不要重复开头。")
    assert "\ntitle: 测试标题\nchapters: 概述 / 正文\n" in user
    assert "DO_NOT_ENCODE" not in user and "书名：" not in user
    opening = user.split("\n原文开头：\n", 1)[1]
    assert len(opening) == min(128, max(1, len(text) // 10))
    assert opening + answer == text
    ids, labels = loader._encode(sample)
    assert labels[-len(answer)-1:] == Tokenizer().encode(answer) + [0]
    assert labels[:-len(answer)-1] == [IGNORE_INDEX] * (len(ids) - len(answer) - 1)
    assert r == original


def test_mixed_recursive_loading_padding_and_real_eos(tmp_path):
    write(tmp_path / "books" / "one.jsonl", text_record("ABC"))
    write(tmp_path / "wiki" / "deep" / "two.jsonl", text_record("ABCDE", title="词条"))
    write(tmp_path / "qa" / "three.jsonl", qa())
    loader = Loader(Tokenizer(), tmp_path, batch_size=3, max_seq_len=256, num_workers=2)
    batches = list(loader)
    assert len(batches) == loader.num_batches == 1
    batch = batches[0]
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape == (3, 128)
    for ids, labels, mask in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        length = int(mask.sum())
        assert ids[length - 1] == labels[length - 1] == 0
        assert (ids[length:] == 0).all()
        assert (labels[length:] == IGNORE_INDEX).all()
        assert (mask[:length] == 1).all() and (mask[length:] == 0).all()
    assert int((batch["labels"] != IGNORE_INDEX).sum()) == 3 + 5 + 4


@pytest.mark.parametrize("record", [qa(), text_record("书籍和百科原文" * 100)])
def test_qa_and_text_tail_truncation_preserves_ids_and_labels(tmp_path, record):
    loader = Loader(Tokenizer(), tmp_path)
    sample = loader._to_sample(record)
    ids, labels = loader._encode(sample)
    path = write(tmp_path / "one.jsonl", record)
    first_answer = next(i for i, label in enumerate(labels) if label != IGNORE_INDEX)
    for limit in (None, len(ids) + 1, len(ids), len(ids) - 1, first_answer + 1, first_answer):
        limited = Loader(Tokenizer(), path, max_seq_len=limit, num_workers=2)
        assert limited._encode(sample) == (ids[:limit], labels[:limit])
        batch = next(iter(limited))
        length = len(ids[:limit])
        assert int(batch["attention_mask"].sum()) == length
        assert batch["input_ids"][0, :length].tolist() == ids[:limit]
        assert batch["labels"][0, :length].tolist() == labels[:limit]
        assert (batch["labels"][0, length:] == IGNORE_INDEX).all()
        if limit is not None:
            assert batch["input_ids"].shape[1] <= limit
        if limit is not None and limit < len(ids):
            assert batch["input_ids"][0, length - 1] != Tokenizer.eos_token_id


def test_workers_shards_epochs_and_batch_count(tmp_path):
    for file in range(3):
        write(tmp_path / f"{file}.jsonl", *(qa(file * 10 + i) for i in range(10)))
    opts = dict(batch_size=2, max_seq_len=128)
    serial = Loader(Tokenizer(), tmp_path, num_workers=1, **opts)
    first = rows(serial)
    assert serial.num_batches == 15
    assert rows(Loader(Tokenizer(), tmp_path, num_workers=4, **opts)) == first
    serial.set_epoch(1)
    second = rows(serial)
    assert second != first and Counter(second) == Counter(first)
    assert rows(serial) == second
    shards = [rows(Loader(Tokenizer(), tmp_path, rank=r, world_size=3, **opts)) for r in range(3)]
    assert Counter(row for shard in shards for row in shard) == Counter(first)
    assert all(not set(shards[i]) & set(shards[j]) for i in range(3) for j in range(i + 1, 3))


def test_empty_rank_and_invalid_rank(tmp_path):
    write(tmp_path / "one.jsonl", qa())
    loader = Loader(Tokenizer(), tmp_path, rank=1, world_size=2, num_workers=1)
    assert list(loader) == [] and loader.num_batches == 0
    for rank, world in ((-1, 2), (2, 2), (0, 0)):
        with pytest.raises(ValueError, match="invalid rank"):
            Loader(Tokenizer(), tmp_path, rank=rank, world_size=world)


def test_read_and_encode_errors_propagate(tmp_path):
    path = tmp_path / "one.jsonl"
    path.write_text("{broken\n")
    with pytest.raises(ValueError):
        list(Loader(Tokenizer(), path))
    write(path, {"unsupported": True})
    with pytest.raises(ValueError, match="Unsupported record type"):
        list(Loader(Tokenizer(), path))

    class BrokenTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("encoding failed")

    write(path, qa())
    with pytest.raises(RuntimeError, match="encoding failed"):
        list(Loader(BrokenTokenizer(), path))


class TinyTokenizer(Tokenizer):
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 100 + 2 for c in text]


def test_actual_training_with_book_wiki_and_qa(tmp_path):
    write(tmp_path / "data" / "books" / "one.jsonl", text_record("ABC"))
    write(tmp_path / "data" / "wiki" / "two.jsonl", text_record("ABCDE"))
    write(tmp_path / "data" / "qa" / "three.jsonl", qa())
    loader = Loader(TinyTokenizer(), tmp_path / "data", batch_size=3,
                    max_seq_len=128, num_workers=2)
    expected = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in loader)
    args = ModelArgs(name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8,
                     vocab_size=128, ffn_type="dense", intermediate_size=24,
                     num_experts=0, num_activated_experts=0, max_seq_len=128)
    model = Transformer(args)
    before = model.output.weight.detach().clone()
    steps, loss = training.train(
        model, loader, torch.optim.AdamW(model.parameters(), lr=.001),
        output=tmp_path / "run", max_steps=2, save_interval=2, log_interval=1,
    )
    assert steps == 2 and torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.output.weight, before)
    assert training.load_checkpoint(tmp_path / "run")[2]["progress"]["loss_tokens"] == 2 * expected


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("tty", [False, True])
def test_progress_is_single_line_rank_zero_only_and_logs_are_clean(
    tmp_path, monkeypatch, capsys, rank, tty,
):
    import io
    import dataloader.loader as module

    class Stream(io.StringIO):
        def isatty(self):
            return tty

    stream = Stream()
    monkeypatch.setattr(module.sys, "stderr", stream)
    actual_tqdm = module.tqdm
    bars, options = [], []

    def track(*args, **kwargs):
        options.append(kwargs)
        bar = actual_tqdm(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(module, "tqdm", track)
    write(tmp_path / "one.jsonl", *(qa(i) for i in range(4)))
    loader = Loader(Tokenizer(), tmp_path, rank=rank, world_size=2, num_workers=2)
    batches = list(loader)
    assert len(batches) == 2
    tokens = sum(int(b["attention_mask"].sum()) for b in batches)
    assert len(bars) == 1
    assert options[0]["disable"] == (rank != 0 or not tty)
    assert bars[0].pos == 0
    assert options[0]["leave"] is True
    assert options[0]["mininterval"] == 0.5
    assert bars[0].disable  # Context manager closed it before yielding batches.
    output = stream.getvalue()
    assert "\x1b[A" not in output  # Never move up into another rank's/log line.
    if rank == 0 and tty:
        assert "Encoding (rank 0)" in output
        assert bars[0].n == 2
    else:
        assert output == ""
    stdout = capsys.readouterr().out
    if rank == 0:
        assert stdout.splitlines() == [
            "[Rank0] Found 1 jsonl files.",
            f"[Data] Rank 0 shard: samples=2, tokens={tokens:,} "
            "(after truncation, no padding)",
        ]
    else:
        assert stdout == ""


def test_progress_closes_when_encoding_fails(tmp_path, monkeypatch):
    import io
    import dataloader.loader as module

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(module.sys, "stderr", Terminal())
    bars = []
    actual_tqdm = module.tqdm

    def track(*args, **kwargs):
        bar = actual_tqdm(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(module, "tqdm", track)
    write(tmp_path / "one.jsonl", {"unsupported": True})
    with pytest.raises(ValueError, match="Unsupported record type"):
        list(Loader(Tokenizer(), tmp_path, num_workers=1))
    assert len(bars) == 1
    assert bars[0].disable
    assert bars[0].n == 0  # Failed records must not count as successful encodings.



@pytest.mark.parametrize("limit", [None, 64, 256])
def test_summary_counts_all_samples_and_unpadded_retained_tokens(tmp_path, capsys, limit):
    write(tmp_path / "one.jsonl", qa(), text_record("文字" * 300), text_record("ABC"))
    loader = Loader(Tokenizer(), tmp_path, batch_size=2, max_seq_len=limit, num_workers=2)
    batches = iter(loader)
    first = next(batches)
    # Summary is ready before training consumes even the first batch.
    output = capsys.readouterr().out
    assert output.count("[Data] Total:") == 1
    all_batches = [first, *batches]
    samples = sum(b["input_ids"].shape[0] for b in all_batches)
    tokens = sum(int(b["attention_mask"].sum()) for b in all_batches)
    assert samples == 3
    assert f"samples={samples:,}, tokens={tokens:,}" in output
    assert tokens < sum(b["input_ids"].numel() for b in all_batches)
    loader.set_epoch(1)
    list(loader)
    assert capsys.readouterr().out == output  # No cumulative/double counting.


def test_empty_dataset_summary_has_zero_counts(tmp_path, capsys):
    write(tmp_path / "empty.jsonl")
    loader = Loader(Tokenizer(), tmp_path, num_workers=1)
    assert list(loader) == []
    assert "[Data] Total: samples=0, tokens=0" in capsys.readouterr().out


def test_agent_directory_is_excluded_before_reading(tmp_path):
    root = tmp_path / "dataset" / "data"
    normal = write(root / "qa" / "one.jsonl", qa())
    cot = write(root / "cot" / "one.jsonl", {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A", "reasoning_content": "Think"},
        ],
    })
    agent = root / "agent" / "Code_Agent" / "one.jsonl"
    agent.parent.mkdir(parents=True)
    agent.write_text("{invalid: must not be read}\n")
    expected = sorted([normal, cot], key=str)
    assert find_jsonl_files(root) == expected
    assert find_jsonl_files(root.parent) == expected
    assert find_jsonl_files(root / "agent") == []
    assert find_jsonl_files(agent.parent) == []
    assert find_jsonl_files(agent) == []
    batches = list(Loader(Tokenizer(), root, batch_size=2, max_seq_len=128, num_workers=1))
    assert len(batches) == 1 and batches[0]["input_ids"].shape[0] == 2


def test_agent_filter_is_directory_based_not_content_or_substring(tmp_path, monkeypatch):
    # An unrelated workspace ancestor called agent must not hide ordinary data.
    root = tmp_path / "agent" / "workspace" / "data"
    keep = write(root / "qa" / "agent.jsonl", qa())
    keep_other = write(root / "agent_notes" / "one.jsonl", qa(1))
    write(root / "nested" / "agent" / "one.jsonl", qa(2))
    assert set(find_jsonl_files(root)) == {keep, keep_other}
    assert find_jsonl_files(keep) == [keep]
    blocked = write(root / "agent" / "Code_Agent" / "one.jsonl", qa(3))
    monkeypatch.chdir(blocked.parent)
    assert find_jsonl_files(".") == []
    assert find_jsonl_files(blocked.name) == []
