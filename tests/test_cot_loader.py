"""CoT field support without changing the original loading pipeline."""
from copy import deepcopy
import json

import pytest

from dataloader import IGNORE_INDEX, PretrainDataLoader


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def cot(loss=False):
    return {
        "id": "NOT_MODEL_INPUT", "language": "en",
        "meta": {"category": "NOT_MODEL_INPUT"}, "tools": [],
        "messages": [
            {"role": "user", "content": "3x=15", "loss": True},
            {"role": "assistant", "reasoning_content": "\nDivide by 3.\n",
             "content": "\nx=5", "loss": loss},
        ],
    }


@pytest.mark.parametrize("loss", [False, True, None])
def test_cot_reasoning_and_answer_are_assistant_targets(loss):
    record = cot(loss)
    before = deepcopy(record)
    loader = PretrainDataLoader(Tokenizer(), "/unused")
    sample = loader._to_sample(record)
    assert type(sample) is list
    assert sample == [
        ("User: ", False), ("3x=15", False),
        ("\nAssistant: ", False),
        ("<think>\nDivide by 3.\n</think>\n", True),
        ("\nx=5", True),
    ]
    ids, labels = loader._encode(sample)
    assert ids == Tokenizer().encode("".join(text for text, _ in sample)) + [0]
    assert [x for x in labels if x != IGNORE_INDEX] == (
        Tokenizer().encode("<think>\nDivide by 3.\n</think>\n\nx=5") + [0]
    )
    assert record == before


@pytest.mark.parametrize("reasoning", [None, ""])
def test_empty_reasoning_keeps_original_qa_format(reasoning):
    loader = PretrainDataLoader(Tokenizer(), "/unused")
    assert loader._to_sample({"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A", "reasoning_content": reasoning},
    ]}) == [("User: ", False), ("Q", False), ("\nAssistant: ", False), ("A", True)]


def test_cot_uses_original_truncation_and_batching(tmp_path):
    path = tmp_path/"one.jsonl"
    path.write_text(json.dumps(cot()) + "\n")
    loader = PretrainDataLoader(Tokenizer(), path)
    sample = loader._to_sample(cot())
    ids, labels = loader._encode(sample)
    for limit in [8, 30, len(ids), len(ids)+10]:
        limited = PretrainDataLoader(Tokenizer(), path, max_seq_len=limit, num_workers=1)
        assert limited._encode(sample) == (ids[:limit], labels[:limit])
        batches = list(limited)
        assert len(batches) == limited.num_batches == 1
        batch = batches[0]
        width = min(limit, len(ids))
        assert batch["input_ids"][0, :width].tolist() == ids[:limit]
        assert batch["labels"][0, :width].tolist() == labels[:limit]


def test_text_continuation_stays_original_qa():
    loader = PretrainDataLoader(Tokenizer(), "/unused")
    record = {"text": "Original text to continue", "meta": {"title": "Book"}}
    sample = loader._to_sample(record)
    assert len(sample) == 4
    assert sample[0] == ("User: ", False)
    assert sample[2] == ("\nAssistant: ", False)
    assert sample[1][0].split("\n原文开头：\n", 1)[1] + sample[3][0] == record["text"]
    assert sample[1][1] is False and sample[3][1] is True
