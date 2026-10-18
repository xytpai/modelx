import random
import sys
import torch
import torch.distributed as dist
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .utils import find_jsonl_files, load_jsonl_files
except ImportError:
    from utils import find_jsonl_files, load_jsonl_files

IGNORE_INDEX = -100


class PretrainDataLoader:
    def __init__(self, tokenizer, root, batch_size=1, max_seq_len=None, seed=42, num_workers=16, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world_size {world_size}")
        self.tokenizer = tokenizer
        self.root = root
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.epoch = 0
        self.num_batches = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def _messages_to_sample(self, messages):
        sample = []
        for index, message in enumerate(messages):
            header = ("\n" if index else "") + f"{message['role'].capitalize()}: "
            sample.append((header, False))
            sample.append((message["content"], message["role"] == "assistant"))
        return sample
    
    def _text_to_sample(self, record):
        text = record["text"]
        meta = record.get("meta") or {}
        fields = []
        title = meta.get("title")
        chapters = meta.get("chapters")
        if isinstance(title, str) and title.strip():
            fields.append(f"title: {title.strip()}")
        if isinstance(chapters, list):
            names = [item.strip() for item in chapters if isinstance(item, str) and item.strip()]
            if names:
                fields.append("chapters: " + " / ".join(names))
        cut = min(128, max(1, len(text) // 10))
        if cut >= len(text):
            cut = len(text) - 1
        opening, answer = text[:cut], text[cut:]
        user = "根据以下原文开头续写后文，保持原文风格，不要重复开头。"
        if fields:
            user += "\n" + "\n".join(fields)
        user += "\n原文开头：\n" + opening
        return self._messages_to_sample([
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ])

    def _to_sample(self, record):
        if "messages" in record:
            return self._messages_to_sample(record["messages"])
        elif "text" in record:
            return self._text_to_sample(record)
        else:
            raise ValueError(f"Unsupported record type: {record}")

    def _encode(self, sample):
        """
        Return (ids, labels). Prompts and role headers are -100; the answer and the final EOS keep their token ids.
        
        sample = [
            ("User: ", False),
            ("问", False),
            ("\nAssistant: ", False),
            ("答", True),
        ]
        
        Returns:
        (
            [85, 115, 101, 114, 58, 32, 38382, 10, 65, 115, 115, 105, 115, 116, 97, 110, 116, 58, 32, 31572, 0],
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 31572, 0],
        )
        """
        ids, labels = [], []
        for text, train_on_text in sample:
            tokens = [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]
            ids.extend(tokens)
            labels.extend(tokens if train_on_text else [IGNORE_INDEX] * len(tokens))
        ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)
        if self.max_seq_len is not None and len(ids) > self.max_seq_len:
            # Drop the tail, keeping token IDs and loss masks aligned.
            ids = ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        return ids, labels

    def _collate(self, rows):
        batch_size = len(rows)
        width = max(len(ids) for ids, _ in rows)
        width = ((width + 127) // 128) * 128
        if self.max_seq_len is not None:
            width = min(width, self.max_seq_len)
        input_ids = torch.full((batch_size, width), self.eos_token_id, dtype=torch.long)
        labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        for index, (ids, targets) in enumerate(rows):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids)
            labels[index, :length] = torch.tensor(targets)
            attention_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __iter__(self):
        jsonl_files = find_jsonl_files(self.root)
        if self.rank == 0:
            print(f"[Rank0] Found {len(jsonl_files)} jsonl files.", flush=True)
        records = load_jsonl_files(jsonl_files, self.num_workers, self.rank, self.world_size)

        def _encode_record(record):
            return self._encode(self._to_sample(record))
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_encode_record, record): index
                for index, record in enumerate(records)
            }
            encoded = [None] * len(records)
            # One terminal line, owned by rank 0; no control codes in log files.
            with tqdm(
                as_completed(futures), total=len(futures),
                desc="Encoding (rank 0)", unit="sample",
                leave=True, dynamic_ncols=True, mininterval=0.5,
                disable=self.rank != 0 or not sys.stderr.isatty(),
            ) as progress:
                for future in progress:
                    encoded[futures[future]] = future.result()

        encoded.sort(key=lambda row: len(row[0]))
        batches = [
            encoded[start:start + self.batch_size]
            for start in range(0, len(encoded), self.batch_size)
        ]
        self.num_batches = len(batches)

        counts = [len(encoded), sum(len(ids) for ids, _ in encoded)]
        distributed = self.world_size > 1 and dist.is_initialized()
        if distributed:
            device = "cuda" if dist.get_backend() == "nccl" else "cpu"
            counts = torch.tensor(counts, dtype=torch.int64, device=device)
            dist.all_reduce(counts)
            counts = counts.tolist()
        if self.rank == 0:
            scope = "Total" if distributed or self.world_size == 1 else "Rank 0 shard"
            print(f"[Data] {scope}: samples={counts[0]:,}, tokens={counts[1]:,} "
                  "(after truncation, no padding)", flush=True)

        random.Random(self.seed + self.epoch).shuffle(batches)
        for rows in batches:
            yield self._collate(rows)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    loader = PretrainDataLoader(tokenizer, "../data/qa", batch_size=2, max_seq_len=4096)
    for batch in loader:
        print(batch["input_ids"].shape, batch["labels"].shape)
        break
