import sys
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path


def find_jsonl_files(root: str | Path):
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data path does not exist: {root}")
    # Temporarily disable the Agent subset; leave other data/loading unchanged.
    parts = root.resolve().parts
    if parts[-1] == "agent" or any(
        a == "data" and b == "agent" for a, b in zip(parts, parts[1:])
    ):
        return []
    if root.is_file():
        if root.suffix != ".jsonl":
            raise ValueError("expected a JSONL file or a directory of JSONL files")
        return [root]
    files = [
        path for path in root.rglob("*.jsonl")
        if path.is_file() and "agent" not in path.relative_to(root).parts[:-1]
    ]
    def line_count(path: Path) -> int:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    # Break equal line-count ties identically on every rank.
    return sorted(files, key=lambda path: (line_count(path), str(path)))


def load_jsonl_files(paths, num_workers=16, rank=0, world_size=1):
    """Load records in path order. Keep only index % world_size == rank."""
    def read_file(path):
        records = []
        with Path(path).open("rb") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    paths = iter(paths)
    records = []
    index = 0

    def take(file_records):
        nonlocal index
        for record in file_records:
            if index % world_size == rank:
                records.append(record)
            index += 1

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = deque()
        try:
            for path in islice(paths, num_workers):
                pending.append(pool.submit(read_file, path))
            while pending:
                take(pending.popleft().result())
                for path in islice(paths, 1):
                    pending.append(pool.submit(read_file, path))
        finally:
            for future in pending:
                future.cancel()
    return records


if __name__ == "__main__":
    jsonl_files = find_jsonl_files('../data/')
    datas = load_jsonl_files(jsonl_files)
    print(datas[0])
