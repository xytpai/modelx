"""Fixed code-file subsampling; all other loading behavior stays the same."""
from collections import Counter
import json
import random

import pytest

from dataloader.utils import find_jsonl_files, load_jsonl_files


def write(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": index})+"\n")
    return path


@pytest.mark.parametrize("count,expected", [(0,0),(1,1),(9,1),(10,1),(11,2),(20,2),(107,11)])
def test_code_file_fraction_and_other_tasks_unchanged(tmp_path, count, expected):
    root=tmp_path/"data"
    code=[write(root/"code"/f"{i:05d}.jsonl",i) for i in range(count)]
    # Other task, similarly named folder and file must not be subsampled.
    others=[write(root/"qa"/"code.jsonl","qa"),
            write(root/"cot"/"one.jsonl","cot"),
            write(root/"code_notes"/"one.jsonl","notes")]
    state=random.getstate()
    result=find_jsonl_files(root)
    assert random.getstate()==state
    selected=sorted(code,key=str)
    random.Random(42).shuffle(selected)
    assert set(result)==set(others+selected[:expected])
    assert result==find_jsonl_files(root)
    assert len([p for p in result if p.parent.name=="code"])==expected
    assert all(p.exists() for p in code+others)


def test_parent_and_code_directory_select_same_shards(tmp_path,monkeypatch):
    root=tmp_path/"data"
    for i in range(31):
        write(root/"code"/f"{i:05d}.jsonl",i)
    assert find_jsonl_files(root)==find_jsonl_files(root/"code")
    assert find_jsonl_files(tmp_path)==find_jsonl_files(root)
    monkeypatch.chdir(root/"code")
    assert {p.resolve() for p in find_jsonl_files(".")}==set(find_jsonl_files(root))
    explicit=root/"code"/"00000.jsonl"
    assert find_jsonl_files(explicit)==[explicit]


def test_discarded_files_are_not_read_even_for_line_count(tmp_path,monkeypatch):
    root=tmp_path/"data"
    code=[write(root/"code"/f"{i:05d}.jsonl",i) for i in range(30)]
    random.Random(42).shuffle(code)
    excluded=set(code[3:])
    from pathlib import Path
    original=Path.open
    def checked_open(self,*args,**kwargs):
        assert self not in excluded, "Excluded code shards must not be opened"
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"open",checked_open)
    assert set(find_jsonl_files(root))==set(code[:3])


def test_sampling_keeps_original_rank_record_partition(tmp_path):
    root=tmp_path/"data"
    for i in range(100):
        write(root/"code"/f"{i:05d}.jsonl",i)
    for i in range(5):
        write(root/"qa"/f"{i:05d}.jsonl",100+i)
    paths=find_jsonl_files(root)
    full=load_jsonl_files(paths,num_workers=2)
    ranks=[load_jsonl_files(find_jsonl_files(root),num_workers=2,rank=r,world_size=3)
           for r in range(3)]
    assert Counter(r["id"] for rows in ranks for r in rows)==Counter(r["id"] for r in full)
    assert all(ranks[r]==full[r::3] for r in range(3))
