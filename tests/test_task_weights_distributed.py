"""Repetition with the existing two-rank DDP loop; no task synchronization."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys


def test_repeated_tasks_two_ranks(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
         str(Path(__file__).resolve()), str(tmp_path)],
        env=env, text=True, capture_output=True, timeout=100,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    reports = [json.loads((tmp_path/f"rank-{r}.json").read_text()) for r in range(2)]
    assert all(r["ok"] for r in reports)
    assert not set(reports[0]["answers"]) & set(reports[1]["answers"])
    counts = Counter(reports[0]["answers"]) + Counter(reports[1]["answers"])
    assert counts == Counter({
        **{f"qa:{i}": 3 for i in range(5)},
        **{f"cot:{i}": 2 for i in range(11)},
    })


def worker(root):
    from datetime import timedelta
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from dataloader import PretrainDataLoader
    from model import ModelArgs, Transformer
    import train

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    dist.init_process_group("gloo", timeout=timedelta(seconds=40))
    rank = dist.get_rank()
    try:
        if rank == 0:
            for task, n in [("qa", 5), ("cot", 11)]:
                p = root/"data"/task/"one.jsonl"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("".join(json.dumps({"messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": f"{task}:{i}"},
                ]})+"\n" for i in range(n)))
        dist.barrier()
        loader = PretrainDataLoader(Tokenizer(), root/"data", task_weights="qa=3,cot=2",
                                   max_seq_len=128, batch_size=2, rank=rank, world_size=2,
                                   seed=91, num_workers=1)
        answers = []
        for batch in loader:
            for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
                text = "".join(chr(i) for i in ids[mask.bool()].tolist() if i)
                answers.append(text.split("Assistant: ")[1])
        expected = {f"qa:{i}": 3 for i in range(rank, 5, 2)}
        expected.update({f"cot:{i}": 2 for i in range(rank, 11, 2)})
        assert Counter(answers) == Counter(expected)
        torch.manual_seed(12)
        cfg = ModelArgs(name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
                        head_dim=8, vocab_size=128, ffn_type="dense", intermediate_size=24,
                        num_experts=0, num_activated_experts=0, max_seq_len=128)
        model = DDP(Transformer(cfg), broadcast_buffers=False)
        step, loss = train.train(model, loader, torch.optim.SGD(model.parameters(), lr=.01),
                                 output=root/"checkpoint", max_steps=10, log_interval=2, save_interval=10)
        assert step == 10 and torch.isfinite(torch.tensor(loss))
        values = torch.cat([p.detach().flatten() for p in model.parameters()])
        reference = values.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(values, reference)
        (root/f"rank-{rank}.json").write_text(json.dumps({"ok": True, "answers": answers}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
