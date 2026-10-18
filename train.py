"""Train with BF16 autocast and checkpoints containing loss history and progress."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
from itertools import chain, islice
import json
import os
import pickle
from pathlib import Path
import warnings

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from dataloader import IGNORE_INDEX, PretrainDataLoader
from model import (
    DEFAULT_MODEL, MODELS, ModelArgs, Transformer,
    default_checkpoint_dir, get_config, token_loss,
)

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_KEEP_CHECKPOINTS = 3


def _validate_keep_checkpoints(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("keep_checkpoints must be an integer >= 1")


def checkpoint_paths(output: Path) -> list[Path]:
    return sorted(
        (p for p in output.glob("model-*.pt") if p.stem[6:].isdigit()),
        key=lambda p: int(p.stem[6:]),
    )


def load_checkpoint(output: Path):
    """Load the newest readable checkpoint; no format/version marker required."""
    for path in reversed(checkpoint_paths(output)):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            return path, payload["weights"], {
                "progress": payload["progress"], "loss_history": payload["loss_history"],
            }
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, TypeError, AttributeError, pickle.UnpicklingError) as exc:
            warnings.warn(f"Skipping unusable checkpoint {path.name}: {exc}", RuntimeWarning)
    raise RuntimeError(f"no usable checkpoint in {output}")


def save_checkpoint(
    output: Path, state: dict, progress: dict, loss_history: list, *,
    keep_checkpoints: int = DEFAULT_KEEP_CHECKPOINTS,
) -> Path:
    """Publish a verified checkpoint before pruning; safe with keep_checkpoints=1."""
    _validate_keep_checkpoints(keep_checkpoints)
    output.mkdir(parents=True, exist_ok=True)
    serial = max((int(p.stem[6:]) for p in checkpoint_paths(output)), default=0) + 1
    path = output / f"model-{serial:08d}.pt"
    state = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in state.items()}
    payload = dict(weights=state, progress=progress, loss_history=loss_history)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        verified = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if verified["progress"] != progress or verified["loss_history"] != loss_history or verified["weights"].keys() != state.keys() or any(
            v.dtype != torch.bfloat16 or not torch.equal(v, state[k])
            for k, v in verified["weights"].items()
        ):
            raise ValueError("checkpoint read-back verification failed")
        del verified
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    for old in checkpoint_paths(output)[:-keep_checkpoints]:
        old.unlink()
    return path


def train(model, loader, optimizer, *, output, max_steps,
          log_interval=100, save_interval=1000, progress=None, loss_history=None,
          keep_checkpoints: int = DEFAULT_KEEP_CHECKPOINTS):
    """Repeat the current data until the cumulative step target is reached."""
    _validate_keep_checkpoints(keep_checkpoints)
    progress = progress or {}
    loss_history = list(loss_history) if loss_history is not None else []
    steps, loss_value = progress.get("step", 0), progress.get("loss")
    input_tokens, loss_tokens = progress.get("input_tokens", 0), progress.get("loss_tokens", 0)
    device = next(model.parameters()).device
    distributed = isinstance(model, DDP)
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    raw_model = model.module if distributed else model
    model.train()
    while steps < max_steps:
        start_step = steps
        loader.set_epoch(steps)  # Loader's shuffle API; use global step, not a pass counter.
        batches = iter(loader)
        try:
            if distributed:
                first = next(batches, None)  # num_batches is known only after loading.
                common = torch.tensor(loader.num_batches if first is not None else 0, device=device)
                dist.all_reduce(common, op=dist.ReduceOp.MIN)
                if common.item() == 0:
                    raise ValueError("no training samples on at least one rank; use fewer GPUs or more data")
                current_batches = islice(chain((first,), batches), int(common.item()))
            else:
                current_batches = batches
            for batch in current_batches:
                tokens, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    # QA and text-continuation share the same mask and one shift.
                    # User (including the text opening) is context, not a target.
                    loss = token_loss(model(tokens)[:, :-1], labels[:, 1:])
                batch_inputs = int(batch["attention_mask"].sum())
                batch_targets = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
                if distributed:
                    totals = torch.stack((
                        loss.detach().double() * batch_targets,
                        loss.new_tensor(batch_targets, dtype=torch.float64),
                        loss.new_tensor(batch_inputs, dtype=torch.float64),
                    ))
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                    # DDP averages gradients: compensate to average over global target tokens.
                    (loss * (batch_targets * world_size / totals[1].item())).backward()
                    loss_value = (totals[0] / totals[1]).item()
                    batch_targets, batch_inputs = int(totals[1].item()), int(totals[2].item())
                else:
                    loss.backward()
                    loss_value = loss.detach().item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                input_tokens += batch_inputs
                loss_tokens += batch_targets
                if steps % log_interval == 0 or steps % save_interval == 0:
                    record = dict(
                        step=steps, loss=loss_value,
                        lr=optimizer.param_groups[0]["lr"],
                        input_tokens=input_tokens, loss_tokens=loss_tokens,
                    )
                    if steps % log_interval == 0:
                        loss_history.append((steps, loss_value))
                        if rank == 0:
                            print(json.dumps(record), flush=True)
                    if steps % save_interval == 0:
                        if rank == 0:
                            save_checkpoint(
                                output, raw_model.state_dict(), record, loss_history,
                                keep_checkpoints=keep_checkpoints,
                            )
                        if distributed:
                            dist.barrier()  # Do not train ahead while rank 0 writes the queue.
                if steps >= max_steps:
                    return steps, loss_value
        finally:
            batches.close()
        if steps == start_step:
            raise ValueError("no training samples")
    return steps, loss_value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=tuple(MODELS),
                        help=f"model preset (default: {DEFAULT_MODEL})")
    parser.add_argument("--steps", type=int, required=True, help="cumulative optimizer-step target, including resumed steps")
    parser.add_argument("--log-interval", type=int, default=100, help="print loss and collect (step, loss) pairs every N optimizer steps (default: 100)")
    parser.add_argument("--save-interval", type=int, default=1000, help="save BF16 weights every N optimizer steps (default: 1000)")
    parser.add_argument("--keep-checkpoints", type=int, default=DEFAULT_KEEP_CHECKPOINTS,
                        help="number of newest checkpoints to retain after each successful save "
                             f"(>=1, default: {DEFAULT_KEEP_CHECKPOINTS})")
    parser.add_argument("--batch-size", type=int, default=1, help="samples per GPU (global batch is batch-size * world-size)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME, help="Hugging Face tokenizer name or local directory")
    parser.add_argument("--data-dir", default=None, help="recursively read all *.jsonl in a directory; accepts QA messages and book/wiki text (default: repository data/)")
    parser.add_argument("--output-dir", type=Path, default=None, help="checkpoint/output directory; existing checkpoints are resumed (default: repository checkpoints/datax-<model>)")
    cli = parser.parse_args(argv)
    try:
        _validate_keep_checkpoints(cli.keep_checkpoints)
    except ValueError as exc:
        parser.error(str(exc))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        run(cli)
    finally:
        if distributed:
            dist.destroy_process_group()


def run(cli) -> None:
    distributed = dist.is_initialized()
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    output = cli.output_dir if cli.output_dir is not None else default_checkpoint_dir(cli.model)
    resume = [bool(checkpoint_paths(output)) if rank == 0 else False]
    if distributed:
        dist.broadcast_object_list(resume, src=0)
    resume = resume[0]
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    if resume:
        args = ModelArgs(**json.loads((output / "config.json").read_text(encoding="utf-8")))
        tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer", local_files_only=True)
    else:
        args = get_config(cli.model, max_seq_len=cli.max_seq_len)
        tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer)
    model = Transformer(args)  # FP32 training parameters; exported weights are BF16.
    progress, loss_history = {}, []
    if resume:
        if rank == 0:
            checkpoint, weights, saved = load_checkpoint(output)
            model.load_state_dict(weights)
            del weights
            progress, loss_history = saved["progress"], saved["loss_history"]
            print(f"Resuming {checkpoint} at step={progress.get('step', 0)}; fresh AdamW.", flush=True)
        if distributed:
            saved_info = [progress, loss_history]
            dist.broadcast_object_list(saved_info, src=0)
            progress, loss_history = saved_info
    model.to(device)
    if distributed:
        # DDP broadcasts rank-0 weights. Only MoE routing can leave experts unused;
        # dense FFNs use every block on every forward pass.
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                    broadcast_buffers=False, find_unused_parameters=args.ffn_type == "moe")
    torch.manual_seed(cli.seed + rank)
    loader = PretrainDataLoader(
        tokenizer, root=cli.data_dir if cli.data_dir is not None else DATA_DIR,
        batch_size=cli.batch_size,
        max_seq_len=args.max_seq_len, num_workers=cli.num_workers, seed=cli.seed,
        rank=rank, world_size=world_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.lr)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not resume:
            (output / "config.json").write_text(json.dumps(asdict(args), indent=2) + "\n", encoding="utf-8")
            tokenizer.save_pretrained(output / "tokenizer")
        print(f"{args.name}: world_size={world_size}, batch_per_gpu={cli.batch_size}, "
              f"device={device}, autocast=bf16, checkpoint=bf16", flush=True)
    if distributed:
        dist.barrier()
    steps, loss = train(
        model, loader, optimizer, output=output, max_steps=cli.steps,
        log_interval=cli.log_interval, save_interval=cli.save_interval,
        progress=progress, loss_history=loss_history,
        keep_checkpoints=cli.keep_checkpoints,
    )
    if rank == 0:
        print(f"Finished: step={steps}, loss={loss}; weights saved at save intervals only.", flush=True)


if __name__ == "__main__":
    main()
