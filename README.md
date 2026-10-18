# ModelX

Model architecture, training, and inference code, separated from
[DataX](https://huggingface.co/datasets/xytpai/datax).

This is a code-only migration of the original DataX implementation. Model
behavior, checkpoint formats, and runtime defaults are unchanged. No datasets,
trained weights, caches, or source repository history are included.

## Layout

- `model/`: Transformer architecture and model presets.
- `dataloader/`: JSONL loading, sequence construction, and loss masking.
- `train.py`: training, distributed execution, and checkpoint handling.
- `infer.py`: inference from a trained checkpoint.
- `tests/`: existing model, loader, training, inference, and utility tests.
- `scripts/fetch_inst.py`: legacy optional Tulu importer, retained unchanged.

## Setup

Python 3.10+ is required by the source syntax. Install a PyTorch build appropriate
for your CPU/CUDA environment, then install the remaining runtime dependencies:

~~~bash
pip install torch transformers tqdm
pip install pytest
~~~

The default tokenizer is `Qwen/Qwen2.5-0.5B`. Its initial download requires
network access; `--tokenizer` can point to a local tokenizer instead.

## Models

| Preset | Architecture | Parameters | Layers | Hidden size |
| --- | --- | ---: | ---: | ---: |
| `2B` (default) | Dense SwiGLU | 1.588B | 20 | 2048 |
| `7B` | Dense SwiGLU | 7.420B | 32 | 4096 |
| `13B` | MoE, 8 experts / top-2 | 12.713B total | 36 | 3072 |

All presets use a 4096-token context and a 152064-entry vocabulary.
Legacy default checkpoint paths remain `checkpoints/datax-<model>` for
compatibility. Use explicit output/checkpoint paths if preferred.

## External data

This repository contains **no training data**. Point `--data-dir` at an external
JSONL file or directory. The separate dataset repository is
[xytpai/datax](https://huggingface.co/datasets/xytpai/datax).

The current loader supports ordinary user/assistant conversations and book/wiki
text. It loads records and encoded samples into memory rather than streaming,
and tail-truncates sequences exceeding `max_seq_len`.

**Agent/CoT support is not added by this migration.** The existing loader reads
message `role` and `content` only; it does not render `tools`, `tool_calls`, or
`reasoning_content`, and does not apply per-message `loss` flags. Adapt the loader
and inference template before using the Agent or CoT subsets for full-fidelity
training.

## Training

Start with a small external plain-QA shard and a batch size that fits your device:

~~~bash
python3 train.py --model 2B --steps 1000 --batch-size 1 --max-seq-len 4096 --data-dir ../datax/data/qa --output-dir checkpoints/modelx-2B
~~~

The sibling path is an example; use your actual external data location.
A fresh output directory trains from random initialization. Existing checkpoints
resume weights and progress with a fresh AdamW optimizer.

For multiple GPUs, use `torchrun --standalone --nproc-per-node=N train.py ...`.
Training uses BF16 autocast and FP32 parameters; each DDP rank holds a full
model and optimizer. There is no parameter sharding or activation checkpointing.

## Inference

Inference requires a checkpoint produced by training:

~~~bash
python3 infer.py --checkpoint checkpoints/modelx-2B --prompt "Explain how binary search works."
~~~

Use `--temperature 0.7` for sampling or `--max-new-tokens 256` to change the
generation budget. Decoding is greedy by default; there is no KV cache.

## Tests

~~~bash
python3 -m pytest -q
~~~

## License

MIT; see `LICENSE`. Data and pretrained tokenizer licenses are separate.
