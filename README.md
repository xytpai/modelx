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

~~~bash
# Log in with an account that has access if the dataset is private.
hf auth login

# Download only data and its checksums; do not overwrite ModelX's README.
hf download xytpai/datax --repo-type dataset --local-dir .

# Verify all downloaded JSONL files.
md5sum -c MD5SUMS
~~~

The current loader supports ordinary user/assistant conversations and book/wiki
text. It loads records and encoded samples into memory rather than streaming,
and tail-truncates sequences exceeding `max_seq_len`.

CoT messages additionally render `reasoning_content` as
`<think>...</think>` before the original `content`. Assistant reasoning and
answers are supervised by role, ignoring source `loss` flags, just like ordinary
QA. Text continuation, final EOS, truncation, reading and batching are unchanged.

Agent tool definitions/calls are not supported in this update. The file finder
temporarily excludes `agent/` directories, including explicit paths under
`data/agent/`. Other subsets, including `cot/`, load normally. This only changes
local file selection; it does not delete data or update the remote dataset.
Long CoT may be truncated by the existing `max_seq_len`; choose the context
length and subset accordingly. No new CLI options or streaming are introduced.

## Training

Run from the repository root. Example: 8-GPU DDP, batch size 4 per GPU
(32 globally). Adjust GPU count, batch size and context length to available
memory; this example is not a memory benchmark.

~~~bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nproc-per-node=8 train.py \
  --model 2B --steps 1000000 --batch-size 4 --max-seq-len 4096 \
  --data-dir /home/yuxu/upstream/datax/data --num-workers 16 --lr 3e-4 \
  --output-dir checkpoints/datax-2B \
  --log-interval 100 --save-interval 2000 --keep-checkpoints 1
~~~

`data/` is an externally prepared local dataset directory, not included in this
repository. Set `--data-dir` to your actual data location if different.
The loader currently excludes Agent directories; long CoT is subject to the
existing 4096-token tail truncation in this example.

A fresh output directory trains from random initialization. Existing checkpoints
resume weights and progress with a fresh AdamW optimizer. `--steps` is the
cumulative optimizer-step target.

Training uses BF16 autocast and FP32 parameters; each DDP rank holds a full
model and optimizer. There is no parameter sharding or activation checkpointing.
The example saves every 2000 steps and retains the newest checkpoint.

## Inference

Run from the repository root after training has saved a checkpoint:

~~~bash
python3 infer.py --model 2B --prompt "你好，请介绍一下自己。"
python3 infer.py --model 2B --prompt "我为什么是我?"
python3 infer.py --model 2B --prompt "世界上最高的是什么山"
python3 infer.py --model 2B --prompt "世界上最可怕的物种是啥？并说明为啥可怕"
~~~

`--model 2B` loads from `checkpoints/datax-2B`, matching the training example.
Use `--checkpoint` to select another checkpoint directory.
Use `--temperature 0.7` for sampling or `--max-new-tokens 256` to change the
generation budget. Decoding is greedy by default; there is no KV cache.

## Tests

~~~bash
python3 -m pytest -q
~~~

## License

MIT; see `LICENSE`. Data and pretrained tokenizer licenses are separate.
