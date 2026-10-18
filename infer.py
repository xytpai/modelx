"""Print a reply from a checkpoint saved by train.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from model import DEFAULT_MODEL, MODELS, ModelArgs, Transformer, default_checkpoint_dir
from train import load_checkpoint


def load_model(checkpoint: Path, device: torch.device):
    args = ModelArgs(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    if len(tokenizer) > args.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds model vocabulary")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required")
    model = Transformer(args)
    _, state, _ = load_checkpoint(checkpoint)
    model.load_state_dict(state, assign=True)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model.bf16()  # Keep RoPE tables in FP32.
    else:
        model.float()  # BF16 checkpoints also work on CPU/non-BF16 GPUs.
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, *, max_new_tokens=128, temperature=0.0):
    # Training encodes headers and content separately, without BOS or a chat template.
    ids = [
        token
        for segment in ("User: ", prompt, "\nAssistant: ")
        for token in tokenizer.encode(segment, add_special_tokens=False)
    ]
    if len(ids) >= model.max_seq_len:
        raise ValueError(
            f"prompt has {len(ids)} tokens; must be shorter than "
            f"max_seq_len={model.max_seq_len} to leave room for a reply"
        )
    device = next(model.parameters()).device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    reply = []
    for _ in range(min(max_new_tokens, model.max_seq_len - len(ids))):
        # No KV cache: recompute the prefix for this small inference test.
        # Exclude vocabulary padding IDs that the tokenizer cannot decode.
        logits = model(tokens)[0, -1, :len(tokenizer)].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite next-token logits")
        if temperature == 0:
            token = logits.argmax().item()
        else:
            probabilities = torch.softmax((logits - logits.max()) / temperature, dim=-1)
            token = torch.multinomial(probabilities, 1).item()
        if token == tokenizer.eos_token_id:
            break
        reply.append(token)
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
    return tokenizer.decode(reply, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=tuple(MODELS),
                        help=f"select the default checkpoint directory (default: {DEFAULT_MODEL}); "
                             "does not override a saved model configuration")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit checkpoint directory; overrides --model's directory "
                             "(default: repository checkpoints/datax-<model>)")
    parser.add_argument("--prompt", required=True, help="user question (quote it in the shell)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for greedy decoding, >0 for sampling (default: 0)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args(argv)
    if not cli.prompt.strip():
        parser.error("--prompt must not be empty")
    if cli.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not math.isfinite(cli.temperature) or cli.temperature < 0:
        parser.error("--temperature must be finite and non-negative")

    torch.manual_seed(cli.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    )
    checkpoint = (
        cli.checkpoint if cli.checkpoint is not None else default_checkpoint_dir(cli.model)
    ).expanduser().resolve()
    print(f"Loading {checkpoint} on {device}...", file=sys.stderr, flush=True)
    model, tokenizer = load_model(checkpoint, device)
    try:
        reply = generate(
            model, tokenizer, cli.prompt,
            max_new_tokens=cli.max_new_tokens, temperature=cli.temperature,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(reply or "(empty reply)", flush=True)


if __name__ == "__main__":
    main()
