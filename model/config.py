"""Aligned presets: dense 2B (default) and 7B, and a 13B-total-parameter MoE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


DEFAULT_MODEL = "2B"


@dataclass
class ModelArgs:
    name: str = "13B"
    dim: int = 3072
    n_layers: int = 36
    n_heads: int = 24
    n_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 152064
    moe_intermediate_size: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    max_seq_len: int = 4096
    dropout_prob: float = 0.0
    num_experts: int = 8
    num_activated_experts: int = 2
    norm_experts_prob: bool = True
    # Default preserves configs/checkpoints saved before dense FFNs were supported.
    ffn_type: str = "moe"
    intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if self.ffn_type not in {"moe", "dense"}:
            raise ValueError("ffn_type must be 'moe' or 'dense'")
        if self.ffn_type == "dense" and (
            self.intermediate_size is None or self.intermediate_size <= 0
        ):
            raise ValueError("dense FFNs require a positive intermediate_size")


# ModelArgs() is the 13B MoE configuration. User-facing APIs/CLI default to 2B.
# Keep ffn_type="moe" as the dataclass fallback for fully specified legacy configs.
MODELS = {
    # ~1.588B total, including untied embeddings and output weights.
    # Preset names round total parameters to the nearest integer billion.
    "2B": ModelArgs(
        name="2B",
        dim=2048,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        head_dim=128,
        vocab_size=152064,
        ffn_type="dense",
        intermediate_size=6144,
        num_experts=0,
        num_activated_experts=0,
    ),
    "7B": ModelArgs(
        name="7B",
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        vocab_size=152064,
        ffn_type="dense",
        intermediate_size=12288,
        num_experts=0,
        num_activated_experts=0,
    ),
    # 3072 = 6*512; expert width 4096 = 8*512; KV width 8*128 = 2*512.
    # 8 experts per layer, top-2 routing; 13B refers to total stored parameters.
    "13B": ModelArgs(),
}


def parameter_count(args: ModelArgs) -> int:
    embed = 2 * args.vocab_size * args.dim + args.dim
    attention = args.dim * args.head_dim * (2 * args.n_heads + 2 * args.n_kv_heads)
    norms = 2 * args.head_dim + 2 * args.dim
    if args.ffn_type == "dense":
        feed_forward = 3 * args.dim * args.intermediate_size
    else:
        gate = args.dim * args.num_experts
        experts = args.num_experts * 3 * args.dim * args.moe_intermediate_size
        feed_forward = gate + experts
    return embed + args.n_layers * (attention + norms + feed_forward)


def model_name(num_params: int) -> str:
    return f"{round(num_params / 1_000_000_000)}B"


def get_config(name: str = DEFAULT_MODEL, max_seq_len: int | None = None) -> ModelArgs:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"unknown model {name!r}; choose from {known}")
    args = replace(MODELS[name], name=name)
    if max_seq_len is None:
        return args
    return replace(args, max_seq_len=max_seq_len)


def default_checkpoint_dir(name: str = DEFAULT_MODEL) -> Path:
    """Repository-local default shared by training and inference.

    Validate the preset name without allocating any model weights.
    Explicit checkpoint/output paths remain caller-controlled.
    """
    model_name = get_config(name).name
    return Path(__file__).resolve().parents[1] / "checkpoints" / f"datax-{model_name}"


for _name, _args in MODELS.items():
    _counted = model_name(parameter_count(_args))
    if _counted != _name:
        raise ValueError(f"{_name} config has {parameter_count(_args):,} parameters ({_counted})")
