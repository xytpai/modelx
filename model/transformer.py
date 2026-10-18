"""Dense or mixture-of-experts transformer for next-token training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataloader import IGNORE_INDEX

from .config import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., dim] bf16; variance is accumulated in fp32
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * y).to(dtype=x.dtype)


def _rope_cache(head_dim: int, max_seq_len: int, theta: float) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
    return angles.cos(), angles.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [batch, seq, heads, head_dim] bf16, rotated in fp32
    # cos, sin: [seq, head_dim // 2] fp32 -> [seq, 1, head_dim // 2]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = x.float().chunk(2, dim=-1)  # [batch, seq, heads, head_dim // 2] fp32
    y = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return y.to(dtype=x.dtype)  # [batch, seq, heads, head_dim] bf16


class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.n_heads % args.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads
        self.dropout_prob = args.dropout_prob
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.q_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.k_norm = RMSNorm(args.head_dim, args.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # x: [batch, seq, dim] bf16
        # cos, sin: [seq, head_dim // 2] fp32
        batch, seq, dim = x.shape
        # q: [batch, seq, n_heads, head_dim] bf16
        q = self.q_norm(self.wq(x).view(batch, seq, self.n_heads, self.head_dim))
        # k, v: [batch, seq, n_kv_heads, head_dim] bf16
        k = self.k_norm(self.wk(x).view(batch, seq, self.n_kv_heads, self.head_dim))
        v = self.wv(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        # q, k, v: [batch, n_heads, seq, head_dim] bf16
        dropout = self.dropout_prob if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
        # [batch, seq, n_heads * head_dim] bf16 -> [batch, seq, dim] bf16
        y = y.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.head_dim)
        return self.wo(y).view(batch, seq, dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden = (
            args.intermediate_size if args.ffn_type == "dense"
            else args.moe_intermediate_size
        )
        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_activated_experts = args.num_activated_experts
        self.norm_experts_prob = args.norm_experts_prob
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = nn.ModuleList(FeedForward(args) for _ in range(args.num_experts))

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, dim] bf16
        batch, seq, dim = x.shape
        flat = x.reshape(batch * seq, dim)
        # weights, chosen: [batch * seq, num_activated_experts] fp32
        weights = F.softmax(self.gate(flat), dim=-1, dtype=torch.float32)
        weights, chosen = torch.topk(weights, self.num_activated_experts, dim=-1)
        if self.norm_experts_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)

        out = torch.zeros(batch * seq, dim, dtype=torch.float32, device=flat.device)
        for expert_id in torch.unique(chosen).tolist():
            # token_idx, slot: [n]; update: [n, dim] fp32
            token_idx, slot = torch.where(chosen == expert_id)
            update = self.experts[expert_id](flat[token_idx]).float() * weights[token_idx, slot, None]
            out = out.index_add(0, token_idx, update)
        return out.to(dtype=x.dtype).view(batch, seq, dim)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = (
            FeedForward(args) if args.ffn_type == "dense" else MOEFeedForward(args)
        )
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        cos, sin = _rope_cache(args.head_dim, args.max_seq_len, args.rope_theta)
        self.register_buffer("cos", cos, persistent=False)  # fp32
        self.register_buffer("sin", sin, persistent=False)  # fp32

    def bf16(self) -> Transformer:
        """Train in bf16. RoPE tables stay fp32."""
        cos, sin = self.cos.detach().float(), self.sin.detach().float()
        self.to(dtype=torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        return self

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [batch, seq] int64
        _, seq = tokens.shape
        if seq > self.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len {self.max_seq_len}")
        h = self.tok_embeddings(tokens)  # [batch, seq, dim] bf16
        cos, sin = self.cos[:seq], self.sin[:seq]  # [seq, head_dim // 2] fp32
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.output(self.norm(h))  # [batch, seq, vocab] bf16


def token_loss(logits: Tensor, labels: Tensor) -> Tensor:
    # logits: [batch, seq, vocab] bf16, reduced in fp32
    # labels: [batch, seq] int64
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
