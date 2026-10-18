"""Dense preset accounting, aligned shapes and small-model functional regressions."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from model import MODELS, ModelArgs, Transformer, get_config, token_loss
from model.config import parameter_count
from model.transformer import FeedForward, MOEFeedForward
import train as training


def tiny_dense():
    # Small test dimensions need not have production-preset alignment.
    return ModelArgs(
        name="tiny-dense", dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=8, vocab_size=64, ffn_type="dense", intermediate_size=96,
        num_experts=0, num_activated_experts=0, max_seq_len=16,
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_all_preset_parameter_counts_match_real_modules_on_meta(name):
    args = get_config(name)
    with torch.device("meta"):
        model = Transformer(args)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(args)


@pytest.mark.parametrize("preset,dim,hidden,layers,heads,kv_heads,expected", [
    ("2B", 2048, 6144, 20, 16, 4, 1_587_633_152),
    ("7B", 4096, 12288, 32, 32, 8, 7_419_998_208),
])
def test_dense_presets_have_512_aligned_matrix_dimensions_and_no_routing(
    preset, dim, hidden, layers, heads, kv_heads, expected,
):
    args = get_config(preset)
    assert args.ffn_type == "dense"
    assert (args.dim, args.intermediate_size, args.n_layers) == (dim, hidden, layers)
    assert (args.n_heads, args.n_kv_heads, args.head_dim) == (heads, kv_heads, 128)
    assert args.vocab_size == 152064
    assert args.intermediate_size == 3 * args.dim
    assert args.num_experts == args.num_activated_experts == 0
    assert parameter_count(args) == expected
    with torch.device("meta"):
        model = Transformer(args)
    assert sum(p.numel() for p in model.parameters()) == expected
    for name, p in model.named_parameters():
        if p.ndim == 2:
            assert all(size % 512 == 0 for size in p.shape), (name, p.shape)
    assert all(type(layer.feed_forward) is FeedForward for layer in model.layers)
    assert not any(isinstance(module, MOEFeedForward) for module in model.modules())
    assert not any(".gate." in name or ".experts." in name for name, _ in model.named_parameters())
    assert model.output.weight is not model.tok_embeddings.weight
    assert model.layers[0].attention.wk.weight.shape == (kv_heads * 128, dim)
    assert model.layers[0].feed_forward.w1.weight.shape == (hidden, dim)
    assert model.layers[0].feed_forward.w2.weight.shape == (dim, hidden)


@pytest.mark.parametrize("preset", ["2B", "7B"])
def test_dense_get_config_is_independent_and_round_trips_checkpoint_json(preset):
    args = get_config(preset, max_seq_len=128)
    assert args.max_seq_len == 128
    assert MODELS[preset].max_seq_len == 4096
    assert ModelArgs(**json.loads(json.dumps(asdict(args)))) == args
    args.dim = 512
    assert get_config(preset).dim == MODELS[preset].dim
    tiny = tiny_dense()
    loaded = ModelArgs(**json.loads(json.dumps(asdict(tiny))))
    assert loaded == tiny
    assert parameter_count(loaded) == parameter_count(tiny)


@pytest.mark.parametrize("ffn_type,hidden", [("invalid", 96), ("dense", None), ("dense", 0), ("dense", -1)])
def test_invalid_dense_configuration_is_rejected(ffn_type, hidden):
    with pytest.raises(ValueError):
        ModelArgs(ffn_type=ffn_type, intermediate_size=hidden)


def test_dense_ffn_uses_its_own_width_and_swiglu_formula():
    args = replace(tiny_dense(), moe_intermediate_size=7)
    ffn = FeedForward(args)
    x = torch.randn(2, 3, args.dim)
    expected = ffn.w2(torch.nn.functional.silu(ffn.w1(x)) * ffn.w3(x))
    torch.testing.assert_close(ffn(x), expected)
    assert ffn.w1.out_features == args.intermediate_size
    assert parameter_count(args) == parameter_count(tiny_dense())


@pytest.mark.parametrize("autocast", [False, True])
def test_small_dense_forward_backward_optimizer_and_causality(autocast):
    torch.manual_seed(23)
    args = tiny_dense()
    model = Transformer(args)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(args)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    labels = tokens.clone()
    labels[:, :2] = -100
    before = model.layers[0].feed_forward.w1.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        logits = model(tokens)
        loss = token_loss(logits[:, :-1], labels[:, 1:])
    assert logits.shape == (2, 6, 64)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    optimizer.step()
    assert not torch.equal(before, model.layers[0].feed_forward.w1.weight)
    model.eval()
    changed = tokens.clone()
    changed[:, 3:] = 7
    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[:, :3], model(changed)[:, :3])
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.ones(1, 17, dtype=torch.long))


def test_old_json_without_dense_fields_keeps_moe_architecture_and_state_dict():
    args = replace(tiny_dense(), ffn_type="moe", intermediate_size=None,
                   moe_intermediate_size=48, num_experts=2, num_activated_experts=1)
    legacy_dict = asdict(args)
    legacy_dict.pop("ffn_type")
    legacy_dict.pop("intermediate_size")
    restored_args = ModelArgs(**json.loads(json.dumps(legacy_dict)))
    assert restored_args.ffn_type == "moe"
    model, restored = Transformer(args), Transformer(restored_args)
    assert all(isinstance(layer.feed_forward, MOEFeedForward) for layer in restored.layers)
    restored.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), restored(tokens), rtol=0, atol=0)


def test_dense_bf16_checkpoint_round_trip_and_rope_precision(tmp_path):
    args = tiny_dense()
    model = Transformer(args).bf16()
    assert model.cos.dtype == model.sin.dtype == torch.float32
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    training.save_checkpoint(tmp_path, model.state_dict(), {"step": 1}, [(1, 2.0)])
    _, weights, metadata = training.load_checkpoint(tmp_path)
    args = ModelArgs(**json.loads(json.dumps(asdict(args))))
    restored = Transformer(args).bf16()
    restored.load_state_dict(weights, strict=True)
    model.eval()
    restored.eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(model(tokens), restored(tokens), rtol=0, atol=0)
    assert metadata["progress"]["step"] == 1


def test_cli_lists_dense_preset(capsys):
    with pytest.raises(SystemExit) as exc:
        training.main(["--help"])
    assert exc.value.code == 0
    assert "--model {2B,7B,13B}" in capsys.readouterr().out
