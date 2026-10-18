"""The public presets, exact shape accounting and backward-compatible config loading."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from model import DEFAULT_MODEL, MODELS, ModelArgs, Transformer, default_checkpoint_dir, get_config
from model.config import model_name, parameter_count
from model.transformer import MOEFeedForward
import infer
import train as training


def test_plain_2b_7b_and_13b_presets_and_default():
    assert list(MODELS) == ["2B", "7B", "13B"]
    assert DEFAULT_MODEL == "2B"
    assert get_config() == get_config("2B")
    assert get_config().ffn_type == "dense"
    assert ModelArgs() == get_config("13B")
    assert get_config("13B").ffn_type == "moe"


@pytest.mark.parametrize("name", ["1B", "3B", "2B-dense", "7B-dense", "7B-moe", "13B-moe"])
def test_removed_or_suffixed_presets_are_not_aliases(name):
    with pytest.raises(KeyError, match="choose from 2B, 7B, 13B"):
        get_config(name)


@pytest.mark.parametrize("name", list(MODELS))
def test_config_copy_override_serialization_and_plain_name(name):
    cfg = get_config(name, max_seq_len=1024)
    assert cfg.name == name and "-" not in cfg.name
    assert cfg.max_seq_len == 1024
    assert MODELS[name].max_seq_len == 4096
    assert ModelArgs(**json.loads(json.dumps(asdict(cfg)))) == cfg
    cfg.n_layers = 1
    assert get_config(name).n_layers == MODELS[name].n_layers
    assert model_name(parameter_count(get_config(name))) == name


def test_13b_real_moe_shapes_and_exact_total_on_meta():
    args = get_config("13B")
    assert (args.dim, args.n_layers, args.moe_intermediate_size) == (3072, 36, 4096)
    assert (args.n_heads, args.n_kv_heads, args.head_dim) == (24, 8, 128)
    assert (args.num_experts, args.num_activated_experts) == (8, 2)
    assert args.norm_experts_prob
    assert args.vocab_size == 152064
    with torch.device("meta"):
        model = Transformer(args)
    total = sum(p.numel() for p in model.parameters())
    assert total == parameter_count(args) == 12_713_005_056
    selected_path = total - args.n_layers * 6 * 3 * args.dim * args.moe_intermediate_size
    assert selected_path == 4_559_278_080
    routers = 0
    for layer in model.layers:
        assert isinstance(layer.feed_forward, MOEFeedForward)
        assert len(layer.feed_forward.experts) == 8
        assert layer.feed_forward.num_activated_experts == 2
        assert tuple(layer.feed_forward.experts[0].w1.weight.shape) == (4096, 3072)
        assert tuple(layer.feed_forward.experts[0].w2.weight.shape) == (3072, 4096)
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if ".feed_forward.gate." in name:
            routers += 1
            assert tuple(p.shape) == (8, 3072)
        else:
            assert all(d % 512 == 0 for d in p.shape), (name, p.shape)
    assert routers == 36


@pytest.mark.parametrize("preset", list(MODELS))
def test_train_cli_dispatches_only_requested_preset_without_allocating_weights(monkeypatch, preset):
    received = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(training, "run", lambda cli: received.append(cli))
    training.main(["--model", preset, "--steps", "1"])
    assert len(received) == 1
    assert received[0].model == preset


def test_cli_defaults_to_2b_and_inference_path(monkeypatch, capsys):
    received = []
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(training, "run", lambda cli: received.append(cli))
    training.main(["--steps", "1"])
    assert received[0].model == "2B"
    assert default_checkpoint_dir().name == "datax-2B"
    with pytest.raises(SystemExit) as exc:
        training.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--model {2B,7B,13B}" in text
    assert "default: 2B" in text
    assert "dense" not in text and "moe" not in text
    with pytest.raises(SystemExit) as exc:
        infer.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--model {2B,7B,13B}" in help_text
    assert "default: 2B" in help_text
    assert "checkpoints/datax-<model>" in help_text


@pytest.mark.parametrize("name", ["1B", "3B", "2B-dense", "7B-dense", "13B-moe"])
def test_train_cli_rejects_removed_names_before_model_creation(monkeypatch, name):
    monkeypatch.setattr(training, "run", lambda cli: pytest.fail("must reject before run"))
    with pytest.raises(SystemExit) as exc:
        training.main(["--model", name, "--steps", "1"])
    assert exc.value.code == 2


@pytest.mark.parametrize("name,dim,layers,heads,hidden,experts,expected", [
    ("1B", 1024, 32, 8, 2048, 4, 1_217_340_416),
    ("2B", 2048, 16, 16, 2048, 4, 1_595_611_136),
    ("7B", 2048, 32, 16, 4096, 8, 7_400_990_720),
])
def test_fully_specified_legacy_moe_json_preserves_architecture(
    name, dim, layers, heads, hidden, experts, expected,
):
    # Complete JSON written by the previous trainer, without the new dense fields.
    legacy = dict(
        name=name, dim=dim, n_layers=layers, n_heads=heads, n_kv_heads=4,
        head_dim=128, vocab_size=151936, moe_intermediate_size=hidden,
        norm_eps=1e-6, rope_theta=10000.0, max_seq_len=4096,
        dropout_prob=0.0, num_experts=experts, num_activated_experts=2,
        norm_experts_prob=True,
    )
    args = ModelArgs(**json.loads(json.dumps(legacy)))
    assert args.ffn_type == "moe" and args.name == name
    assert parameter_count(args) == expected
    with torch.device("meta"):
        model = Transformer(args)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert all(isinstance(layer.feed_forward, MOEFeedForward) for layer in model.layers)


def test_plain_7b_name_does_not_override_saved_architecture():
    old_dense = asdict(get_config("7B"))
    old_dense["name"] = "7B-dense"
    loaded = ModelArgs(**json.loads(json.dumps(old_dense)))
    assert loaded.ffn_type == "dense"
    assert parameter_count(loaded) == 7_419_998_208
    assert loaded.name == "7B-dense"  # Saved metadata is not silently rewritten.


@pytest.mark.parametrize("preset", list(MODELS))
def test_run_ddp_unused_parameter_setting_matches_loaded_architecture(monkeypatch, tmp_path, preset):
    """Intercept production initialization before data loading or training."""
    args = get_config(preset)
    class StopAfterDDP(Exception):
        pass
    class StubModel:
        def to(self, device):
            return self
    seen = []
    def ddp(model, **kwargs):
        seen.append(kwargs)
        raise StopAfterDDP
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(training.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(training.dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(training, "get_config", lambda *a, **k: args)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(training, "Transformer", lambda cfg: StubModel())
    monkeypatch.setattr(training, "DDP", ddp)
    cli = SimpleNamespace(model=preset, output_dir=tmp_path/"empty", seed=42,
                          max_seq_len=None, tokenizer="unused")
    with pytest.raises(StopAfterDDP):
        training.run(cli)
    assert seen[0]["find_unused_parameters"] is (preset == "13B")
    assert seen[0]["broadcast_buffers"] is False


@pytest.mark.parametrize("name", list(MODELS))
def test_default_checkpoint_directory_is_model_derived_and_repository_relative(name, tmp_path, monkeypatch):
    from pathlib import Path
    import model.config as config

    monkeypatch.chdir(tmp_path)
    path = default_checkpoint_dir(name)
    assert path == Path(config.__file__).resolve().parents[1] / "checkpoints" / f"datax-{get_config(name).name}"
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("name", ["../other", "1B", "7B-dense"])
def test_default_directory_rejects_unknown_presets(name):
    with pytest.raises(KeyError):
        default_checkpoint_dir(name)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("explicit", [False, True])
def test_training_uses_same_default_path_as_inference(
    name, explicit, tmp_path, monkeypatch,
):
    from pathlib import Path

    class StopBeforeLoading(Exception):
        pass

    path = Path("custom-output") if explicit else None
    expected = path if explicit else default_checkpoint_dir(name)
    observed = []

    def checkpoints(output):
        observed.append(output)
        raise StopBeforeLoading

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(training, "checkpoint_paths", checkpoints)
    if explicit:
        def forbid(*a, **kw):
            pytest.fail("explicit output path must bypass the default")
        monkeypatch.setattr(training, "default_checkpoint_dir", forbid)
    cli = SimpleNamespace(model=name, output_dir=path)
    with pytest.raises(StopBeforeLoading):
        training.run(cli)
    assert observed == [expected]
