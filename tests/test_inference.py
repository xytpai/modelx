from dataclasses import asdict
import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import infer
from model import ModelArgs, Transformer


@pytest.fixture
def tokenizer():
    vocab = {word: i for i, word in enumerate(
        ["<unk>", "<eos>", "User", "Assistant", ":", "Q", "A"]
    )}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>",
    )


class ScriptedModel(torch.nn.Module):
    def __init__(self, outputs, max_seq_len=32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.max_seq_len = max_seq_len
        self.outputs = iter(outputs)
        self.inputs = []

    def forward(self, tokens):
        assert not torch.is_grad_enabled()
        self.inputs.append(tokens.clone())
        logits = torch.full((*tokens.shape, 8), -100.0, device=tokens.device)
        logits[:, -1, next(self.outputs)] = 100.0
        logits[:, -1, 7] = 1000.0  # Padded model vocabulary: must not be generated.
        return logits


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_generation_matches_training_segments_and_stops_at_eos(tokenizer, monkeypatch, temperature):
    segments = []
    encode = tokenizer.encode

    def record(text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        segments.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", record)
    model = ScriptedModel([6, 1, 6])
    assert infer.generate(model, tokenizer, "Q", temperature=temperature) == "A"
    assert segments == ["User: ", "Q", "\nAssistant: "]
    assert model.inputs[0].tolist() == [[2, 4, 5, 3, 4]]
    assert model.inputs[1].tolist() == [[2, 4, 5, 3, 4, 6]]
    assert len(model.inputs) == 2


def test_generation_can_return_immediate_eos(tokenizer):
    model = ScriptedModel([1])
    assert infer.generate(model, tokenizer, "Q") == ""
    assert len(model.inputs) == 1


@pytest.mark.parametrize("limit,context,expected_steps", [(2, 32, 2), (10, 7, 2)])
def test_generation_respects_token_and_context_limits(tokenizer, limit, context, expected_steps):
    model = ScriptedModel([6] * 10, max_seq_len=context)
    assert infer.generate(model, tokenizer, "Q", max_new_tokens=limit) == "A A"
    assert len(model.inputs) == expected_steps
    assert all(tokens.size(1) <= context for tokens in model.inputs)


def test_overlong_prompt_fails_without_truncation(tokenizer):
    model = ScriptedModel([6], max_seq_len=5)
    with pytest.raises(ValueError, match="prompt has 5 tokens"):
        infer.generate(model, tokenizer, "Q")
    assert not model.inputs


def test_nonfinite_logits_fail(tokenizer, monkeypatch):
    model = ScriptedModel([6])
    monkeypatch.setattr(model, "forward", lambda tokens: torch.full((*tokens.shape, 8), float("nan")))
    with pytest.raises(FloatingPointError, match="non-finite"):
        infer.generate(model, tokenizer, "Q")


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_roundtrip_and_cli(tmp_path, tokenizer, capsys, storage_dtype):
    args = ModelArgs(
        name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8,
        vocab_size=8, moe_intermediate_size=24, num_experts=2,
        num_activated_experts=2, max_seq_len=32,
    )
    original = Transformer(args).eval()
    state = {name: value.to(storage_dtype) for name, value in original.state_dict().items()}
    from train import save_checkpoint
    save_checkpoint(tmp_path, state, {"step": 1}, [(1, 4.0)])
    state = {name: value.bfloat16() for name, value in state.items()}
    original.load_state_dict(state)  # FP32 reference with the same storage rounding.
    (tmp_path / "config.json").write_text(json.dumps(asdict(args)))
    tokenizer.save_pretrained(tmp_path / "tokenizer")

    loaded, restored_tokenizer = infer.load_model(tmp_path, torch.device("cpu"))
    assert not loaded.training
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    assert loaded.cos.dtype == loaded.sin.dtype == torch.float32
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    prompt = torch.tensor([[2, 4, 5, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(prompt), original(prompt), rtol=0, atol=0)

    expected = infer.generate(loaded, restored_tokenizer, "Q", max_new_tokens=2)
    infer.main([
        "--model", "13B", "--checkpoint", str(tmp_path), "--prompt", "Q",
        "--max-new-tokens", "2", "--device", "cpu",
    ])
    output = capsys.readouterr()
    assert output.out == (expected or "(empty reply)") + "\n"
    assert f"Loading {tmp_path} on cpu..." in output.err


@pytest.mark.parametrize("options", [
    ["--prompt", "  "],
    ["--max-new-tokens", "0"],
    ["--max-new-tokens", "-1"],
    ["--temperature", "-1"],
    ["--temperature", "nan"],
    ["--temperature", "inf"],
])
def test_invalid_cli_fails_before_loading(options, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("invalid CLI must not load model")
    monkeypatch.setattr(infer, "load_model", forbid)
    with pytest.raises(SystemExit) as exc:
        infer.main(["--prompt", "Q", *options])
    assert exc.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_gpu_generation_smoke():
    args = ModelArgs(
        name="tiny", dim=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8,
        vocab_size=8, moe_intermediate_size=24, num_experts=2,
        num_activated_experts=2, max_seq_len=32,
    )
    model = Transformer(args).eval()
    if torch.cuda.is_bf16_supported():
        model.bf16()
    model.cuda()
    with torch.inference_mode():
        logits = model(torch.tensor([[2, 4, 5, 3, 4]], device="cuda"))
    assert torch.isfinite(logits).all()
    assert model.cos.dtype == model.sin.dtype == torch.float32


@pytest.mark.parametrize("name", ["2B", "7B", "13B"])
def test_model_option_selects_default_checkpoint_without_cwd_dependency(
    name, tmp_path, monkeypatch, capsys,
):
    from model import default_checkpoint_dir

    expected = default_checkpoint_dir(name)
    monkeypatch.chdir(tmp_path)
    seen = []

    def load(path, device):
        seen.append((path, device))
        return object(), object()

    monkeypatch.setattr(infer, "load_model", load)
    monkeypatch.setattr(infer, "generate", lambda *a, **kw: "ok")
    infer.main(["--model", name, "--prompt", "Q", "--device", "cpu"])
    assert seen == [(expected, torch.device("cpu"))]
    assert expected.name == f"datax-{name}"
    assert capsys.readouterr().out == "ok\n"
    assert not (tmp_path/"checkpoints").exists()


def test_no_model_option_uses_shared_default(monkeypatch):
    from model import DEFAULT_MODEL, default_checkpoint_dir

    seen = []
    monkeypatch.setattr(infer, "load_model",
                        lambda path, device: (seen.append(path), object()))
    monkeypatch.setattr(infer, "generate", lambda *a, **kw: "ok")
    infer.main(["--prompt", "Q", "--device", "cpu"])
    assert DEFAULT_MODEL == "2B"
    assert seen == [default_checkpoint_dir()]
    assert seen[0].name == "datax-2B"


@pytest.mark.parametrize("name", ["2B", "7B", "13B"])
@pytest.mark.parametrize("path_kind", ["relative", "absolute", "home"])
def test_explicit_checkpoint_overrides_model_default(name, path_kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    checkpoint = {"relative": "custom/model",
                  "absolute": str(tmp_path/"custom/model"),
                  "home": "~/custom/model"}[path_kind]
    expected = tmp_path/"custom/model"
    seen = []

    def forbid(*a, **kw):
        pytest.fail("an explicit checkpoint must not consult the default directory")

    monkeypatch.setattr(infer, "default_checkpoint_dir", forbid)
    monkeypatch.setattr(infer, "load_model",
                        lambda path, device: (seen.append(path), object()))
    monkeypatch.setattr(infer, "generate", lambda *a, **kw: "ok")
    infer.main(["--model", name, "--checkpoint", checkpoint, "--prompt", "Q",
                "--device", "cpu"])
    assert seen == [expected]


@pytest.mark.parametrize("name", ["1B", "7B-dense", "13B-moe", "../arbitrary"])
def test_invalid_model_option_fails_before_loading(name, monkeypatch):
    def forbid(*a, **kw):
        pytest.fail("invalid model names must not load a checkpoint")
    monkeypatch.setattr(infer, "load_model", forbid)
    with pytest.raises(SystemExit) as exc:
        infer.main(["--model", name, "--prompt", "Q"])
    assert exc.value.code == 2
