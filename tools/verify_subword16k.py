"""Check deployed snapshots and actual long-prompt generation after the timed run."""
import gc
import hashlib
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
if not (root / "spatiallm/subword16k.py").exists():
    raise RuntimeError("Run from the SpatialLM-Mini project root")
sys.path.insert(0, str(root))
from spatiallm.runtime import load_checkpoint, seed_everything, save_json
import torch
from spatiallm.model import LanguageModel, ModelConfig
from spatiallm.subword16k import efficient_attention, generate
from spatiallm.subword_data import GPT2Tokenizer, load_subwords
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

torch.set_num_threads(4)
seed_everything(42)
# Compare the installed memory-efficient CUDA kernel with ordinary causal attention.
base = [torch.randn(1, 4, 128, 32, device="cuda", dtype=torch.float16) for _ in range(3)]
outputs, gradients = [], []
for backend in (SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION):
    inputs = [item.detach().clone().requires_grad_() for item in base]
    with sdpa_kernel(backend):
        value = F.scaled_dot_product_attention(*inputs, is_causal=True)
        value.float().square().mean().backward()
    outputs.append(value.detach())
    gradients.append([item.grad.detach() for item in inputs])
torch.testing.assert_close(outputs[0], outputs[1], rtol=0.01, atol=0.002)
for expected, actual in zip(*gradients):
    torch.testing.assert_close(expected, actual, rtol=0.05, atol=1e-5)
backend_check = dict(passed=True, precision="FP16", causal=True,
                     max_output_abs_error=(outputs[0] - outputs[1]).abs().max().item(),
                     max_gradient_abs_error=max((a-b).abs().max().item() for a,b in zip(*gradients)))
del base, inputs, outputs, gradients, value
out = root / "runs/subword16k_checks"
out.mkdir(exist_ok=True)
splits, manifest = load_subwords(root / "data/subword16k")
result = json.loads((root / "runs/subword16k_v1/results.json").read_text(encoding="utf-8"))
assert len(result["records"]) == 18
checks = []
for row in result["records"]:
    path = root / "runs/subword16k_v1" / f"{row['architecture']}_seed{row['seed']}" / f"{row['comparison']}.pt"
    state = load_checkpoint(path)
    assert state["experiment"] == row
    assert state["tokenizer"] == "gpt2"
    model = LanguageModel(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"], strict=True)
    assert model.parameter_count() == row["parameters"]
    checks.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    del model, state

tokenizer = GPT2Tokenizer()
prompt = (root / "data/natural/test.txt").read_text(encoding="utf-8")
assert len(tokenizer.encode(prompt)) == 18225
long_checks = []
for arch in ("spatial", "transformer", "hybrid"):
    state = load_checkpoint(root / f"runs/subword16k_v1/{arch}_seed42/equal_time.pt")
    model = LanguageModel(ModelConfig(**state["model_config"])).cuda()
    model.load_state_dict(state["model"], strict=True)
    with efficient_attention():
        text = generate(model, prompt, count=2, temperature=0)
    assert text.startswith(prompt)
    assert len(tokenizer.encode(text)) > len(tokenizer.encode(prompt))
    (out / f"long_prompt_{arch}.txt").write_text(text, encoding="utf-8")
    long_checks.append(dict(architecture=arch, prompt_tokens=18225, active_context=16384,
                            appended_text=text[len(prompt):], passed=True))
    del state, model
    gc.collect()
    torch.cuda.empty_cache()

# Confirm an older byte checkpoint still loads without changing parameter names.
from spatiallm.generate import generate as generate_bytes
old_paths = sorted((root / "runs/natural_v1").rglob("*.pt"))
assert old_paths
old = load_checkpoint(old_paths[0])
model = LanguageModel(ModelConfig(**old["model_config"]))
model.load_state_dict(old["model"], strict=True)
sample = generate_bytes(model, "Elizabeth ", max_new_tokens=4, temperature=0)
assert sample.startswith("Elizabeth ")
save_json(out / "verification.json", dict(snapshots=checks, long_prompt_generation=long_checks,
          efficient_vs_math=backend_check,
          legacy_checkpoint=str(old_paths[0].relative_to(root)), legacy_sample=sample,
          dataset_manifest=manifest, passed=True))
print(json.dumps(dict(snapshots_verified=len(checks), long_prompt_generation=long_checks,
                      legacy_checkpoint_loaded=True, passed=True), ensure_ascii=False))
