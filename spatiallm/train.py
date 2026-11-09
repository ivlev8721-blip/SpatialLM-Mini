import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

from .runtime import environment, load_checkpoint, save_checkpoint, save_json, seed_everything, select_device
import torch
from .data import load_corpus, sample_batch
from .model import LanguageModel, ModelConfig


DEFAULTS = dict(steps=200, batch_size=8, seq_len=64, lr=0.0006, weight_decay=0.01,
                grad_accum=1, seed=42, val_fraction=0.1, eval_every=50,
                eval_batches=4, amp=False)


@torch.no_grad()
def evaluate(model, tokens, settings, device, autocast):
    model.eval()
    generator = torch.Generator().manual_seed(settings["seed"] + 10000)
    losses = []
    for _ in range(settings["eval_batches"]):
        x, y = sample_batch(tokens, settings["batch_size"], settings["seq_len"], generator, device)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(config, data, out, device_name="auto", resume=None, threads=4):
    if set(config) - {"model", "training"}:
        raise ValueError("unknown top-level config keys")
    cfg = ModelConfig(**config.get("model", {}))
    if cfg.vocab_size != 257:
        raise ValueError("train.py uses byte tokens; use the subword16k module for GPT-2 tokens")
    supplied = config.get("training", {})
    if set(supplied) - set(DEFAULTS):
        raise ValueError(f"unknown training keys: {set(supplied) - set(DEFAULTS)}")
    settings = DEFAULTS | supplied
    for key in ("steps", "batch_size", "seq_len", "grad_accum", "eval_every", "eval_batches", "seed"):
        if not isinstance(settings[key], int) or settings[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    if settings["seq_len"] > cfg.max_seq_len or settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("invalid sequence length or optimizer settings")
    torch.set_num_threads(threads)
    device = select_device(device_name)
    if settings["amp"] and device.type != "cuda":
        raise ValueError("amp=true requires CUDA; use the tiny CPU config")
    seed_everything(settings["seed"])
    train_data, val_data, digest = load_corpus(data, settings["seq_len"], settings["val_fraction"])
    out = Path(out)
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("output already contains last.pt; choose a new --out or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    model = LanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if settings["amp"] else nullcontext
    sampler = torch.Generator().manual_seed(settings["seed"] + 1)
    signature = {k: v for k, v in settings.items() if k not in ("steps", "eval_every", "eval_batches")}
    signature.update(data_sha256=digest, device_type=device.type, torch=str(torch.__version__))
    history, start = [], 0
    if resume:
        state = load_checkpoint(resume)
        from dataclasses import asdict
        if asdict(ModelConfig(**state["model_config"])) != model.config_dict() or state["signature"] != signature:
            raise ValueError("resume config, corpus, device or PyTorch version differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        sampler.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, start = state["history"], state["step"]
        if settings["steps"] <= start:
            raise ValueError("--steps must exceed checkpoint step")
    info = environment(device)
    save_json(out / "config.json", {"model": model.config_dict(), "training": settings})
    save_json(out / "environment.json", info | {"data_sha256": digest, "parameters": model.parameter_count()})
    print(json.dumps(info | {"parameters": model.parameter_count()}, ensure_ascii=False), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_initial_val = evaluate(model, val_data, settings, device, autocast)
    initial_val = state.get("initial_val_loss", run_initial_val) if resume else run_initial_val
    print(f"validation loss at start of this run: {run_initial_val:.4f}", flush=True)
    started = time.perf_counter()
    for step in range(start + 1, settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(settings["grad_accum"]):
            x, y = sample_batch(train_data, settings["batch_size"], settings["seq_len"], sampler, device)
            with autocast():
                _, loss = model(x, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite training loss")
            train_loss += loss.item() / settings["grad_accum"]
            scaler.scale(loss / settings["grad_accum"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if step % settings["eval_every"] == 0 or step == settings["steps"]:
            val_loss = evaluate(model, val_data, settings, device, autocast)
            row = dict(step=step, train_loss=train_loss, val_loss=val_loss,
                       val_byte_perplexity=math.exp(min(val_loss, 80)), grad_norm=float(grad_norm))
            history.append(row)
            print(json.dumps(row), flush=True)
            save_json(out / "metrics.json", {"initial_val_loss": initial_val,
                      "start_val_loss_this_run": run_initial_val, "history": history,
                      "elapsed_seconds_this_run": time.perf_counter() - started,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None})
            save_checkpoint(out / "last.pt", dict(format_version=1, model_config=model.config_dict(),
                            model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            step=step, signature=signature, history=history, initial_val_loss=initial_val,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            sampler_rng=sampler.get_state()))
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train a causal byte-level SpatialLM or Transformer")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--out", default="runs/spatial")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.steps is not None:
        config.setdefault("training", {})["steps"] = args.steps
    train(config, args.data, args.out, args.device, args.resume, args.threads)


if __name__ == "__main__":
    main()
