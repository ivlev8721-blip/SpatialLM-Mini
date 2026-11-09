"""GPT-2 subword experiments with bounded head memory and efficient CUDA SDPA."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import statistics
import time

from .runtime import environment, load_checkpoint, save_checkpoint, save_json, seed_everything, select_device
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from .data import sample_batch
from .experiment import ARCHITECTURES, matched_configs
from .model import LanguageModel, ModelConfig
from .subword_data import GPT2Tokenizer, load_subwords, prepare_subwords


@contextmanager
def efficient_attention():
    # Prevent an unnoticed quadratic-memory fallback on the 8 GiB GPU.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        yield


@torch.no_grad()
def evaluate(model, tokens, chunk_size, device):
    previous = model.training
    model.eval()
    context = model.config.max_seq_len
    if len(tokens) < context + 1:
        raise ValueError("evaluation split must contain at least context+1 tokens")
    # Exact common prefix: one complete 16K-token block for every architecture.
    x, y = tokens[:context][None].to(device), tokens[1:context + 1][None].to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.forward_loss(x, y, chunk_size=chunk_size).item()
    model.train(previous)
    return dict(loss=loss, token_perplexity=math.exp(min(loss, 80)), targets=context)


@torch.no_grad()
def generate(model, prompt, count=64, seed=42, temperature=0.8):
    if model.config.vocab_size != 50257 or count < 0 or temperature < 0:
        raise ValueError("GPT-2 model and nonnegative count/temperature required")
    tokenizer = GPT2Tokenizer()
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(prompt) or [tokenizer.eot_id]], device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    previous = model.training
    model.eval()
    for _ in range(count):
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = model.forward_hidden(tokens[:, -model.config.max_seq_len:])
            logits = model.output(hidden[:, -1:])[:, 0].float()
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            threshold = logits.topk(40, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=rng)
        tokens = torch.cat((tokens, nxt), dim=1)
        if nxt.item() == tokenizer.eot_id:
            break
    model.train(previous)
    return tokenizer.decode(tokens[0].tolist())


def train_one(cfg, seed, splits, directory, steps, seconds, chunk_size, lr, pilot=False):
    seed_everything(seed)
    device = select_device("cuda")
    model = LanguageModel(cfg).to(device)
    model.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    sampler = torch.Generator().manual_seed(seed + 1)
    # Warm up one backward without applying an optimizer update.
    warm_rng = torch.Generator().manual_seed(999)
    x, y = sample_batch(splits["train"], 1, cfg.max_seq_len, warm_rng, device)
    with torch.autocast("cuda", dtype=torch.float16):
        warm_loss = model.forward_loss(x, y, chunk_size=chunk_size)
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
    del warm_loss
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    initial_validation = evaluate(model, splits["validation"], chunk_size, device)
    elapsed, step, records = 0.0, 0, {}
    goal_count = 1 if pilot else 2
    while len(records) < goal_count:
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x, y = sample_batch(splits["train"], 1, cfg.max_seq_len, sampler, device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.forward_loss(x, y, chunk_size=chunk_size)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - tick
        step += 1
        if step % 10 == 0:
            print(json.dumps(dict(progress=True, architecture=cfg.architecture, seed=seed, step=step,
                                  training_seconds=elapsed, train_loss=loss.item())), flush=True)
        events = []
        if step == steps:
            events.append("pilot" if pilot else "equal_steps")
        if not pilot and elapsed >= seconds and "equal_time" not in records:
            events.append("equal_time")
        if events:
            allocated = torch.cuda.max_memory_allocated(device) / 2**20
            reserved = torch.cuda.max_memory_reserved(device) / 2**20
            evaluation = {name: evaluate(model, splits[name], chunk_size, device) for name in ("validation", "test")}
            for event in events:
                row = dict(architecture=cfg.architecture, seed=seed, comparison=event,
                           model_config=asdict(cfg), parameters=model.parameter_count(), steps=step,
                           training_seconds=elapsed, training_tokens=step * cfg.max_seq_len,
                           training_tokens_per_second=step * cfg.max_seq_len / elapsed,
                           peak_allocated_mib=allocated, peak_reserved_mib=reserved,
                           initial_validation=initial_validation,
                           train_loss_last_batch=loss.item(), grad_norm=float(grad_norm), **evaluation)
                records[event] = row
                if not pilot:
                    save_checkpoint(directory / f"{event}.pt", dict(format_version=1, model_config=asdict(cfg),
                                    model=model.state_dict(), experiment=row, inference_only=True,
                                    tokenizer="gpt2", attention_backend="EFFICIENT_ATTENTION"))
                save_json(directory / "metrics.json", list(records.values()))
                print(json.dumps(row), flush=True)
    if not pilot:
        sample = generate(model, "Elizabeth ", count=48, seed=seed)
        (directory / "sample.txt").write_text(sample, encoding="utf-8")
    return list(records.values())


def summarize(records):
    summary, paired = [], []
    comparisons = sorted(set(row["comparison"] for row in records))
    for comparison in comparisons:
        for arch in ARCHITECTURES:
            rows = [r for r in records if r["comparison"] == comparison and r["architecture"] == arch]
            if not rows:
                continue
            entry = dict(comparison=comparison, architecture=arch, seeds=[r["seed"] for r in rows],
                         parameters=rows[0]["parameters"])
            for key, values in {
                "test_loss": [r["test"]["loss"] for r in rows],
                "validation_loss": [r["validation"]["loss"] for r in rows],
                "steps": [r["steps"] for r in rows],
                "training_seconds": [r["training_seconds"] for r in rows],
                "tokens_per_second": [r["training_tokens_per_second"] for r in rows],
                "peak_allocated_mib": [r["peak_allocated_mib"] for r in rows],
            }.items():
                entry[key + "_mean"] = statistics.mean(values)
                entry[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(entry)
        for arch in ("spatial", "hybrid"):
            deltas = []
            for row in records:
                if row["comparison"] != comparison or row["architecture"] != arch:
                    continue
                baseline = next((r for r in records if r["comparison"] == comparison and
                                 r["architecture"] == "transformer" and r["seed"] == row["seed"]), None)
                if baseline:
                    deltas.append(row["test"]["loss"] - baseline["test"]["loss"])
            if deltas:
                paired.append(dict(comparison=comparison, architecture=arch, deltas=deltas,
                                   mean_delta=statistics.mean(deltas), wins=sum(x < 0 for x in deltas)))
    return dict(summary=summary, paired_deltas=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "pilot", "run", "generate"])
    parser.add_argument("--data", default="data/subword16k")
    parser.add_argument("--text", default="data/natural")
    parser.add_argument("--out", default="runs/subword16k_v1")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", default="Elizabeth ")
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "prepare":
        print(json.dumps(prepare_subwords(args.text, args.data), indent=2))
        return
    if args.mode == "generate":
        state = load_checkpoint(args.checkpoint)
        if state.get("tokenizer") != "gpt2":
            raise ValueError("GPT-2 checkpoint required")
        seed_everything(args.seeds[0])
        model = LanguageModel(ModelConfig(**state["model_config"])).cuda()
        model.load_state_dict(state["model"])
        with efficient_attention():
            print(generate(model, args.prompt, args.count, args.seeds[0]))
        return
    if (min(args.steps, args.seq_len, args.chunk_size, args.lr, args.seconds) <= 0 or
            not math.isfinite(args.seconds) or not math.isfinite(args.lr) or len(set(args.seeds)) != len(args.seeds)):
        parser.error("positive finite budgets and distinct seeds required")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("choose a new output directory")
    splits, manifest = load_subwords(args.data)
    if any(len(v) < args.seq_len + 1 for v in splits.values()):
        raise ValueError("all token splits must contain at least context+1 tokens")
    configs = matched_configs(128, 4, 4, args.seq_len, 8, vocab_size=50257)
    protocol = dict(arguments=vars(args), environment=environment(select_device("cuda")), tokenizer=manifest,
                    precision="FP16 autocast, FP32 parameters/optimizer, GradScaler initial scale 128",
                    attention_backend="EFFICIENT_ATTENTION only; math fallback disabled",
                    activation_checkpointing=True, checkpointed_output_chunks=args.chunk_size,
                    deterministic_algorithms=True,
                    evaluation="first complete context-sized block of validation/test; same target positions for all models",
                    time_budget="synchronized training steps including sampler, transfer, forward/backward and optimizer; excluding warmup, evaluation, saving and generation",
                    configs={name: asdict(cfg) for name, cfg in configs.items()})
    save_json(out / "protocol.json", protocol)
    print(json.dumps(protocol), flush=True)
    records = []
    for index, seed in enumerate(args.seeds):
        order = ARCHITECTURES[index % 3:] + ARCHITECTURES[:index % 3]
        for arch in order:
            directory = out / f"{arch}_seed{seed}"
            directory.mkdir()
            with efficient_attention():
                records.extend(train_one(configs[arch], seed, splits, directory, args.steps, args.seconds,
                                         args.chunk_size, args.lr, pilot=args.mode == "pilot"))
            gc.collect()
            torch.cuda.empty_cache()
            save_json(out / "results.json", dict(protocol=protocol, records=records, **summarize(records)))


if __name__ == "__main__":
    main()
