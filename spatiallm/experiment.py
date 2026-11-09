"""Paired natural-language experiment: matched parameters, steps and wall time.

Checkpoint files are inference snapshots, intentionally not training-resume files.
Evaluation and serialization are excluded from the equal-training-time budget.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .runtime import environment, save_checkpoint, save_json, seed_everything, select_device
import torch
from .data import sample_batch
from .generate import generate
from .model import LanguageModel, ModelConfig


ARCHITECTURES = ("spatial", "transformer", "hybrid")


def matched_configs(dim, layers, heads, seq_len, local_window, vocab_size=257):
    common = dict(dim=dim, layers=layers, heads=heads, max_seq_len=seq_len,
                  local_window=local_window, dropout=0.0, vocab_size=vocab_size)
    baseline = ModelConfig(architecture="transformer", **common)
    target = LanguageModel(baseline).parameter_count()
    configs = {}
    for architecture in ARCHITECTURES:
        cfg = ModelConfig(architecture=architecture, **common)
        difference = target - LanguageModel(cfg).parameter_count()
        # Increasing FFN width by one adds (2*dim+1) parameters per layer.
        cfg.ff_dim = 4 * dim + round(difference / (layers * (2 * dim + 1)))
        configs[architecture] = cfg
    return configs


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def restrict_evaluation(partitions, seq_len, targets=None):
    """Use the same target positions when comparing different context lengths."""
    if targets is None:
        return partitions
    if targets < 1 or targets % seq_len:
        raise ValueError("eval_targets must be positive and divisible by seq_len")
    selected = dict(partitions)
    for name in ("validation", "test"):
        if len(partitions[name]) < targets + 1:
            raise ValueError(f"{name} has fewer than eval_targets+1 bytes")
        selected[name] = partitions[name][:targets + 1]
    return selected


@torch.no_grad()
def evaluate_partition(model, tokens, seq_len, batch_size, device):
    """All complete disjoint blocks, common target positions in every run."""
    previous = model.training
    model.eval()
    total_loss, total_targets = 0.0, 0
    starts = torch.arange(0, len(tokens) - seq_len, seq_len)
    if not len(starts):
        raise ValueError("partition must contain at least seq_len+1 bytes")
    for first in range(0, len(starts), batch_size):
        indices = starts[first:first + batch_size, None] + torch.arange(seq_len + 1)[None, :]
        batch = tokens[indices].to(device)
        loss = model(batch[:, :-1], batch[:, 1:])[1]
        count = batch.shape[0] * seq_len
        total_loss += loss.item() * count
        total_targets += count
    model.train(previous)
    return {"loss": total_loss / total_targets,
            "bits_per_byte": total_loss / total_targets / math.log(2),
            "byte_perplexity": math.exp(total_loss / total_targets),
            "targets": total_targets, "unused_tail_bytes": len(tokens) - 1 - total_targets}


def run_one(cfg, seed, partitions, out, device, steps, seconds, batch_size, lr, weight_decay):
    seed_everything(seed)
    model = LanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sampler = torch.Generator().manual_seed(seed + 1)
    # Warm up kernels without changing the model or optimizer. Dropout is zero.
    warm_rng = torch.Generator().manual_seed(999)
    x, y = sample_batch(partitions["train"], batch_size, cfg.max_seq_len, warm_rng, device)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        model(x, y)[1].backward()
    model.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records, elapsed, step = {}, 0.0, 0
    model.train()
    while len(records) < 2:
        tick = time.perf_counter()
        x, y = sample_batch(partitions["train"], batch_size, cfg.max_seq_len, sampler, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        if not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        elapsed += time.perf_counter() - tick
        step += 1
        events = []
        if step == steps and "equal_steps" not in records:
            events.append("equal_steps")
        if elapsed >= seconds and "equal_time" not in records:
            events.append("equal_time")
        # If both thresholds are crossed together, evaluate and serialize once.
        if events:
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
            metrics = {name: evaluate_partition(model, partitions[name], cfg.max_seq_len, batch_size, device)
                       for name in ("validation", "test")}
            for event in events:
                checkpoint = out / f"{event}.pt"
                row = dict(architecture=cfg.architecture, seed=seed, comparison=event,
                           model_config=asdict(cfg), parameters=model.parameter_count(),
                           steps=step, training_seconds=elapsed, training_tokens=step * batch_size * cfg.max_seq_len,
                           training_tokens_per_second=step * batch_size * cfg.max_seq_len / elapsed,
                           train_loss_last_batch=loss.item(), peak_allocated_mib=peak,
                           peak_reserved_mib=reserved, **metrics)
                save_checkpoint(checkpoint, dict(format_version=1, model_config=asdict(cfg),
                                                model=model.state_dict(), experiment=row,
                                                inference_only=True))
                records[event] = row
                save_json(out / "metrics.json", list(records.values()))
                print(json.dumps({k: row[k] for k in ("architecture", "seed", "comparison", "steps", "training_seconds", "validation", "test")}), flush=True)
    # Save examples after metrics; examples never affect model selection.
    text = generate(model, "Elizabeth ", max_new_tokens=200, temperature=0.8, seed=seed)
    (out / "sample.txt").write_text(text, encoding="utf-8")
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return list(records.values())


def summarize(records):
    summary = []
    for comparison in ("equal_steps", "equal_time"):
        for architecture in ARCHITECTURES:
            subset = [r for r in records if r["comparison"] == comparison and r["architecture"] == architecture]
            if not subset:
                continue
            row = dict(comparison=comparison, architecture=architecture, seeds=[r["seed"] for r in subset],
                       parameters=subset[0]["parameters"])
            for metric, values in {
                "test_loss": [r["test"]["loss"] for r in subset],
                "validation_loss": [r["validation"]["loss"] for r in subset],
                "steps": [r["steps"] for r in subset],
                "training_seconds": [r["training_seconds"] for r in subset],
                "tokens_per_second": [r["training_tokens_per_second"] for r in subset],
                "peak_allocated_mib": [r["peak_allocated_mib"] for r in subset],
            }.items():
                if values[0] is not None:
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(row)
    paired = []
    for comparison in ("equal_steps", "equal_time"):
        for architecture in ("spatial", "hybrid"):
            deltas = []
            for row in records:
                if row["comparison"] != comparison or row["architecture"] != architecture:
                    continue
                baseline = next((r for r in records if r["comparison"] == comparison and
                                 r["architecture"] == "transformer" and r["seed"] == row["seed"]), None)
                if baseline:
                    deltas.append(row["test"]["loss"] - baseline["test"]["loss"])
            if deltas:
                paired.append(dict(comparison=comparison, architecture=architecture,
                                   test_loss_delta_vs_transformer=deltas,
                                   mean_delta=statistics.mean(deltas),
                                   sd_delta=statistics.stdev(deltas) if len(deltas) > 1 else None,
                                   note="Negative favours candidate; three seeds are exploratory, not a significance test."))
    return {"summary": summary, "paired_deltas": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/natural")
    parser.add_argument("--out", default="runs/natural_v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-targets", type=int, help="shared evaluation prefix target count, divisible by seq-len")
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (not math.isfinite(args.seconds) or not math.isfinite(args.lr) or
            min(args.steps, args.seconds, args.batch_size, args.lr) <= 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("budgets and batch size must be positive, seeds must be unique")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Choose a fresh --out; existing experiments are never overwritten")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    configs = matched_configs(args.dim, args.layers, args.heads, args.seq_len, args.local_window)
    partitions, hashes, eval_hashes = {}, {}, {}
    for name in ("train", "validation", "test"):
        raw = (Path(args.data) / f"{name}.txt").read_bytes()
        raw.decode("utf-8")
        if len(raw) < args.seq_len + 1:
            raise ValueError(f"{name} partition too short")
        partitions[name] = torch.tensor(list(raw), dtype=torch.long)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if name != "train":
            used = raw if args.eval_targets is None else raw[:args.eval_targets + 1]
            eval_hashes[name] = hashlib.sha256(used).hexdigest()
    partitions = restrict_evaluation(partitions, args.seq_len, args.eval_targets)
    out.mkdir(parents=True)
    protocol = dict(arguments=vars(args), environment=environment(device), data_sha256=hashes,
                    evaluation_prefix_sha256=eval_hashes,
                    configs={k: asdict(v) for k, v in configs.items()},
                    precision="FP32, deterministic algorithms, TF32 disabled",
                    evaluation="all complete nonoverlapping context blocks; same positions for all runs",
                    time_budget="sum of synchronized training step wall times, including sampling and optimizer; excludes warmup, evaluation, serialization, generation",
                    architecture_order="rotated by seed to reduce fixed ordering effects",
                    caveats="Parameter matching changes FFN width; same width/depth/heads/context. One book, three seeds, short budget. Test scores are reported only; no tuning on test.")
    save_json(out / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)
    records = []
    for index, seed in enumerate(args.seeds):
        order = ARCHITECTURES[index % 3:] + ARCHITECTURES[:index % 3]
        for architecture in order:
            directory = out / f"{architecture}_seed{seed}"
            directory.mkdir()
            records.extend(run_one(configs[architecture], seed, partitions, directory, device,
                                   args.steps, args.seconds, args.batch_size, args.lr, args.weight_decay))
            save_json(out / "results.json", {"protocol": protocol, "records": records, **summarize(records)})
    print(json.dumps(summarize(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
