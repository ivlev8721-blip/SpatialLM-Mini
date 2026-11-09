import argparse
from contextlib import nullcontext
import gc
import time

from .runtime import environment, save_json, seed_everything, select_device
import torch
from .model import LanguageModel, ModelConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(device_name="auto", lengths=(64, 256), batch_size=4, dim=96, layers=2,
              heads=4, local_window=8, warmup=5, iterations=20, amp=False, threads=4):
    if min(*lengths, batch_size, warmup, iterations) < 1:
        raise ValueError("lengths, batch size, warmup and iterations must be positive")
    torch.set_num_threads(threads)
    device = select_device(device_name)
    if amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    seed_everything(42)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else nullcontext
    rows = []
    for length in lengths:
        # Exactly the same input and targets for both architectures.
        rng = torch.Generator().manual_seed(42 + length)
        x = torch.randint(256, (batch_size, length), generator=rng).to(device)
        y = torch.randint(256, (batch_size, length), generator=rng).to(device)
        for architecture in ("spatial", "transformer"):
            seed_everything(42)
            cfg = ModelConfig(architecture=architecture, dim=dim, layers=layers, heads=heads,
                              max_seq_len=max(lengths), local_window=local_window)
            model = LanguageModel(cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)

            def train_step():
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()

            model.train()
            for _ in range(warmup):
                train_step()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(iterations):
                train_step()
            synchronize(device)
            training_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
            model.eval()
            with torch.inference_mode():
                for _ in range(warmup):
                    with autocast():
                        model(x)
                synchronize(device)
                started = time.perf_counter()
                for _ in range(iterations):
                    with autocast():
                        model(x)
                synchronize(device)
                forward_seconds = time.perf_counter() - started
            row = dict(architecture=architecture, seq_len=length, batch_size=batch_size,
                       parameters=model.parameter_count(),
                       training_tokens_per_second=batch_size * length * iterations / training_seconds,
                       prefill_tokens_per_second=batch_size * length * iterations / forward_seconds,
                       training_step_ms=1000 * training_seconds / iterations,
                       peak_training_allocated_mib=peak, peak_training_reserved_mib=reserved,
                       edges_per_token=len(model.blocks[0].mix.offsets) if architecture == "spatial" else length)
            rows.append(row)
            print(row, flush=True)
            del model, optimizer, scaler, train_step
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"environment": environment(device), "seed": 42, "amp": amp,
            "warmup": warmup, "iterations": iterations,
            "model": dict(dim=dim, layers=layers, heads=heads, local_window=local_window,
                          max_seq_len=max(lengths)),
            "notes": "Same width/depth, different parameter counts. Synthetic fixed batches; timing includes optimizer and clipping, excludes data loading. Prefill is full context, not autoregressive decode. No quality or speedup claim.",
            "results": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="runs/benchmark.json")
    args = parser.parse_args()
    result = benchmark(args.device, args.lengths, args.batch_size, args.dim, args.layers,
                       args.heads, args.local_window, args.warmup, args.iterations, args.amp, args.threads)
    save_json(args.out, result)


if __name__ == "__main__":
    main()
