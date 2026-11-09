# Recorded experiments

## Reference result: 16,384 GPT-2 tokens

The recorded dataset is Jane Austen's *Pride and Prejudice*. The exact source bytes, contiguous partitions, and GPT-2 token IDs are included. Prepared partitions contain 141,979 / 18,537 / 18,225 train / validation / test tokens. The tokenizer is fixed, `tiktoken==0.12.0`, vocabulary 50,257. No pretrained model weights are used.

The experiment is intentionally small: one book, three seeds, short training budgets. It measures learning and throughput in this implementation. It does not test instruction following, long-range fact retrieval, or general-purpose language-model quality.

### Shared conditions

| Setting | Value |
|---|---|
| Device | NVIDIA GeForce RTX 3050, 8,192 MiB VRAM |
| Recorded environment | Windows, Python 3.13.14, PyTorch 2.10.0+cu128, NumPy 2.2.6 |
| Hidden size / layers / heads | 128 / 4 / 4 |
| Context / microbatch | 16,384 tokens / 1 sequence |
| Gradient accumulation | None |
| Seeds | 42, 43, 44 |
| Optimizer | AdamW, lr 0.0006, weight decay 0.01, gradient clipping 1.0 |
| Dropout / TF32 | 0 / disabled |
| Deterministic algorithms | Enabled |
| Precision | FP16 autocast, FP32 parameters/optimizer, GradScaler initial scale 128 |
| Memory measures | Activation checkpointing, output head checkpointed in chunks of 512 positions |
| Full-attention kernel | PyTorch `EFFICIENT_ATTENTION`, math fallback disabled |
| Spatial topology | Local window 8 plus dyadic causal edges |
| Hybrid | Alternating spatial and full-attention blocks |
| FFN widths | Spatial 628, Transformer 512, Hybrid 570 |
| Parameters | Spatial 9,323,248; Transformer 9,323,392; Hybrid 9,323,320 |

FFN widths are deliberately different to match parameter counts approximately. This leaves an architectural difference besides the attention mechanism. All models use learned absolute position embeddings and tied token/output embeddings.

For each seed the training window sequence is identical across architectures. Architecture order rotates between seeds. One backward-only warmup is excluded from timing and performs no optimizer update. Initial validation is recorded before timed updates. The pilot checks feasibility; main-run settings were then fixed. Test results do not control stopping or select a best checkpoint.

### Two predetermined comparison points

1. **Equal updates:** exactly 100 optimizer updates, or 1,638,400 target tokens.
2. **Equal time:** the first completed step after 180 seconds of accumulated synchronized training-step time. The budget can be exceeded by less than one step.

Step time includes sampling, transfers, forward/backward, checkpoint recomputation, clipping, and optimizer updates. Warmup, evaluation, checkpoint saving, and generation are excluded. This is not end-to-end elapsed runtime. Faster models process more training tokens at equal time, which is part of that comparison.

Validation/test each use one complete context block: input positions 0…16383 predict positions 1…16384. All models and seeds use exactly the same target positions. The remaining held-out targets are excluded consistently. A single long block per split is a limitation, not a broad independent-document evaluation.

### Results

| Model | Equal-time test loss, mean ± SD | Equal-time tokens/s | Peak allocated MiB | Equal-update test loss | Time for 100 updates, seconds |
|---|---:|---:|---:|---:|---:|
| Transformer | 5.2176 ± 0.0087 | 13,126 | 499.1 | 5.6905 ± 0.0155 | 125.10 |
| Spatial | 4.8771 ± 0.0148 | 32,364 | 671.1 | 5.6527 ± 0.0111 | 50.64 |
| Hybrid | 4.8689 ± 0.0125 | 18,725 | 657.2 | 5.6394 ± 0.0139 | 87.55 |

Test loss is mean next-token negative log-likelihood using natural logarithms; lower is better. Token perplexity is `exp(loss)`. Standard deviation across three seeds is not a confidence interval or significance test.

Relative to this Transformer implementation, Spatial trains 2.47× faster and Hybrid 1.43× faster. Both have lower equal-time test loss in all three paired runs. The speed advantage does not imply smaller models, less allocated memory, faster decoding, or better quality than modern optimized sparse/local-attention baselines.

Memory is the peak allocated by PyTorch during training, including optimizer state. It excludes some driver/runtime/desktop allocations. The fused full-attention backend uses less allocated memory here than the explicit geometric gather implementation.

### Recorded artifacts

- [All per-run metrics and aggregates](../results/subword16k_v1/results.json)
- [Complete protocol and environment](../results/subword16k_v1/protocol.json)
- [Final snapshot/backend verification](../results/subword16k_checks/verification.json)
- [Original 25-test log](../results/subword16k_checks/tests_final.txt)
- [Detailed Russian report](../SUBWORD16K_RESULTS.md)
- [Pinned package list](../requirements-lock-subword-cuda.txt)

Recorded measurements were copied from local runs without changing values. Model checkpoint files are not distributed in Git. Reproduction creates 18 snapshots under your chosen `runs/` directory, roughly 0.67 GB in total. They are evaluation/generation snapshots, without optimizer/RNG state for resume. `tools/verify_subword16k.py` is a local audit script for the original run layout and older byte checkpoints; it is not required for a fresh clone's unit tests.

The GPU audit compared efficient and math attention on a small causal FP16 input. Maximum absolute output difference was 0.000977 and gradient difference 6.56e-7, within the recorded tolerances. This backend comparison complements the model loss/gradient tests; it is not an exhaustive numerical proof for all inputs.

### Reproduce

Install the tested CUDA-enabled PyTorch version and the project's `subword` extra, then run from the repository root:

```bash
python -m spatiallm.subword16k pilot --steps 3 --seeds 42 --out runs/pilot_repeat
python -m spatiallm.subword16k run --steps 100 --seconds 180 --seeds 42 43 44 --out runs/subword16k_repeat
```

Use the virtual environment's interpreter as shown in the READMEs. Do not run simultaneous GPU jobs. The source refuses an existing output directory. For the original run name, reporting can be rebuilt with `python tools/summarize_subword16k.py`; for another path use its `--results` and `--out` options.

### What this result cannot isolate

The graph is sparse and its weights use learned 3D distances. Without an otherwise identical graph with dot-product or non-geometric scores, we cannot determine how much geometry contributes. No direct comparison with Longformer, BigBird, Routing Transformer, LongNet, or Gemma has been run. The prototype's particular combination has not been shown to be novel.

The most informative next checks are geometry ablations, controlled-distance retrieval, additional corpora, stronger positional encodings/baselines, and longer training. Generated samples have repetitions and coherence errors.

## Earlier byte-token experiments

Earlier experiments used a vocabulary of 257 byte/BOS tokens and contexts of 128, 256, or 512 bytes. Their reports remain available in [NATURAL_RESULTS.md](../NATURAL_RESULTS.md), [CONTEXT_RESULTS.md](../CONTEXT_RESULTS.md), and [VALIDATION.md](../VALIDATION.md). Corresponding JSON measurements are preserved under `results/natural_v1/` and `results/context*/`.

The 16K series changes tokenizer, vocabulary, model size, precision, and memory strategy as well as context length. Comparing loss or perplexity across the series does not isolate context length and is not meaningful without a shared tokenization/evaluation unit. Historical reports describe the original local `runs/` layout; published raw metrics are under `results/`.
