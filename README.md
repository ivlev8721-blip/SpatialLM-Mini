# SpatialLM-Mini

**Small language models with sparse geometric attention — built to experiment on an 8 GB GPU.**

[Русский](README.ru.md) · [Results and protocol](docs/EXPERIMENTS.md) · [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

**Author: [Vladimir Ivlev](https://github.com/ivlev8721-blip) (nickname: Ivlev8721).**

SpatialLM-Mini is a working Python/PyTorch research prototype for asking a practical question: **can a small model learn faster on long sequences when each token exchanges information through a sparse graph, weighted by learned 3D distances?**

It includes training from scratch, text generation, a full-attention Transformer baseline, a hybrid of both architectures, reproducible experiment scripts, and automated tests. The code is small enough to inspect and modify on a personal computer.

> **Research snapshot, not a general-purpose assistant.** The current result is a comparison of three approximately 9.32M-parameter models on one novel. It does not establish a new state of the art, superiority over existing sparse-attention methods, or a separate benefit from the 3D coordinates.

## The idea in plain language

Full attention lets every token read all previous tokens. That is flexible, but expensive as the sequence grows.

The spatial variant gives each token a smaller, fixed set of connections: recent tokens and earlier positions at distances 1, 2, 4, 8, and so on. Learned query/key coordinates in three dimensions determine how strongly those connected tokens interact. The **connections are fixed; their weights depend on the text**.

The hybrid alternates spatial layers with full-attention layers, retaining occasional access to every preceding position.

```mermaid
flowchart LR
    T[Token IDs] --> E[Token and position embeddings]
    E --> S[Sparse geometric layer]
    S --> G[Full causal attention layer]
    G --> R[Repeat spatial and full layers]
    R --> O[Next-token probabilities]
```

The diagram shows the hybrid. The spatial model uses only spatial layers; the baseline uses only full-attention layers. Each block also contains normalization, residual connections, and a feed-forward network.

## What has been measured?

An initial experiment used an **NVIDIA RTX 3050 with 8 GB VRAM**, Windows, PyTorch 2.10.0+cu128, a context of **16,384 GPT-2 subword tokens**, and three random seeds: 42, 43, and 44. All models were trained from scratch. GPT-2 supplies the tokenizer only; no pretrained model weights are used.

The table averages the three runs at a budget of **180 seconds of accumulated training-step time per model**. Lower test loss is better.

| Model | Parameters | Training tokens/s | Speed vs. baseline | Test loss, mean ± SD | Peak allocated memory |
|---|---:|---:|---:|---:|---:|
| Transformer | 9,323,392 | 13,126 | 1.00× | 5.2176 ± 0.0087 | 499.1 MiB |
| Spatial | 9,323,248 | **32,364** | **2.47×** | 4.8771 ± 0.0148 | 671.1 MiB |
| Hybrid | 9,323,320 | 18,725 | 1.43× | **4.8689 ± 0.0125** | 657.2 MiB |

**The measured advantage is training throughput on this setup.** Model size is effectively the same, spatial attention used more allocated memory than the optimized baseline, and autoregressive generation speed has not been benchmarked.

At the same **100 optimizer updates**, test losses were 5.6905 for Transformer, 5.6527 for Spatial, and 5.6394 for Hybrid. The much larger improvement at equal time partly reflects the additional updates completed by the faster architectures.

The corpus is Jane Austen's *Pride and Prejudice*, split into contiguous train/validation/test sections. Each evaluation uses the same 16,384 target positions. Timings include sampling, device transfer, forward/backward, recomputation, clipping, and optimizer updates; warmup, evaluation, saving, and generation are excluded. Memory is PyTorch allocated memory, not total device usage. SD across three seeds is not a confidence interval.

[English experiment notes](docs/EXPERIMENTS.md) explain the conditions and limitations. [Raw results](results/subword16k_v1/results.json), [protocol](results/subword16k_v1/protocol.json), and the [detailed Russian report](SUBWORD16K_RESULTS.md) are included.

## Try it locally

Clone this repository or download its ZIP, then open a terminal in the project root. Python 3.10–3.13 is supported by the package metadata; the recorded GPU experiment and local tests used Python 3.13.14. The commands below pin the tested PyTorch release. GPU experiments need a compatible NVIDIA driver; a separate CUDA toolkit is not required by these wheels.

### Windows / PowerShell: CPU demo

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6 -e ".[subword]"
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m spatiallm.train --config configs/tiny.json --device cpu --out runs/demo
.\.venv\Scripts\python.exe -m spatiallm.generate --checkpoint runs/demo/last.pt --prompt 'The blue ' --max-new-tokens 120
```

### Linux / bash: CPU demo

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy==2.2.6 -e '.[subword]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m spatiallm.train --config configs/tiny.json --device cpu --out runs/demo
.venv/bin/python -m spatiallm.generate --checkpoint runs/demo/last.pt --prompt 'The blue ' --max-new-tokens 120
```

The tiny demo uses **byte tokens and a short context**, and learns repeated synthetic examples. It checks the training pipeline; it does not reproduce the 16K benchmark. Its output can be repetitive or malformed. CUDA-specific tests are skipped on CPU. Linux commands are provided for portability; the recorded hardware measurements were made on Windows.

### NVIDIA GPU and the 16K experiment

For a new GPU environment, replace the CPU PyTorch installation with the tested CUDA wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6 -e ".[subword]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# First check memory and backend support: three training steps per architecture.
.\.venv\Scripts\python.exe -m spatiallm.subword16k pilot --steps 3 --seeds 42 --out runs/pilot_16k
# Nine sequential runs: three architectures x three seeds.
.\.venv\Scripts\python.exe -m spatiallm.subword16k run --steps 100 --seconds 180 --seeds 42 43 44 --out runs/subword16k_repeat
# Generate from a checkpoint produced by your run.
.\.venv\Scripts\python.exe -m spatiallm.subword16k generate --checkpoint runs/subword16k_repeat/hybrid_seed42/equal_time.pt --prompt 'Elizabeth ' --count 64
```

On Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`. Run GPU experiments sequentially. The recorded nine-run series needs about 27 minutes of timed training plus setup, evaluation, and saving on the reference machine; other systems and background workloads change this.

The 16K path uses FP16 autocast with FP32 parameters/AdamW, activation checkpointing, and a checkpointed output head split into 512-position chunks. **All target positions and the entire 50,257-token vocabulary still participate in cross-entropy.** Transformer layers force PyTorch `EFFICIENT_ATTENTION`, preventing an accidental quadratic-memory math fallback. FlashAttention was not compiled into the reference Windows build. If the efficient backend is unavailable, the pilot fails; the full 16K experiment requires a supported CUDA setup. Consult the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) and [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Always choose a new output directory. Checkpoints are generated locally and are excluded from Git. The 16K snapshots contain inference weights and metadata, without optimizer/RNG state for resuming training. The byte trainer supports full checkpoint resume.

The exact recorded CUDA package versions are in [requirements-lock-subword-cuda.txt](requirements-lock-subword-cuda.txt); the [Russian 16K guide](SUBWORD16K_GUIDE.md) contains additional commands.

## Data and reproducibility

- The original Gutenberg text, fixed UTF-8 partitions, token manifest, and prepared GPT-2 token IDs are included. Text and token sequence hashes identify the corpus used for the measurements.
- GPT-2 tokenizer assets are downloaded by `tiktoken` on first use; an internet connection is needed for that first initialization. The local cache is excluded from Git. Model weights are never downloaded.
- To rebuild token IDs into a fresh directory: `python -m spatiallm.subword16k prepare --text data/natural --data data/subword16k_rebuilt`.
- For a byte-level experiment on your own UTF-8 text: `python -m spatiallm.train --config configs/tiny.json --data data/my_text.txt --device cpu --out runs/my_corpus`.
- The automated suite has 25 tests covering causal masking, topology, coordinate gradients, loss/gradient equivalence under chunking and checkpointing, tokenization, generation, resume, and CUDA smoke checks. GPU-only tests skip where CUDA is unavailable. A CPU CI workflow is included; it does not re-run the GPU benchmark.
- In the recorded environment, all 25 tests passed. All 18 experiment snapshots were reloaded, and all three architectures generated with an active 16,384-token window. [Verification record](results/subword16k_checks/verification.json).
- Seeds and deterministic settings improve reproducibility within an environment. Different hardware, drivers, kernels, or package versions can change results; equal-time runs can complete different numbers of updates.

[Data provenance and third-party terms](data/README.md) are separate from the project's code license.

## How the spatial attention works

For each layer and head, the hidden state is projected into a query coordinate and a key coordinate, each with three components. On the allowed causal edges:

```text
q, k = tanh(learned coordinate projections)
score(t, j, h) = -exp(clamp(log_scale[h])) * squared_distance(q[t,h], k[j,h])
                 + learned_edge_bias[h, offset(t,j)]
weights = softmax(scores over allowed neighbours)
output[t] = projection(sum_j weights[t,j] * values[j])
```

The graph contains self, local, and dyadic edges; duplicates are removed and future positions are masked. No global nearest-neighbour search is performed. With `K = O(local_window + log(max_seq_len))`, geometric mixing costs `O(B*T*K*D)` rather than forming a full `T×T` score matrix. The feed-forward layers and projections still perform ordinary matrix multiplications. Gathering neighbours materializes intermediate tensors, which explains why fewer arithmetic operations do not automatically mean less memory than fused full attention.

Here, “spatial” describes latent coordinates used to weight token interactions. It does not mean polygon rendering, physical 3D understanding, ray tracing, or use of GPU RT cores. This repository's name is unrelated to other projects that may use the name SpatialLM.

## Limitations and the next useful experiments

- One novel, small models, three seeds, and short training budgets. Generated text has repetitions and coherence errors.
- A 16K input window does not establish long-range retrieval or reasoning ability.
- There is no ablation against the same graph without coordinates or against a local-attention baseline yet. We cannot attribute the gain specifically to learned geometry.
- FFN widths differ to approximately match parameter counts: 628 / 512 / 570 for Spatial / Transformer / Hybrid. This is another architectural difference.
- Both models use learned absolute positions. RoPE and other stronger baselines have not been tested.
- There is no KV cache, custom sparse CUDA kernel, distributed training, or generation throughput benchmark.
- Earlier byte-token experiments use a different vocabulary, model size, precision, and evaluation setup. Their loss/perplexity cannot be directly compared with the GPT-2-token table above.

Helpful contributions include geometry ablations, long-range retrieval tests, additional corpora, profiling, incremental decoding, and independent reproductions. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Related work

Sparse and hybrid attention are established research directions. Relevant comparisons include [Longformer](https://arxiv.org/abs/2004.05150), [BigBird](https://arxiv.org/abs/2007.14062), [Routing Transformer](https://aclanthology.org/2021.tacl-1.4/), [LongNet](https://arxiv.org/abs/2307.02486), and the alternating local/global layers of [Gemma 2](https://arxiv.org/abs/2408.00118). These are related ideas, not methods this repository has outperformed in direct experiments. The novelty of this particular combination has not been established.

## Repository map

| Path | Contents |
|---|---|
| `spatiallm/model.py` | Spatial attention, full attention, hybrid blocks, chunked loss |
| `spatiallm/train.py`, `generate.py` | Byte-level training, resume, and generation |
| `spatiallm/subword_data.py`, `subword16k.py` | GPT-2 token data, 16K experiment, and generation |
| `spatiallm/experiment.py`, `benchmark.py` | Earlier quality experiments and timing benchmarks |
| `configs/` | Small CPU, GPU, and baseline configurations |
| `tests/` | Automated correctness and pipeline checks |
| `data/` | Demo corpus, source text, fixed splits, token IDs, and provenance |
| `results/` | Recorded protocols and measurements; no model weights |
| `tools/` | Data preparation, reporting, and local snapshot verification |
| `docs/EXPERIMENTS.md` | English interpretation and reproducibility details |

## Author and license

**Vladimir Ivlev (Ivlev8721)** is the project author. Copyright © 2026 Vladimir Ivlev.

Project code and original documentation are available under the [MIT license](LICENSE). Retain its copyright and permission notice when redistributing substantial portions. Third-party text, tokenizer assets, and dependencies retain their respective authorship and terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project author is not the author of the related attention methods cited above.
