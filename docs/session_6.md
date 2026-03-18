# Session 6: Framework Interoperability — JAX/Flax and PyTorch/XLA on Shared Hardware

## Overview

Sessions 1–5 reached the GPU and TPU through PyTorch and PyTorch/XLA. Session 6 asks
whether the ML framework matters, or whether the XLA compiler decides everything underneath.

Two production ML frameworks — **JAX/Flax** and **PyTorch/XLA** — are compared on
the same hardware, benchmarking the same Transformer encoder architecture. Both ultimately
compile to XLA HLO before executing. The question is whether the two front-ends generate
equivalent programs, and what the performance consequences are if they don't.

This session illustrates a broader point about the ML hardware stack: **shared hardware
does not mean equivalent execution**. The same accelerator can behave very differently
depending on how a framework traces, compiles, and dispatches operations. Understanding
this is critical for practitioners choosing or migrating between frameworks.

The same Transformer encoder architecture used in Sessions 1–5 is re-implemented in
Flax/Linen (`flax_model.py`) and benchmarked across the same batch sweep as Session 1
([4, 8, 16, 32, 64, 128, 256, 512, 1024]).

Notebooks: [`session_6/`](../session_6/)

---

## Hardware

Same devices as Sessions 1–5. See [`session_1.md`](session_1.md) for full hardware specs.

| Device | Memory | Note |
|---|---|---|
| NVIDIA L4 (GPU) | 23.7 GB GDDR6 | JAX uses CUDA XLA backend (not cuDNN-native like PyTorch) |
| TPU v5litepod-1 | 16 GB HBM2 | JAX and PyTorch/XLA both compile to XLA HLO here |

---

## Software Environment

| Component | GPU | TPU |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| JAX | 0.9.0.1 | 0.9.0.1 |
| Flax | 0.12.4 | 0.12.4 |
| Backend | `jax[cuda12]` | `jax[tpu]` |
| Run timestamp | 2026-02-28T00:38 UTC | 2026-02-28T00:36 UTC |

---

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Model | `FlaxTransformerBlock` (matches `BenchmarkTransformerBlock` architecture) |
| `D_MODEL` | 512 |
| `N_HEAD` | 8 |
| `DIM_FEEDFORWARD` | 2048 |
| `SEQ_LEN` | 128 (fixed) |
| `BATCH_SIZES` | 4, 8, 16, 32, 64, 128, 256, 512, 1024 |
| Steps (batch ≤ 256) | 50 (+ 5 warmup) |
| Steps (batch ≥ 512) | 20 (+ 5 warmup) — GPU only, adaptive to prevent hang |
| Loop | forward → backward → Adam step (`jax.value_and_grad` + `optax.adam`) |
| Metric | throughput (samples/sec), latency (ms/step), first-call compilation time (s) |

### Key difference from PyTorch sessions

JAX uses **functional parameter management**: model weights are a pytree returned by
`model.init()` and passed explicitly into `model.apply()` on every call. There is no
stateful model object. The `jax.jit`-compiled `train_step` function captures the model
and optimizer as Python closures; JAX traces through both the forward pass and the
`jax.value_and_grad` backward pass in a single XLA program.

---

## Architecture equivalence

`FlaxTransformerBlock` in `flax_model.py` replicates `BenchmarkTransformerBlock` exactly:

```
# PyTorch (transformer_block.py)          # Flax (flax_model.py)
x = norm1(x + attn(x, x, x)[0])          attn_out = MultiHeadDotProductAttention(...)(x, x)
x = norm2(x + ff(x))                      x = LayerNorm()(x + attn_out)
                                           ff = Dense(DIM_FEEDFORWARD)(x) → gelu → Dense(D_MODEL)(x)
                                           x = LayerNorm()(x + ff)
```

Both are **Post-LayerNorm** (norm applied after residual). The feedforward uses GELU.
The attention block is self-attention (query = key = value = x) with
`qkv_features = out_features = D_MODEL`.

---

## Results

### GPU — Throughput (samples/sec) and Compilation Time

JAX/Flax results vs PyTorch Session 1 baseline:

| Batch | JAX/Flax | PyTorch (S1) | JAX/PT ratio | Compile time |
|------:|---------:|-------------:|-------------:|-------------:|
| 4 | 3,186 | 1,388 | 2.30× | 15.6 s |
| 8 | 4,409 | 2,521 | 1.75× | 18.0 s |
| 16 | 5,946 | 2,866 | 2.07× | 19.5 s |
| 32 | 6,070 | 2,895 | 2.10× | 22.6 s |
| 64 | 5,304 | 2,760 | 1.92× | 18.8 s |
| 128 | 4,823 | 2,688 | 1.79× | 19.5 s |
| 256 | 4,660 | 2,653 | 1.76× | 20.4 s |
| 512 | 4,571 | 2,618 | 1.75× | 22.0 s |
| 1,024 | 4,463 | 2,575 | 1.73× | 26.5 s |

**Unexpected finding:** JAX/Flax is **1.7–2.3× faster** than PyTorch native on GPU
across all batch sizes. This reverses the pre-run expectation. See Analysis below.

### TPU — Throughput (samples/sec) and Compilation Time

JAX/Flax results vs PyTorch/XLA Session 1 baseline:

| Batch | JAX/Flax | PyTorch/XLA (S1) | JAX/PT ratio | Compile time |
|------:|---------:|----------------:|-------------:|-------------:|
| 4 | 8,454 | 370 | 22.8× | 2.5 s |
| 8 | 15,867 | 745 | 21.3× | 3.6 s |
| 16 | 26,983 | 1,497 | 18.0× | 3.8 s |
| 32 | 38,689 | 2,984 | 13.0× | 3.9 s |
| 64 | 39,160 | 5,968 | 6.6× | 2.8 s |
| 128 | 36,405 | 11,900 | 3.1× | 3.3 s |
| 256 | 30,820 | 23,862 | 1.3× | 3.3 s |
| 512 | 28,620 | 47,580 | **0.60×** | 3.5 s |
| 1,024 | 28,418 | 94,795 | **0.30×** | 3.5 s |

**Two unexpected findings on TPU:**
1. JAX is **dramatically faster at small batch sizes** — up to 22.8× at batch=4 — but
   **PyTorch/XLA wins decisively at large batch sizes** (batch ≥ 512).
2. JAX TPU throughput **plateaus around batch=64** (~39k samples/sec) and then **declines**,
   while PyTorch/XLA scales linearly all the way to batch=1024 (95k samples/sec).

The crossover is between batch=256 (JAX still ahead 1.3×) and batch=512 (PyTorch/XLA 1.67× faster).

---

## Analysis

### Why JAX outperforms PyTorch on GPU (whole-graph fusion)

JAX's `jax.jit` compiles the *entire* train step — forward pass, `jax.value_and_grad`
backward, and the `optax.adam` optimizer update — into a single XLA program. XLA can
then fuse adjacent operations across this full graph: it eliminates intermediate tensor
materializations, reorders memory-bound ops, and generates a single kernel sequence
that minimises GDDR6 round-trips.

PyTorch's eager mode dispatches each operation as a separate kernel call: the forward
pass, then the backward pass (autograd), then each optimizer parameter update. Each call
crosses the Python–CUDA boundary and potentially writes an intermediate result to GDDR6
before the next call reads it back. For a single Transformer block, this per-operation
overhead is the dominant cost at small batch sizes — not raw compute.

The XLA advantage declines from 2.30× at batch=4 to 1.73× at batch=1024 as the GPU
becomes more compute-saturated and the overhead is diluted.

### Why JAX plateaus and declines on TPU at large batch sizes

On TPU, both frameworks compile to XLA HLO — but they generate *different* programs.

**PyTorch/XLA** traces through the PyTorch autograd graph and emits a series of matrix
operations. Its throughput scales perfectly linearly with batch size (Session 1 data),
indicating the MXU is purely compute-bound and XLA is efficiently pipelining operations
through the systolic array.

**JAX/Flax** compiles the full train step (including Adam optimizer) in a single
`jax.jit` trace. The resulting XLA program materialises all intermediate gradient and
optimizer state tensors in a single step. At large batch sizes, the gradient tensors
(same shape as the activations, scaling with batch size) and the Adam moment updates
create HBM pressure that the JAX-generated program cannot fully hide behind compute.

The plateau at batch=64 (~39k samples/sec) and subsequent decline suggests JAX's XLA
program becomes **HBM bandwidth-bound at large batch sizes** on the v5litepod, despite
the MXU still being available. PyTorch/XLA's approach — which separates the forward,
backward, and optimizer steps into a sequence of `mark_step()` flushes — allows XLA to
compile smaller, more focused kernels that pipeline better at scale.

### What this reveals about framework-hardware abstraction

Both frameworks claim XLA as their backend. Both run on the same physical hardware. Yet
they deliver up to 22.8× different throughput (batch=4 on TPU) and reverse their
relative ranking depending on batch size. The key lesson is:

**"Same backend" does not mean "same program."** The XLA HLO emitted by JAX and
PyTorch/XLA differs in:
- Graph granularity (JAX: one program per train step; PyTorch/XLA: one program per flush)
- Memory layout decisions made during lowering
- Which operations XLA can fuse once it receives the graph

Practitioners choosing between frameworks for TPU workloads should benchmark both at
their actual batch size — the decision depends on where on the batch-size axis they operate.

### Compilation cost comparison

| Device | JAX compile (per shape) | PyTorch equivalent |
|---|---|---|
| GPU | 15–27 s | ~0 ms (cuDNN kernel lookup) |
| TPU | 2.5–3.9 s | First `mark_step()`: ~seconds |

JAX TPU compilation is ~7× faster than JAX GPU compilation, reflecting that XLA is more
native to TPU and the compilation pipeline is more optimised for that target.

---

## Charts

### Chart 1 — Throughput vs batch size (JAX/Flax vs PyTorch)

![Throughput chart](assets/session_6_chart_throughput.png)

**GPU panel:** JAX/Flax sits above the PyTorch line across all batch sizes.
JAX peaks at batch=32 (~6,070 s/s) then declines to ~4,460 s/s at batch=1024 — a
characteristic of XLA whole-graph fusion hitting bandwidth limits at larger footprints.
PyTorch is roughly flat from batch=16 onward (~2,600–2,900 s/s).

**TPU panel:** JAX peaks at batch=32–64 (~39k samples/sec) and declines. PyTorch/XLA
rises linearly to ~95k samples/sec at batch=1024. The two lines cross between
batch=256 and batch=512.

### Chart 2 — JAX / PyTorch throughput ratio per device

![Ratio chart](assets/session_6_chart_ratio.png)

**GPU:** Ratio stays above 1.0× across all batch sizes (JAX always faster on GPU).
**TPU:** Ratio starts at 22.8× (batch=4), falls through 1.0× at batch=256–512,
reaches 0.30× at batch=1024 (PyTorch/XLA 3.3× faster).

### Chart 3 — Compilation time per input shape

![Compile chart](assets/session_6_chart_compile.png)

GPU compilation: 15–27 s (grows slightly with batch size — larger input shapes
take longer to lower to PTX). TPU compilation: 2.5–3.9 s (relatively flat —
the XLA→TPU lowering is more efficient than XLA→CUDA).

---

## Key Takeaways

- **On GPU, JAX/Flax is 1.7–2.3× faster than PyTorch native.** XLA's whole-graph
  compilation of the full train step eliminates intermediate tensor materialisation costs
  that PyTorch eager dispatch incurs. The advantage is larger at small batch sizes where
  bandwidth overhead dominates.

- **On TPU at small batch sizes, JAX is dramatically faster than PyTorch/XLA** —
  up to 22.8× at batch=4. The JAX XLA program is far more aggressive about fusing
  small-batch operations than PyTorch/XLA's `mark_step()` approach.

- **On TPU at large batch sizes, PyTorch/XLA wins decisively.** JAX throughput
  plateaus at ~39k samples/sec (batch=64) and declines to 28k at batch=1024, while
  PyTorch/XLA scales linearly to ~95k. The crossover is around batch=300–400.

- **"Both use XLA" does not mean "equivalent performance."** The frameworks generate
  meaningfully different XLA programs. The performance crossover is batch-size-dependent,
  and the winning framework can differ by up to 22×.

- **JAX TPU compilation is 2.5–3.9 s per shape** — roughly 7× faster than JAX GPU
  (15–27 s). PyTorch/XLA first-step compilation is in a similar range on TPU.

---

## Decision Rule from This Session

| Scenario | Recommendation | Reason |
|---|---|---|
| GPU, existing PyTorch code | **Stay on PyTorch** | JAX faster, but migration cost exceeds benefit for existing codebases |
| GPU, new project, fixed batch | **Consider JAX** | 1.7–2.3× throughput gain is real; compilation cost amortises quickly |
| TPU, large batch (≥ 512) | **PyTorch/XLA** | Scales linearly to 95k samples/sec; JAX declines to 28k |
| TPU, small batch (≤ 64) | **JAX** | Up to 22.8× faster; PyTorch/XLA is slow here |
| TPU, batch ≈ 256 | **Either** | Performance is within 1.3× — ergonomics decides |
