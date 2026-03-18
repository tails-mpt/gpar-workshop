# GPU vs TPU for Transformer Workloads — Workshop Summary

> **Benchmark:** Single Transformer encoder block (D\_MODEL=512, N\_HEAD=8, DIM\_FEEDFORWARD=2048, SEQ\_LEN=128)
> **Hardware:** NVIDIA L4 GPU (23.7 GB GDDR6, ~300 GB/s, ~121 TFLOPS FP16) vs Google TPU v5litepod-1 (16 GB HBM2, ~819 GB/s, ~393 TFLOPS BF16)
> **Pricing (early 2026, GCP on-demand):** GPU $0.70/hr · TPU $1.20/hr

---

## Session 1 — GPU vs TPU Throughput Scaling

**Question:** At what batch size does the TPU overtake the GPU, and how large does the gap get?

| Batch | GPU (s/s) | TPU (s/s) | TPU / GPU |
|------:|----------:|----------:|----------:|
| 4 | 1,388 | 370 | 0.27× |
| 8 | 2,521 | 745 | 0.30× |
| 16 | **2,866** | 1,497 | 0.52× |
| **32** | 2,895 | **2,984** | **1.03× ← crossover** |
| 64 | 2,760 | 5,968 | 2.16× |
| 128 | 2,688 | 11,900 | 4.43× |
| 256 | 2,653 | 23,862 | 9.0× |
| 512 | 2,618 | 47,580 | 18.2× |
| 1,024 | 2,575 | 94,795 | **36.8×** |

**Why the GPU plateaus:** The L4's 300 GB/s memory bus saturates at batch=16. Larger batches add no throughput — data queues behind the memory bus.

**Why the TPU scales linearly:** The systolic array MXU is compute-bound across all tested batch sizes. XLA compiles the full forward/backward/step graph once; per-step overhead is negligible. TPU throughput from batch=4 to batch=1024 is 256.2× — nearly the theoretical 256× perfect linear scaling.

**Key takeaways:**
- GPU peaks at batch=16 (~2,866 s/s); crossover at batch=32; TPU is 37× faster at batch=1024.
- For small-batch latency-sensitive inference (batch ≤ 16), the GPU is competitive and simpler to operate.
- For high-throughput training, the TPU's architecture is categorically superior.

---

## Session 2 — Sequence Length Scaling

**Question:** How does each device respond as attention's O(seq²) memory footprint grows?

*Setup: batch=32 fixed (Session 1 crossover), seq\_len swept [64, 128, 256, 512, 1024, 2048].*

| seq\_len | GPU (s/s) | TPU (s/s) | TPU / GPU |
|---------:|----------:|----------:|----------:|
| 64 | **5,895** | 3,146 | 0.53× (GPU wins) |
| **128** | 2,906 | 3,166 | **1.09× ← crossover** |
| 256 | 1,249 | 3,160 | 2.53× |
| 512 | 516 | 3,129 | 6.06× |
| 1,024 | 196 | 3,140 | 16.0× |
| 2,048 | 66 | **3,128** | **47.2×** |

**GPU degradation:** Throughput drops from 5,895 → 66 s/s (89× fall) for a 32× sequence increase. The O(seq²) attention matrix increasingly dominates memory traffic, pushing the degradation slope toward quadratic.

**TPU immunity:** Throughput varies only 1.2% across all six seq\_len points. At batch=32, the bottleneck is XLA synchronisation overhead (~10ms/step), not memory bandwidth. The growing computation is absorbed within the same sync window.

**Key takeaways:**
- GPU is viable only for seq\_len ≤ 128 at batch=32. Every doubling past that cuts throughput by 2–3×.
- TPU advantage reaches 47× at seq\_len=2048 — larger than Session 1's 37× peak.
- For long-context workloads (≥ 256 tokens) above the batch crossover, the GPU is not a viable option.

---

## Session 3 — Model Depth and Memory Limits

**Question:** How do throughput and VRAM scale from 1 to 24 layers? Where does each device run out of memory?

*Setup: batch=64, seq\_len=128, n\_layers ∈ [1, 2, 4, 6, 8, 12, 16, 24].*

### GPU results

| n\_layers | GPU (s/s) | Peak VRAM (MB) | VRAM % of L4 |
|----------:|----------:|---------------:|-------------:|
| 1 | 2,635 | 531 | 2.3% |
| 4 | 599 | 1,606 | 7.0% |
| 8 | 295 | 3,039 | 13.2% |
| 12 (BERT) | 199 | 4,473 | 19.4% |
| 16 | 152 | 5,891 | 25.6% |
| 24 (GPT-2) | 101 | 8,772 | 38.1% |

VRAM grows at ~357 MB per encoder layer (activations + gradients + Adam moments + allocator overhead). **No OOM in [1–24] layers at batch=64.** Extrapolating: OOM boundary ≈ 60 layers.

### TPU results

| n\_layers | TPU (s/s) | TPU / GPU |
|----------:|----------:|----------:|
| 1 | 6,033 | 2.29× |
| 6 | 1,059 | 2.70× |
| 12 | 519 | 2.61× |
| 24 | 255 | 2.52× |

**No OOM on TPU within [1–24] layers** (16 GB HBM2 sufficient). TPU/GPU ratio is stable at ~2.5× regardless of depth.

**Key takeaways:**
- Throughput scales as 1/n\_layers on both devices — each extra layer adds a fixed compute cost.
- BERT-base (12 layers) fits at 4,473 MB (19% of L4). GPT-2 scale (24 layers) uses only 38%.
- The 2.5× TPU throughput advantage from Session 1 holds at all tested depths.
- If VRAM exceeds 80% of device capacity, switch to BF16 (Session 5) before increasing depth.

---

## Session 4 — Static vs Dynamic Computation Graphs

**Question:** What happens when the model uses data-dependent control flow? Can it be fixed?

*Setup: batch=256, seq\_len=128.*

### Part A — Abstract variants

| Variant | GPU (s/s) | TPU (s/s) | GPU vs Static | TPU vs Static |
|---|---:|---:|---:|---:|
| StaticFF (baseline) | 2,038 | 24,017 | 1.000× | 1.000× |
| DynamicFF (`if out.mean() > 0`) | 1,199 | 9,453 | −41% | **−61%** |
| StaticEquivalentFF (`torch.where`) | 1,144 | 23,806 | −44% | **−0.9%** |

`torch.where` recovers **98.5%** of the TPU penalty by keeping the condition inside the XLA graph.

**Why the penalty differs:**
- **TPU:** `if tensor > 0` forces mid-step XLA sync + graph recompilation. `torch.where` eliminates this entirely.
- **GPU:** `out.mean()` requires a CUDA-to-CPU sync even in eager mode. `torch.where` does not help — the reduction still executes.

### Part B — Realistic scenarios

| Scenario | GPU (s/s) | TPU (s/s) | TPU/GPU |
|---|---:|---:|---:|
| StaticFF (baseline) | 2,038 | 24,017 | 11.8× |
| B1: Padding mask (tensor-op) | 1,162 | 23,960 | **20.6×** |
| B2: Early exit (dynamic) | 1,038 | 7,522 | 7.2× |
| B2: Early exit (static mask) | 603 | 18,577 | 30.8× |
| B3: Fixed-length loop | 407 | 15,565 | 38.2× |
| B3: Variable-length loop | 806 | 15,520 | 19.3× |

**Key takeaways:**
- Padding masks are essentially free on TPU (−0.2%). This is the most common dynamic NLP pattern.
- For most tensor-valued branches, `torch.where` / `masked_fill` is a one-line fix with near-full TPU recovery.
- Early exit is genuinely complex: static mask version favours TPU; dynamic version favours GPU.
- Most research-code branches can be refactored. When they cannot, profile both frameworks.

---

## Session 5 — Precision and Dtype (FP32 / FP16 / BF16 / INT8)

**Question:** How much does lower precision improve throughput and reduce memory?

### GPU throughput speedup over FP32

| Batch | FP16 speedup | BF16 speedup |
|------:|-------------:|-------------:|
| 32 | 2.09× | 2.11× |
| 256 | 2.10× | 2.07× |
| 1,024 | 2.10× | 2.03× |

FP16 and BF16 deliver a consistent **~2.0–2.1× throughput gain** across all batch sizes, with **21–29% VRAM reduction** (batch-dependent; smaller saving at small batch where fixed FP32 optimizer states dilute activation savings).

### TPU — BF16 vs FP32

| Batch | FP32 (s/s) | BF16 (s/s) | BF16 speedup |
|------:|-----------:|-----------:|-------------:|
| 32 | 3,006 | 2,965 | **0.986×** |
| 1,024 | 97,749 | 94,522 | **0.967×** |

BF16 is **1.4–3.3% slower** than FP32 on TPU v5litepod. XLA may choose a more aggressively fused FP32 kernel, and mixed-precision cast operations add latency for this model size.

### INT8

- **GPU:** `torch.ao.quantization.quantize_dynamic` runs on CPU, not GPU Tensor Cores — measured ~400–498 s/s (3–4× slower than GPU FP32). True GPU INT8 requires a CUDA-aware quantisation path.
- **TPU:** INT8 weight casting errored in torch\_xla 2.9.0.

**Key takeaways:**
- **GPU:** always use BF16 (or FP16 + GradScaler on older hardware). 2× throughput and 21–29% VRAM savings are effectively free.
- **TPU:** use FP32 as the default for this model size. BF16 is consistently slightly slower. Benchmark per-model before switching.
- At batch=1024, the TPU FP32 throughput (97,749 s/s) is ~85× the GPU FP32 (1,146 s/s). Dtype is a secondary lever on the TPU; batch size is the primary one.

---

## Session 6 — Framework Interoperability (JAX/Flax vs PyTorch/XLA)

**Question:** Does it matter whether you use JAX/Flax or PyTorch/XLA on the same hardware?

*Both frameworks compile to XLA HLO. JAX: 0.9.0.1, Flax: 0.12.4.*

### GPU — JAX/Flax vs PyTorch

| Batch | JAX/Flax (s/s) | PyTorch (s/s) | JAX / PyTorch | Compile time |
|------:|---------------:|--------------:|--------------:|-------------:|
| 4 | 3,186 | 1,388 | 2.30× | 15.6 s |
| 32 | 6,070 | 2,895 | 2.10× | 22.6 s |
| 256 | 4,660 | 2,653 | 1.76× | 20.4 s |
| 1,024 | 4,463 | 2,575 | 1.73× | 26.5 s |

JAX/Flax is **1.7–2.3× faster on GPU** by compiling the entire train step (forward + backward + Adam) into a single XLA program, eliminating intermediate tensor writes to GDDR6 that PyTorch eager dispatch incurs.

### TPU — JAX/Flax vs PyTorch/XLA

| Batch | JAX/Flax (s/s) | PyTorch/XLA (s/s) | JAX / PyTorch | Compile time |
|------:|---------------:|------------------:|--------------:|-------------:|
| 4 | 8,454 | 370 | **22.8×** | 2.5 s |
| 32 | 38,689 | 2,984 | 13.0× | 3.9 s |
| 64 | 39,160 | 5,968 | 6.6× | 2.8 s |
| 256 | 30,820 | 23,862 | 1.3× | 3.3 s |
| **512** | 28,620 | **47,580** | **0.60×** | 3.5 s |
| 1,024 | 28,418 | 94,795 | **0.30×** | 3.5 s |

JAX plateaus at ~39k s/s (batch=64) and declines. PyTorch/XLA scales linearly to 95k at batch=1024. **Crossover: batch ≈ 300–400.**

**Why:** JAX's single-program compilation creates HBM pressure at large batch sizes. PyTorch/XLA's `mark_step()` flushes produce smaller, more focused kernels that pipeline more efficiently through the MXU at scale.

**Key takeaways:**
- "Both use XLA" does not mean equivalent performance — frameworks generate different XLA programs.
- Use JAX on TPU for small batches (≤ 64); use PyTorch/XLA for large batches (≥ 512).
- JAX GPU compilation costs 15–27s per unique shape vs near-zero for PyTorch cuDNN.

**Decision table:**

| Scenario | Recommendation |
|---|---|
| GPU, existing PyTorch codebase | Stay on PyTorch |
| GPU, new project | Consider JAX (1.7–2.3× throughput gain) |
| TPU, batch ≥ 512 | PyTorch/XLA (scales to 95k s/s) |
| TPU, batch ≤ 64 | JAX (up to 22.8× faster) |
| TPU, batch ≈ 256 | Either — within 1.3× |

---

## Economic Summary — Samples per Dollar

*Based on FP32 throughput from Session 1. GPU $0.70/hr, TPU $1.20/hr.*

| Batch | GPU (samples/$) | TPU (samples/$) | TPU cheaper? |
|------:|----------------:|----------------:|:---:|
| 4 | 7.1M | 1.1M | No |
| 16 | 14.7M | 4.5M | No |
| 32 | 14.9M | 9.0M | No |
| **64** | **14.2M** | **17.9M** | **Yes — 1.26×** |
| 128 | 13.8M | 35.7M | Yes — 2.6× |
| 256 | 13.6M | 71.6M | Yes — 5.3× |
| 512 | 13.4M | 142.7M | Yes — 10.6× |
| 1,024 | 13.2M | 284.4M | Yes — **21.5×** |

**Cost crossover: batch=64.** Below that, the GPU's lower hourly rate wins despite lower throughput. Above it, the TPU's throughput advantage outpaces its 1.71× price premium — and the gap grows steeply with every batch doubling.

---

## Decision Guide

| Workload | Recommended device | Rationale |
|---|---|---|
| Small-batch inference (batch ≤ 16) | **GPU** | Faster at low batch; simpler to operate |
| Long-context inference (seq\_len ≥ 256, batch ≥ 32) | **TPU** | GPU degrades quadratically; TPU is immune |
| High-throughput training (batch ≥ 64) | **TPU** | Cost crossover at batch=64; 21.5× better samples/$ at batch=1024 |
| BERT-base depth (12 layers) | **Either** | Both fit comfortably; TPU is 2.5× faster throughput |
| Data-dependent control flow | **Profile both** | `torch.where` recovers TPU penalty for most patterns |
| Dtype optimisation | GPU → BF16; TPU → FP32 | 2× GPU gain; BF16 slightly hurts TPU for this model size |
| Framework choice (TPU) | JAX (small batch) / PyTorch/XLA (large batch) | Crossover ~batch=300–400 |
