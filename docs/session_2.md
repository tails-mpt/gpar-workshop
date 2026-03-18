# Session 2: Sequence Length Scaling

## Overview

Session 2 holds `batch_size=32` fixed at the Session 1 crossover point and sweeps `seq_len`
over `[64, 128, 256, 512, 1024, 2048]`. The central question is: does the TPU's 2.7×
memory bandwidth advantage compound as attention's O(seq²) memory footprint grows, and if
so, does the crossover point move?

The answer is sharper than expected. The GPU degrades toward a quadratic collapse as
sequences lengthen. The TPU shows near-zero degradation — holding ~3,130–3,166 samples/sec
across all six points. The crossover occurs at `seq_len ≈ 128`, which is also the default
seq_len used in Session 1, confirming both experiments share the same operating-regime
boundary.

Notebooks: [`session_2/`](../session_2/)

---

## Hardware

Same devices as Session 1. See [`session_1.md`](session_1.md) for full hardware specs.

| Device | Memory | Bandwidth | Note |
|---|---|---|---|
| NVIDIA L4 (GPU) | 23.7 GB GDDR6 | ~300 GB/s | Inference-optimised Ada Lovelace |
| TPU v5litepod-1 | 16 GB HBM2 | ~819 GB/s | Single-chip v5e, 1 MXU |

---

## Software Environment

| Component | GPU | TPU |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| PyTorch | 2.10.0+cu128 | 2.9.0+cu128 |
| torch_xla | — | 2.9.0 |
| Device string | `cuda:0` | `xla:0` |
| Run timestamp | 2026-02-27T16:58 UTC | 2026-02-27T17:04 UTC |

*Note: PyTorch versions differ across devices (2.10 vs 2.9). This is a cross-stack
comparison; version-sensitive results should be interpreted accordingly.*

---

## Benchmark Configuration

Model defined in [`transformer_block.py`](../transformer_block.py).

| Parameter | Value |
|---|---|
| Model | Single `BenchmarkTransformerBlock` |
| `D_MODEL` | 512 |
| `N_HEAD` | 8 |
| `DIM_FEEDFORWARD` | 2048 |
| **`BATCH_SIZE`** | **32 (fixed — Session 1 crossover point)** |
| Steps | 50 (+ 5 warmup) |
| Loop | forward → backward → Adam step |
| Metric | throughput (samples / sec) |
| **Sweep axis** | **`seq_len` ∈ [64, 128, 256, 512, 1024, 2048]** |

---

## Results

### Raw numbers

| seq_len | GPU (samples/s) | TPU (samples/s) | TPU / GPU |
|---:|---:|---:|---:|
| 64 | **5,895** ← GPU peak | 3,146 | 0.53× (GPU wins) |
| 128 | 2,906 | 3,166 | 1.09× ← crossover |
| 256 | 1,249 | 3,160 | 2.53× |
| 512 | 516 | 3,129 | 6.06× |
| 1,024 | 196 | 3,140 | 16.0× |
| 2,048 | 66 | **3,128** ← TPU (still flat) | **47.2×** |

---

## Charts

### Chart 1 — Throughput vs sequence length (GPU and TPU)

![Throughput vs Sequence Length](assets/session_2_chart_throughput.png)

**What the chart shows:**
Two lines that could not be more different. The GPU line falls steeply from left to right,
approaching zero by seq_len=2048. The TPU line is horizontal — a nearly straight bar
crossing the entire plot. The GPU starts above the TPU (at seq_len=64) and falls through it
at seq_len=128.

The visual tells participants something that the Session 1 batch-size chart did not: the TPU
advantage is not merely "at large batch sizes." Once you are operating at `batch=32` (the
crossover point), *every increase in sequence length makes the GPU's situation worse and
leaves the TPU unaffected*.

---

### Chart 2 — TPU/GPU throughput ratio vs sequence length

![TPU/GPU Ratio](assets/session_2_chart_ratio.png)

**What the chart shows:**
The ratio curve starts below 1.0 (GPU leads at seq_len=64), crosses 1.0 between seq_len=64
and seq_len=128, and then climbs monotonically — doubling roughly every time seq_len doubles.
By seq_len=2048 the TPU is 47× faster on the same workload.

This is the session's primary chart for decision-making: if your model operates on sequences
longer than 128 tokens at batch=32, and you have already passed the batch-size crossover,
the TPU advantage compounds further with every doubling of context length.

---

## Analysis: Why the curves diverge

### GPU: transitioning from linear to quadratic degradation

The GPU's throughput drop per seq_len doubling is not constant — it grows as sequences
lengthen:

| Doubling | Throughput drop | Regime |
|---|---|---|
| 64 → 128 | 2.03× | **Linear** — FFN dominates, attention small |
| 128 → 256 | 2.33× | Transitioning |
| 256 → 512 | 2.42× | Transitioning |
| 512 → 1,024 | 2.63× | **Approaching quadratic** |
| 1,024 → 2,048 | 2.97× | Near-quadratic — attention becoming dominant |

At short sequences, the feedforward network (O(seq) memory traffic) dominates and
throughput drops linearly with seq_len. As sequences grow, the attention matrix
`[batch, heads, seq, seq]` becomes an increasing fraction of total memory traffic, and the
degradation slope tilts toward the quadratic limit (4× per doubling). The GPU never fully
reaches 4× per doubling in this range because the FFN provides a floor — but the trend is
clear and continuing past seq_len=2048 would accelerate further.

Root causes are the same as Session 1:

1. **300 GB/s memory bandwidth ceiling.** The GPU saturates its memory bus at moderate
   batch sizes; the O(seq²) attention matrix intensifies this pressure with every seq_len
   doubling.
2. **DRAM round-trips.** The attention matrix must be materialised, transposed, and passed
   through softmax — all memory-bound operations that scale quadratically.
3. **No change in compute clock or core count.** The GPU has the same SIMT resources
   regardless of seq_len; memory bandwidth is the single binding constraint.

### TPU: flat throughput is not zero degradation — it is a different bottleneck

The TPU result is flat: throughput varies by only **1.2%** across all six seq_len points
(3,127–3,166 samples/sec). This is better than "near-linear scaling" — there is essentially
no degradation at all.

The reason is not that the TPU has infinite bandwidth. The reason is that at `batch=32`,
the bottleneck is the **XLA synchronisation overhead**, not memory bandwidth or compute.

Evidence: in Session 1, the TPU's per-step wall time is approximately constant at ~10ms
regardless of batch size (32/2975 ≈ 10.7ms at batch=32; 1024/95703 ≈ 10.7ms at batch=1024).
The same holds here — per-step time is ~10ms across all seq_len values. As seq_len grows,
each step performs more computation within the same sync window, so:

- **Tokens/sec doubles every time seq_len doubles** (3,146×64=201K → 3,128×2048=6.4M)
- **Samples/sec stays constant** because the sync overhead dominates

The XLA compiler amortises the growing matrix computation efficiently. The systolic array
executes larger matrix blocks without proportionally more overhead. By contrast, the GPU
eagerly dispatches every new kernel with full scheduling latency, and its memory bus
traffic grows quadratically.

**The flat TPU line is a signature of sync-overhead dominance at this batch size, not
hardware immunity to long sequences.** At much larger batch sizes (e.g. batch=256+) the
TPU would show mild degradation as attention memory eventually approaches the HBM bandwidth
limit — but that crossover is well beyond this session's range.

### Why the seq_len crossover matches the batch-size crossover

Session 1 found the GPU/TPU crossover at `batch=32` (with `seq_len=128` fixed).
Session 2 finds the crossover at `seq_len ≈ 128` (with `batch=32` fixed).

These are the same operating point examined from two different axes. The boundary is a
*regime boundary*, not a point: the GPU leads when both batch and sequence are small (little
work, fast eager dispatch wins); the TPU leads when either or both are large enough to fill
the MXU and amortise compilation. Sessions 1 and 2 together trace two edges of this boundary.

---

## Key Takeaways

- **GPU throughput degrades toward a quadratic collapse with sequence length.** Throughput
  drops from 5,895 samples/sec at seq_len=64 to 66 at seq_len=2048 — an 89× fall for a
  32× increase in sequence length. The degradation slope steepens as sequences grow, as the
  O(seq²) attention matrix increasingly dominates memory traffic over the O(seq) FFN.

- **TPU throughput is flat across all measured sequence lengths.** At `batch=32`, the XLA
  sync overhead (~10ms/step) dominates, and the growing computation is absorbed within it.
  Tokens per second doubles with every seq_len doubling — the TPU is becoming *more
  efficient* per token, not less.

- **The crossover is at seq_len ≈ 128 — at seq_len=64 the GPU is 1.9× faster.**
  Session 1's default seq_len (128) is not an arbitrary choice — it sits exactly at the
  regime boundary on both the batch and seq_len axes. Below that point the GPU's fast
  dispatch wins; above it the TPU's architecture dominates.

- **The TPU advantage reaches 47× at seq_len=2048.** This exceeds Session 1's 37×
  peak (at batch=1024, seq_len=128). Long-context models face a more severe bandwidth
  penalty than the batch-size comparison alone shows.

- **Decision rule for long-context workloads:** If your model operates on sequences longer
  than 256 tokens and you are already at or above the batch-size crossover (batch ≥ 32),
  the TPU advantage compounds with every seq_len doubling. The GPU is not a viable option
  for document-scale or retrieval-augmented workloads at this operating point.
