# Session 1: GPU vs TPU Throughput Scaling

## Overview

Session 1 benchmarks the throughput scaling behaviour of a standard Transformer
encoder block across two accelerator classes — GPU (NVIDIA L4) and TPU (v5litepod-1)
— as batch size increases from 4 to 1024. The central question is: does adding more
work (larger batches) translate into more throughput, and why do the two devices
respond so differently?

Notebooks: [`session_1/`](../session_1/)

---

## Hardware

### GPU — NVIDIA L4

| Property | Value |
|---|---|
| Architecture | Ada Lovelace (2022) |
| VRAM | 23.7 GB GDDR6 |
| Memory Bandwidth | ~300 GB/s |
| FP16 Tensor Core TFLOPS | ~121 TFLOPS |
| INT8 TOPS | ~242 TOPS |
| CUDA Cores | 7,680 |
| TDP | 72 W |
| Form Factor | PCIe (low-profile) |

The L4 is an inference-optimised card. Compared to training-class GPUs (A100, H100)
it trades peak FLOPS and memory bandwidth for low power draw and dense rack
deployability.

### TPU — Google TPU v5e (v5litepod-1)

| Property | Value |
|---|---|
| Chip family | TPU v5e (v5lite) |
| Chips used | 1 |
| HBM Memory per chip | 16 GB HBM2 |
| Memory Bandwidth | ~819 GB/s (HBM2) |
| BF16 MXU TFLOPS per chip | ~393 TFLOPS |
| INT8 TOPS per chip | ~786 TOPS |
| Interconnect | ICI (Inter-Chip Interconnect) |
| Deployment | Google Cloud TPU VM |

The v5e is Google's efficiency-focused TPU generation. Each chip contains a systolic
array Matrix Multiply Unit (MXU) purpose-built for dense matrix operations. Memory
bandwidth is nearly 3× that of the L4.

---

## Software Environment

| Component | GPU | TPU |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| PyTorch | 2.10.0+cu128 | 2.9.0+cu128 |
| torch_xla | — | 2.9.0 |
| Device string | `cuda:0` | `xla:0` |
| TPU chip | — | v5litepod-1 |
| N chips | — | 1 |

---

## Benchmark Configuration

Defined in [`transformer_block.py`](../transformer_block.py).

| Parameter | Value |
|---|---|
| Model | Single Transformer encoder block |
| `D_MODEL` | 512 |
| `N_HEAD` | 8 |
| `DIM_FEEDFORWARD` | 2048 |
| `SEQ_LEN` | 128 tokens |
| Steps | 50 (+ 5 warmup) |
| Loop | forward → backward → Adam step |
| Metric | throughput (samples / sec) |
| Batch sizes | 4, 8, 16, 32, 64, 128, 256, 512, 1024 |

---

## Results

### Raw numbers

| Batch | GPU (samples/s) | TPU (samples/s) |
|---:|---:|---:|
| 4 | 1,388 | 370 |
| 8 | 2,521 | 745 |
| 16 | 2,866 | 1,496 |
| 32 | **2,895** ← GPU peak | 2,984 |
| 64 | 2,760 | 5,968 |
| 128 | 2,688 | 11,900 |
| 256 | 2,653 | 23,862 |
| 512 | 2,618 | 47,580 |
| 1024 | 2,575 | **94,795** ← TPU peak |

---

## Charts

### Chart 1 — GPU vs TPU: throughput vs batch size

![Throughput vs Batch Size](assets/session_1_chart_throughput.png)

**What the chart shows:**
The combination of a log₂ X-axis and a linear Y-axis makes the divergence immediately
visible. The GPU (green) is a near-flat band compressed near the bottom of the chart.
The TPU (blue) starts below the GPU at small batch sizes, crosses it at batch=32, and
then climbs steeply — eventually reaching ~94,795 samples/sec at batch=1024, a value
so large that the GPU's plateau (~2,600) becomes visually indistinguishable from zero
on this scale.

The visual "crush" of the GPU is not an artefact — it is the correct representation.
At batch=1024 the TPU processes **37× as many samples per second** as the GPU on the
same workload.

---

### Chart 2 — GPU vs TPU: scaling and crossover

![GPU vs TPU Crossover](assets/session_1_chart_crossover.png)

**What the chart shows:**
The same data as Chart 1, now with a red dotted line marking **batch=32** — the
exact batch size at which TPU throughput (2,984 samples/sec) first exceeds GPU
throughput (2,895 samples/sec). Notably, batch=32 is also the GPU's performance
peak: the TPU overtakes the GPU precisely when the GPU has nothing more to give.

Left of the line: GPU leads. The GPU's fast CUDA dispatch and mature cuDNN
kernels give it an advantage when there isn't enough work to keep the TPU's
systolic array busy. At batch=16 the GPU is already near its ceiling while the TPU
is still warming up its compile-and-dispatch pipeline.

Right of the line: TPU throughput scales proportionally to batch size — every
doubling of batch size nearly doubles TPU output (reflecting near-perfect MXU
utilisation). The GPU line is almost perfectly flat — it has hit its memory
bandwidth ceiling and adding more work produces no additional throughput.

The gap widens steadily: at batch=64 the TPU is 2.1× faster; at batch=256
it is 9×; at batch=1024 it is **37×**.

---

### Chart 3 — TPU / GPU speedup ratio across batch sizes

![TPU Speedup over GPU](assets/session_1_chart_speedup.png)

**What the chart shows:**
The Y-axis is the TPU/GPU throughput ratio. The red dotted line at 1× marks parity.
At small batch sizes the GPU leads (ratio < 1): the TPU is only 0.27× as fast at
batch=4 and 0.52× at batch=16. The ratio crosses 1× at batch=32, then accelerates
— reaching 9× at batch=256 and **37× at batch=1024**. The shaded blue region (TPU
faster) grows as batch size doubles, making the increasing advantage visible in a
single glance.

This view is complementary to Charts 1 and 2: instead of comparing absolute
throughput, it directly answers "by how much does the choice of hardware matter at
each operating point?"

---

## Analysis: Why the curves diverge

### GPU: memory bandwidth ceiling

The L4 peaks at **batch=32** (2,895 samples/sec) and declines for all larger batches.
The curve is essentially flat from batch=16 onward — throughput gains less than 1%
going from batch=16 to batch=32, then falls monotonically:

1. **Memory bandwidth bottleneck (~300 GB/s).** The GPU's CUDA cores are fed data
   through the memory bus. Once the bus is saturated, adding more samples per step
   does not produce additional throughput — data simply queues up.
2. **CUDA scheduling overhead.** Dispatching and synchronising kernel launches adds
   per-step overhead that grows with batch size, partially eroding any marginal gain.
3. **Cache pressure.** Larger activations spill out of L2/shared memory, increasing
   DRAM round-trips and further penalising large batches.
4. **Slight throughput decrease beyond batch=32** (~11% from peak to batch=1024)
   is consistent with the above — pure memory-traffic penalty.

### TPU: near-perfect linear scaling

The TPU scales almost exactly **2× for every 2× increase in batch size**:

| Ratio | Value |
|---|---|
| batch=1024 / batch=4 | 94,795 / 370 = **256.2×** |
| Expected if perfectly linear | 1024 / 4 = **256×** |

The deviation from perfect linearity is less than 1%. This arises from:

1. **Systolic array (MXU) architecture.** The MXU is specialised hardware for
   matrix multiplications. Doubling the batch dimension doubles the work, and the
   MXU executes that work in proportionally doubled time — no hidden overhead.
2. **High HBM bandwidth (~819 GB/s).** Nearly 3× wider than the L4, the memory
   subsystem is tightly integrated with the MXU and the workload remains
   *compute-bound* rather than *memory-bound* across all batch sizes tested.
3. **XLA graph compilation.** `torch_xla` traces the forward/backward/step graph
   once and compiles it to optimised XLA HLO. The compiled kernel has minimal
   per-step dispatch overhead regardless of batch size.
4. **Under-utilisation at small batches.** The MXU tiles and XLA compile path
   require a minimum amount of work to become efficient — hence the TPU losing to
   the GPU at batch ≤ 16.

---

## Key Takeaways

- **GPU throughput is memory-bandwidth-limited.** The NVIDIA L4 peaks at
  batch=32 (~2,895 samples/sec) and gains nothing beyond that — adding more
  work just queues behind the memory bus. The curve is already near-flat at
  batch=16 (~2,866 samples/sec), confirming saturation sets in early.

- **TPU throughput is compute-limited and scales near-perfectly linearly.** The
  v5litepod MXU delivers 256× throughput gain from batch=4 to batch=1024,
  reaching 94,795 samples/sec — **37× faster than the L4 at the same batch size**.

- **The crossover is at batch=32.** Below that, the GPU's fast dispatch wins;
  above it, the TPU's systolic array dominates and the gap grows with every
  batch doubling.

- **Hardware choice depends entirely on operating regime.** For latency-sensitive,
  small-batch inference (batch ≤ 16) the GPU is competitive and simpler to
  operate. For high-throughput training or large-batch inference the TPU's
  architecture is categorically superior for this class of workload.
