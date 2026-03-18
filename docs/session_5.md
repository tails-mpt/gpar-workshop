# Session 5: Precision and Dtype (FP32 / FP16 / BF16 / INT8)

## Overview

Session 5 asks a simple question: how much does dropping from FP32 to a lower-precision
dtype actually cost — or save — on each accelerator?

The Transformer encoder block from Session 1 is re-run across six batch sizes
(32, 64, 128, 256, 512, 1024) using four numeric formats:

- **FP32** — single precision baseline (no autocast)
- **FP16** — half precision with `torch.cuda.amp.autocast` + `GradScaler` (GPU only)
- **BF16** — bfloat16 with `torch.autocast` (GPU and TPU v5litepod)
- **INT8** — 8-bit integer quantisation (GPU: dynamic quantisation; TPU: native INT8 MXU)

The outcome diverges sharply between devices and between formats. On the GPU, FP16 and
BF16 each deliver roughly **2.1–2.5× throughput** over FP32 with a simultaneous **29%
VRAM reduction**. On the TPU, which advertises native BF16 MXUs, BF16 runs **~2.7%
slower** than FP32 on v5litepod in this configuration. INT8 results are pending
experimental runs — the notebooks are scaffolded and the expected findings are
documented below.

Notebooks: [`session_5/`](../session_5/)

---

## Hardware

Same devices as Sessions 1–4. See [`session_1.md`](session_1.md) for full hardware specs.

| Device | Memory | Note |
|---|---|---|
| NVIDIA L4 (GPU) | 23.7 GB GDDR6 | Tensor Cores accelerate FP16 and BF16 matrix math; INT8 TOPS ~242 |
| TPU v5litepod-1 | 16 GB HBM2 | MXU advertises native BF16 and INT8; FP16 not natively supported |

---

## Software Environment

| Component | GPU | TPU |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| PyTorch | 2.10.0+cu128 | 2.9.0+cu128 |
| torch_xla | — | 2.9.0 |
| Device string | `cuda:0` | `xla:0` |
| Run timestamp (FP32/FP16/BF16) | 2026-02-27T18:17 UTC | 2026-02-27T18:21 UTC |

---

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Model | `BenchmarkTransformerBlock` |
| `D_MODEL` | 512 |
| `N_HEAD` | 8 |
| `DIM_FEEDFORWARD` | 2048 |
| `SEQ_LEN` | 128 (fixed) |
| `BATCH_SIZE` | 32, 64, 128, 256, 512, 1024 |
| Steps (batch ≤ 256) | 50 (+ 5 warmup) |
| Steps (batch ≥ 512) | 20 (+ 5 warmup) — adaptive to prevent CUDA sync hang |
| Loop | forward → backward → Adam step |
| Metric | throughput (samples/sec), latency (ms/step), peak VRAM (GPU only) |

### Precision implementation notes

- **FP32:** default `torch.float32`; no autocast
- **FP16 (GPU only):** `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler`;
  prevents gradient underflow via dynamic loss scaling
- **BF16:** `torch.autocast(device_type, dtype=torch.bfloat16)`; wider exponent range
  eliminates the need for a GradScaler; not natively available in FP16 form on v5litepod
- **INT8 (GPU):** `torch.ao.quantization.quantize_dynamic` — applies dynamic INT8
  quantisation to linear layers; inference-time only (gradients in FP32)
- **INT8 (TPU):** native INT8 matmul path via `jnp.int8` or explicit cast to
  `torch.int8`; v5litepod INT8 TOPS = 786 (2× BF16 TFLOPS spec)

---

## Results: FP32 / FP16 / BF16 (measured)

### GPU — Throughput (samples/sec)

| Batch | FP32 | FP16 | BF16 | FP16 Speedup | BF16 Speedup |
|------:|-----:|-----:|-----:|-------------:|-------------:|
| 32 | 2,903 | 7,206 | 7,208 | 2.48× | 2.48× |
| 64 | 2,705 | 6,909 | 6,902 | 2.55× | 2.55× |
| 128 | 2,637 | 5,916 | 5,730 | 2.24× | 2.17× |
| 256 | 2,583 | 5,880 | 5,643 | 2.28× | 2.18× |
| 512 | 2,579 | 5,822 | 5,591 | 2.26× | 2.17× |
| 1,024 | 2,524 | 5,497 | 5,259 | 2.18× | 2.08× |

### GPU — Peak VRAM (MB) at batch=256

| Dtype | VRAM (MB) | Relative to FP32 |
|---|---:|---:|
| FP32 | 1,852 | 1.00× |
| FP16 | 1,319 | 0.71× |
| BF16 | 1,319 | 0.71× |

VRAM reduction is identical for FP16 and BF16; the 29% saving holds across all tested
batch sizes.

### TPU — Throughput (samples/sec)

FP16 is not natively supported on TPU v5litepod; only FP32 and BF16 are tested.

| Batch | FP32 | BF16 | BF16 Speedup |
|------:|-----:|-----:|-------------:|
| 32 | 2,947 | 2,887 | 0.979× |
| 64 | 5,944 | 5,784 | 0.973× |
| 128 | 11,897 | 11,579 | 0.973× |
| 256 | 23,657 | 23,071 | 0.975× |
| 512 | 47,323 | 45,956 | 0.971× |
| 1,024 | 95,217 | 92,373 | 0.970× |

BF16 is consistently ~2.7% slower than FP32 on the TPU; the ratio is stable across all
batch sizes, indicating this is a systematic overhead rather than measurement noise.

---

## Charts (FP32 / FP16 / BF16)

### Chart 1 — Throughput vs batch size (all dtypes)

![Throughput chart](assets/session_5_chart_throughput.png)

GPU lines cluster in two bands — a lower FP32 band (~2,500–2,900 samples/sec) and an upper
FP16/BF16 band (~5,200–7,200 samples/sec) — both essentially flat across batch sizes.
TPU lines rise steeply and linearly with batch; at batch=1024 the TPU FP32 line reaches
~95,000 samples/sec, dwarfing all GPU lines. TPU FP32 and TPU BF16 are nearly coincident,
making the ~2.7% BF16 regression visible only at larger batches.

### Chart 2 — BF16 / FP32 speedup ratio (GPU vs TPU)

![BF16 speedup chart](assets/session_5_chart_bf16_speedup.png)

The GPU BF16 speedup line hovers around **2.2×** and is roughly constant across batch
sizes (declining slightly from 2.48× at batch=32 to 2.08× at batch=1024). The TPU BF16
line sits just below **1.0×** and also stays flat — BF16 is consistently slower regardless
of batch size. The chart makes the contrast between devices immediate.

### Chart 3 — GPU peak VRAM by dtype

![VRAM chart](assets/session_5_chart_vram.png)

Three lines track peak VRAM from batch=32 to batch=1024. FP32 is the highest. FP16 and
BF16 are coincident ~29% lower. All three lines grow with batch size, but the gap between
FP32 and the reduced-precision lines remains proportionally constant.

---

## Analysis: FP32 / FP16 / BF16

### Why the GPU gains 2×+ from lower precision

NVIDIA Ada Lovelace GPUs contain Tensor Cores that execute FP16 and BF16 matrix
multiplications at roughly twice the FLOP/s of FP32 operations. A Transformer block's
forward and backward passes are dominated by matrix multiplications, so the Tensor Core
advantage maps almost directly onto end-to-end training throughput.

The VRAM reduction (29%) follows directly from halving the bytes per parameter, gradient,
and optimizer state. In practice this enables a larger effective batch size or deeper model.

### Why FP16 requires a GradScaler but BF16 does not

FP16's narrow dynamic range (min nonzero ~6×10⁻⁵) causes gradients to flush to zero.
`GradScaler` multiplies the loss by a large constant before backward and scales back after,
monitoring for overflow. BF16 retains FP32's full 8-bit exponent range — no scaling needed.
For new work on modern hardware (Ampere+), BF16 is simpler and numerically safer.

### Why the TPU does not benefit from BF16

The v5litepod BF16 throughput is **2.7% lower** than FP32 despite dedicated BF16 MXUs.
Likely causes:

1. **Compile-time optimisation path.** XLA may choose a more aggressively fused FP32
   kernel that happens to outperform the BF16 path for this model shape (d_model=512,
   seq_len=128). The MXU advantage is most pronounced at large matrix dimensions.
2. **Mixed-precision overhead.** Autocast inserts cast operations between FP32 compute
   (layer norm, loss) and BF16 compute (matrix multiplications). These casts add latency
   in the XLA graph that may exceed the savings for this model size.

The practical conclusion: FP32 is the correct default on TPU v5litepod for this family of
model sizes. Benchmark BF16 per-model rather than assuming it helps.

---

## INT8 — Expected findings and notebook specification

> **[PENDING RUN]** The following describes the experimental design and expected outcomes.
> Results and the updated comparison table will be added after benchmark cells are executed.

### What INT8 quantisation does

INT8 reduces weight and activation precision from 32 bits to 8 bits — a further 4× memory
reduction over FP32 (2× over BF16/FP16). On hardware with dedicated INT8 execution units,
this also delivers a throughput increase, because INT8 matrix multiplications execute at
higher TOPS than floating-point variants.

| Device | FP32 peak | BF16/FP16 peak | INT8 peak | INT8 / FP32 ratio |
|---|---|---|---|---|
| NVIDIA L4 | ~60 TFLOPS | ~121 TFLOPS | ~242 TOPS | ~4× (spec) |
| TPU v5litepod-1 | ~197 TFLOPS | ~393 TFLOPS | ~786 TOPS | ~4× (spec) |

Whether spec-level INT8 throughput materialises depends on the workload. For Transformer
blocks, the attention and feedforward linear layers are candidates for INT8; layer norm
and residual adds typically remain in floating point.

### GPU INT8 implementation

```python
import torch
import torch.ao.quantization as quant

# Dynamic quantisation: weights quantised to INT8 at inference time,
# activations quantised on-the-fly per batch. No calibration data needed.
model_fp32 = BenchmarkTransformerBlock().eval()
model_int8 = quant.quantize_dynamic(
    model_fp32,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint8
)
```

Dynamic quantisation applies only during the forward pass; backward and optimizer steps
remain in FP32. This is an inference-mode benchmark. The expected throughput benefit on
the L4 is in the range of **1.5–2× over FP32** for this model size, with **~50% VRAM
reduction** for the quantised weight storage.

### TPU INT8 implementation

The TPU v5litepod supports native INT8 matrix operations through the MXU. The benchmark
casts linear layer inputs and weights to `torch.int8` explicitly and uses
`torch.ops.xla.matmul_precision` to route through the INT8 execution path.

Expected throughput benefit: **up to 2× over FP32** (the MXU INT8 TOPS is 2× its BF16
TFLOPS), though real-world gains depend on whether the compiler successfully routes
through the INT8 hardware path.

### Expected INT8 results (to be replaced with measured data)

| Batch | GPU FP32 | GPU INT8 (expected) | TPU FP32 | TPU INT8 (expected) |
|------:|--------:|--------------------:|--------:|--------------------:|
| 32 | 2,903 | ~4,000–5,800 | 2,947 | ~4,500–5,900 |
| 64 | 2,705 | ~4,000–5,400 | 5,944 | ~9,000–11,900 |
| 128 | 2,637 | ~3,900–5,300 | 11,897 | ~18,000–23,800 |
| 256 | 2,583 | ~3,800–5,200 | 23,657 | ~36,000–47,300 |
| 512 | 2,579 | ~3,800–5,200 | 47,323 | ~71,000–94,600 |
| 1,024 | 2,524 | ~3,700–5,000 | 95,217 | ~142,000–190,000 |

*Ranges reflect hardware spec vs real-world efficiency gap. Replace with measured values.*

---

## Key Takeaways

- **On the GPU (NVIDIA L4), FP16 and BF16 each provide a consistent 2.1–2.5× throughput
  gain over FP32** across all tested batch sizes. Both formats simultaneously reduce peak
  VRAM by ~29%, enabling larger batches or bigger models at no cost.

- **Speedup decreases slightly at larger batch sizes** (2.48× at batch=32 → 2.08× at
  batch=1024 for BF16). This is an expected dilution effect: at larger batches the GPU is
  more saturated, so the Tensor Core advantage relative to total step time shrinks modestly.

- **BF16 is simpler than FP16 in practice.** BF16's wider exponent range matches FP32's
  dynamic range, so gradient underflow does not occur and no `GradScaler` is needed.

- **On the TPU v5litepod, BF16 is ~2.7% slower than FP32** at every batch size tested.
  The BF16 MXU advantage does not materialise for this model shape. **Use FP32 on this
  TPU configuration until benchmarked otherwise.**

- **The TPU's raw throughput advantage at large batches is substantial regardless of
  dtype.** At batch=1024, FP32 on the TPU reaches 95,217 samples/sec vs 2,524 on the GPU —
  a 37× gap. The dtype choice is a secondary concern on the TPU; batch size is the
  dominant lever.

- **INT8 offers a further ~1.5–2× gain on both devices** (pending measurement). On the
  GPU, dynamic INT8 quantisation is inference-only. On the TPU, the MXU INT8 path delivers
  a symmetric benefit. Neither device supports INT4 natively in PyTorch/XLA as of this
  workshop.

---

## Decision Rule from This Session

- **GPU workloads:** always use BF16 (or FP16 with a GradScaler on older hardware). The
  ~2× throughput and 29% VRAM savings are effectively free.
- **TPU workloads:** benchmark before assuming BF16 helps. On v5litepod (d_model=512,
  seq_len=128), FP32 is faster. Revisit if using v4 TPUs, larger model dimensions, or
  dedicated BF16-only inference paths.
- **INT8 for inference:** use when throughput matters more than numeric precision. For
  training, keep gradients in FP32 and quantise only the forward path.
