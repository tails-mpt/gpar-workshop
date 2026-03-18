# Workshop Introduction: GPU vs TPU for Transformer Workloads

## What this workshop is

This workshop benchmarks a single Transformer encoder block systematically across two
cloud accelerators — an NVIDIA L4 GPU and a Google TPU v5e — to answer a practical
question: **when should you pick one over the other, and by how much does it matter?**

The benchmark is intentionally small. One encoder block is a controlled proxy: it
isolates the hardware and software variables without the noise of a full model pipeline.
The findings scale to real workloads — the same memory-bandwidth ceiling, the same
compilation model, the same precision trade-offs apply at BERT-base or GPT-2 scale.

---

## Why these two accelerators?

### The hardware

| Property | NVIDIA L4 (GPU) | Google TPU v5e (v5litepod-1) |
|---|---|---|
| Architecture | Ada Lovelace (2022) | Systolic array MXU |
| Memory | 23.7 GB GDDR6 | 16 GB HBM2 |
| Memory bandwidth | ~300 GB/s | ~819 GB/s |
| Peak compute (BF16) | ~121 TFLOPS | ~393 TFLOPS |
| Peak compute (INT8) | ~242 TOPS | ~786 TOPS |
| TDP | 72 W | — |
| GCP hourly rate (us-central1, early 2026) | ~$0.70 / hr | ~$1.20 / hr |

On paper the TPU has a 3.25× compute advantage and a 2.7× memory bandwidth advantage —
for a 1.71× price premium. This asymmetry is not a problem with the comparison; it is the
central question of the workshop. The goal is to find out under which conditions that
hardware advantage actually translates into cheaper, faster training.

### Why the L4, not an A100?

The L4 is GCP's most accessible modern GPU for training workloads. The next step up —
an A100 — is roughly 3–5× more expensive per hour and places the comparison in a
different budget tier. The L4 is what a cost-conscious practitioner reaches for first.

### Why the TPU v5e, not a cheaper option?

The TPU v5e single-chip (v5litepod-1) is the cheapest TPU available on GCP. There is no
sub-$1.20/hr TPU option. Older generations (v2, v3) are being deprecated. The v5e is the
realistic entry point.

### The comparison is honest

The L4 is inference-optimised — lower power, lower TFLOPS than a training GPU of the same
generation. That is acknowledged throughout the workshop. The comparison is not "best GPU
vs best TPU" — it is "the GPU you would most likely spin up on GCP vs the TPU you would
most likely spin up on GCP, at their listed prices."

The right comparison metric is not raw throughput but **samples per dollar**. The workshop
keeps both in view.

---

## The economic context

The cost analysis established the
following using FP32 training throughput from Session 1:

| Batch | GPU (samples/$) | TPU (samples/$) | TPU cheaper? |
|------:|----------------:|----------------:|:---:|
| 4 | 7.1M | 1.1M | No |
| 8 | 13.0M | 2.2M | No |
| 16 | 14.7M | 4.5M | No |
| 32 | 14.9M | 9.0M | No |
| **64** | **14.2M** | **17.9M** | **Yes — 1.26×** |
| 128 | 13.8M | 35.7M | Yes — 2.6× |
| 256 | 13.6M | 71.6M | Yes — 5.3× |
| 512 | 13.4M | 142.7M | Yes — 10.6× |
| 1,024 | 13.2M | 284.4M | Yes — **21.5×** |

*Based on early-2026 GCP on-demand pricing ($0.70/hr GPU, $1.20/hr TPU). Substitute
current rates from the [GCP pricing page](https://cloud.google.com/pricing) before
drawing conclusions.*

**The cost crossover is at batch=64.** Below that, the GPU's lower hourly rate wins
despite lower throughput. Above it, the TPU's throughput advantage outpaces its 1.71×
price premium — and the gap grows steeply with every batch doubling.

![Cost-performance chart](assets/intro_chart_cost.png)

---

## The model

All sessions benchmark `BenchmarkTransformerBlock` — a single Transformer encoder block
defined in `transformer_block.py` at the workshop root.

```
Input: [batch, seq_len, d_model]

 MultiHeadSelfAttention (8 heads, d_model=512)
 └─ Residual + LayerNorm  (Post-LN)
 FeedForward: Linear(512→2048) + GELU + Linear(2048→512)
 └─ Residual + LayerNorm  (Post-LN)

Output: [batch, seq_len, d_model]
```

| Parameter | Value |
|---|---|
| `D_MODEL` | 512 |
| `N_HEAD` | 8 |
| `DIM_FEEDFORWARD` | 2048 |
| `SEQ_LEN` | 128 tokens (default) |
| Benchmark loop | forward → backward → Adam step |
| Warmup steps | 5 |
| Metric | throughput (samples / sec) |

This is deliberately a *minimal* model — one block, not a stack. It isolates the hardware
response without the confounding effects of depth, residual skip connections across layers,
or cross-layer memory reuse. Session 3 in the workshop scales this to 24 layers.

---

## Session outline

| Session | Title | What it answers |
|---|---|---|
| [Session 1](session_1.md) | GPU vs TPU Throughput Scaling | At what batch size does the TPU overtake the GPU, and how large does the gap get? |
| [Session 2](session_2.md) | Sequence Length Scaling | How does GPU throughput degrade as sequences lengthen, and why is the TPU immune? |
| [Session 3](session_3.md) | Model Depth and Memory Limits | How does throughput and VRAM scale from 1 to 24 layers? How much headroom does each device have before it runs out of memory? |
| [Session 4](session_4.md) | Static vs Dynamic Graphs | What happens to TPU performance when a model uses data-dependent control flow? How bad is it, and can it be fixed? |
| [Session 5](session_5.md) | Precision and Dtype | How much do FP16, BF16, and INT8 actually improve throughput and reduce memory on each device? |
| [Session 6](session_6.md) | Framework Interoperability | Does it matter whether you use JAX/Flax or PyTorch/XLA? How do two frameworks that both compile to XLA differ in practice? |
