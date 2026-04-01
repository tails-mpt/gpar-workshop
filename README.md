# GPU vs TPU for Transformer Workloads

A hands-on benchmark workshop that pits an NVIDIA L4 GPU against a Google TPU v5e
on a single Transformer encoder block. The goal is to figure out when each device
actually wins — not on paper, but measured end-to-end on GCP at real prices.

## The short version

The TPU is 3.25x faster on paper and 1.71x more expensive per hour. So when does it
actually pay off? **At batch size 64.** Below that, the GPU's lower hourly rate wins.
Above it, the TPU pulls ahead fast — by batch 1024 it delivers 21x more samples per
dollar.

## What's in here

Six sessions, each with Jupyter notebooks you can run on GCP, saved results, and
written findings:

| Session | Question |
|---|---|
| **1. Throughput Scaling** | At what batch size does the TPU overtake the GPU? |
| **2. Sequence Length** | Does the TPU's bandwidth advantage compound as sequences get longer? |
| **3. Model Depth** | How far can you scale layers before each device runs out of memory? |
| **4. Static vs Dynamic Graphs** | What happens to TPU performance with data-dependent control flow? |
| **5. Precision / Dtype** | How much do FP16, BF16, and INT8 actually help on each device? |
| **6. Framework Interop** | Does it matter if you use JAX/Flax or PyTorch/XLA? |

## Repo layout

```
session_1/ through session_6/   Notebooks + saved result JSONs
docs/                           Written findings per session + charts
transformer_block.py            The shared model used across all sessions
```

## Hardware

| | NVIDIA L4 | TPU v5e (v5litepod-1) |
|---|---|---|
| Memory | 23.7 GB GDDR6 | 16 GB HBM2 |
| Bandwidth | ~300 GB/s | ~819 GB/s |
| BF16 compute | ~121 TFLOPS | ~393 TFLOPS |
| GCP price (us-central1) | ~$0.70/hr | ~$1.20/hr |

The L4 is inference-optimised. The comparison is intentionally "the GPU you'd most
likely spin up" vs "the cheapest TPU available" — not best-vs-best, but realistic-vs-realistic.

## Running the notebooks

You'll need a GCP instance with either an L4 GPU or a TPU v5e attached.
The GPU notebooks use PyTorch; the TPU notebooks use PyTorch/XLA (sessions 1-5)
and JAX/Flax (session 6).

Each session folder has numbered notebooks — run them in order. Results are saved
to `results/` as JSON so the analysis notebook can run without re-benchmarking.

## The model

All sessions benchmark the same single Transformer encoder block: 8-head self-attention,
d_model=512, FFN dim 2048, post-LN residuals. Defined in `transformer_block.py`.
Session 3 stacks it to 24 layers to test depth scaling.
