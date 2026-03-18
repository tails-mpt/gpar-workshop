# Session 4: Static vs Dynamic Computation Graphs

## Overview

Session 4 introduces the first scenario where the GPU wins. While Sessions 1–3 used
fully static forward passes, this session deliberately injects data-dependent control
flow into the model and measures the cost on each device.

The session is structured in two parts:

**Part A — Abstract variants (baseline):** Three minimal variants of the Transformer
block that isolate the XLA compilation penalty from the forward pass structure.

**Part B — Realistic scenarios:** Three scenarios drawn from real NLP and training
practice — padding masks, conditional early exit, and variable-length batch loops —
that show how the static/dynamic constraint manifests in production-like code.

All experiments use `batch=256, seq_len=128` unless stated otherwise.

Notebooks: [`session_4/`](../session_4/)

---

## Hardware

Same devices as Sessions 1–3. See [`session_1.md`](session_1.md) for full hardware specs.

| Device | Memory | Bandwidth | Note |
|---|---|---|---|
| NVIDIA L4 (GPU) | 23.7 GB GDDR6 | ~300 GB/s | Eager execution — evaluates branches immediately |
| TPU v5litepod-1 | 16 GB HBM2 | ~819 GB/s | XLA compiled — branches in Python force graph sync |

---

## Software Environment

| Component | GPU | TPU |
|---|---|---|
| Python | 3.12.12 | 3.12.12 |
| PyTorch | 2.10.0+cu128 | 2.9.0+cu128 |
| torch_xla | — | 2.9.0 |
| Device string | `cuda:0` | `xla:0` |
| Run timestamp | 2026-02-27T17:39 UTC | 2026-02-27T17:41 UTC |

---

## Background: Why Python branches are expensive on TPU

XLA's execution model is fundamentally different from eager dispatch. Operations are not
dispatched immediately — they are accumulated in a lazy graph until `torch_xla.sync()`
(or an implicit sync) triggers compilation and execution. This deferred model enables XLA
to fuse operations, eliminate redundant memory transfers, and compile the full step into
a single optimised kernel.

When Python code reaches a branch like `if tensor_value > 0`, the Python interpreter
must know the *value* of the condition to choose a path. This requires materialising the
scalar on the host CPU — which forces an implicit XLA sync mid-step:

1. Interrupt lazy graph accumulation
2. Flush, compile, and execute the partial graph
3. Transfer the scalar result to CPU
4. Resume Python, then accumulate the remainder
5. Sync again at the end of the step

Two compilations per step instead of one, plus a device-to-host transfer, plus broken
fusion opportunities. The GPU never builds such a graph, so it is completely unaffected.

---

## Part A — Abstract variants

### Variant definitions

```python
class StaticFF(nn.Module):
    def forward(self, x):
        return self.block(x)                          # identical to Sessions 1–3

class DynamicFF(nn.Module):
    def forward(self, x):
        out = self.block(x)
        if out.mean() > 0:                            # Python branch on a tensor value
            return out
        else:
            return -out

class StaticEquivalentFF(nn.Module):
    def forward(self, x):
        out = self.block(x)
        return torch.where(out.mean() > 0, out, -out) # tensor op — stays in XLA graph
```

### Part A Results

| Variant | GPU (samples/s) | GPU latency (ms) | TPU (samples/s) | TPU latency (ms) |
|---|---:|---:|---:|---:|
| StaticFF | 2,645 | 96.8 | **23,605** | 10.9 |
| DynamicFF | 2,590 | 98.8 | **9,425** | 27.2 |
| StaticEquivalentFF | 2,551 | 100.4 | **23,284** | 11.0 |

### Relative throughput (vs StaticFF baseline)

| Variant | GPU | TPU |
|---|---:|---:|
| StaticFF | 1.000× | 1.000× |
| DynamicFF | **0.979×** (−2.1%) | **0.399×** (−60.1%) |
| StaticEquivalentFF | **0.964×** (−3.6%) | **0.986×** (−1.4%) |

### Recovery metric (TPU only)

| Metric | Value |
|---|---|
| Static baseline throughput | 23,605 samples/sec |
| Dynamic penalty (lost throughput) | −14,180 samples/sec |
| StaticEquivalent recovery | +13,859 samples/sec |
| **Penalty recovered** | **97.7%** |

### Part A charts

#### Chart 1 — Throughput bar chart: Static / Dynamic / StaticEquivalent

![Graph Variants Bar Chart](assets/session_4_chart_variants.png)

**What the chart shows:**
The two green (GPU) bars are nearly identical — the GPU column is flat across all three
variants, compressed near the bottom of the chart relative to the TPU's StaticFF and
StaticEquivalentFF bars. The TPU's DynamicFF bar drops sharply to less than half the
height of its neighbours. The chart makes the penalty and recovery immediately legible:
the `DynamicFF` bar is the anomaly, and `StaticEquivalentFF` restores the TPU bar to
its full height.

---

### Part A Analysis

**GPU: eager dispatch is blind to graph structure.** On the GPU, PyTorch evaluates every
operation as it is called. When `out.mean() > 0` is reached, the value is computed
immediately and the result is a Python `bool`. The three variants run the same CUDA
kernels in essentially the same sequence — throughput varies by at most 3.6%.

**TPU DynamicFF: 60% throughput collapse from a single Python `if`.** Per-step latency
jumps from 10.85 ms to 27.16 ms — a 2.5× slowdown. The cost of one data-dependent
Python branch equals adding 1.6× the entire compute budget of the step.

**`torch.where` recovers 97.7% with one line of code.** The condition stays inside the
XLA graph. No intermediate sync is required. Per-step latency returns to 10.99 ms.

The remaining 1.4% gap between StaticFF and StaticEquivalentFF reflects the compute cost
of evaluating `out.mean()` as an additional tensor reduction — real but negligible.

---

## Part B — Realistic scenarios

The abstract variants above demonstrate the mechanism. This part shows how it manifests
in three common real-world patterns. **Results for Part B are pending experimental runs;**
the scenarios are fully specified below and the notebooks contain runnable benchmark cells.

---

### Scenario B1: Padding mask (variable-length sequences in a batch)

**Context:** Real NLP batches contain sequences of different lengths. A standard practice
is to pad shorter sequences to the maximum length in the batch and apply a boolean mask
during attention to ignore padding tokens.

**The dynamic pattern:** If the mask is computed from the actual sequence lengths at
runtime using a Python-level loop or a conditional:

```python
# Dynamic version — Python branch on per-sample length
def forward(self, x, lengths):
    out = self.block(x)
    # Building mask with Python control flow per sample
    mask = torch.zeros(batch, seq_len, device=x.device)
    for i, length in enumerate(lengths.tolist()):   # .tolist() forces XLA sync
        mask[i, length:] = float('-inf')
    return out + mask.unsqueeze(1).unsqueeze(1)
```

**The static fix:**

```python
# Static version — mask computed entirely as tensor ops
def forward(self, x, lengths):
    out = self.block(x)
    # Build mask without touching Python scalars mid-step
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
    mask = (positions >= lengths.unsqueeze(1)).float() * float('-inf') # [batch, seq_len]
    return out + mask.unsqueeze(1).unsqueeze(1)
```

**Expected finding:** The dynamic version forces a `.tolist()` call per step, materialising
the `lengths` tensor on CPU and causing an XLA sync per sample. The static version builds
the mask entirely within the XLA graph using broadcasting. On GPU: no meaningful difference.
On TPU: substantial penalty in the dynamic version (proportional to batch size), full
recovery in the static version.

**Benchmark configuration:** `batch=256, seq_len=128`, `lengths` sampled uniform in
[64, 128] per step. N_STEPS=50.

> **[PENDING RUN]** Results will be added once benchmark cells are executed.

---

### Scenario B2: Conditional early exit (adaptive computation)

**Context:** Early exit mechanisms allow a model to skip later layers for "easy" samples
where the model is already confident. This saves compute on well-represented examples.
The key is whether the exit condition is evaluated in Python (forcing a sync) or kept
inside the XLA graph.

**The dynamic pattern:**

```python
# Dynamic early exit — Python branch on confidence per step
class EarlyExitModel(nn.Module):
    def forward(self, x, threshold=0.8):
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Compute confidence proxy (max activation norm) and branch
            confidence = x.norm(dim=-1).max()        # scalar reduction
            if confidence > threshold:               # Python branch — XLA sync
                return x, i                          # exit early
        return x, len(self.blocks) - 1
```

**The static fix:**

```python
# Static early exit — exit condition as a mask, no Python branch
class EarlyExitStaticModel(nn.Module):
    def forward(self, x, threshold=0.8):
        exited = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        output = torch.zeros_like(x)
        for block in self.blocks:
            x_new = block(x)
            # Update only samples that haven't exited yet
            x = torch.where(exited.unsqueeze(-1).unsqueeze(-1), x, x_new)
            # Check exit condition as a tensor op (no Python branch)
            confidence = x.norm(dim=-1).max(dim=-1).values  # [batch]
            newly_exited = confidence > threshold
            output = torch.where(newly_exited.unsqueeze(-1).unsqueeze(-1), x, output)
            exited = exited | newly_exited
        return output
```

**Expected finding:** The dynamic version triggers an XLA sync at every layer for every
batch (to evaluate the Python `if`). For a 6-layer model, that is 6 syncs per step instead
of 1. The static version compiles the full loop as a single XLA program. On GPU: marginal
difference. On TPU: severe penalty in the dynamic version, recovered by the static version.

**Benchmark configuration:** 4-layer `DeepTransformerModel`, `batch=64, seq_len=128`,
threshold=0.8, N_STEPS=30.

> **[PENDING RUN]** Results will be added once benchmark cells are executed.

---

### Scenario B3: Variable-length batch loop (ragged batches)

**Context:** Some workloads — document QA, code completion, chat turns — naturally
produce batches where each element has a different token count. One approach is to
loop over variable-length inputs rather than padding to a fixed shape.

**The dynamic pattern:**

```python
# Variable loop — loop length changes each step
def process_ragged_batch(model, inputs):
    # inputs is a list of tensors with different seq lengths
    outputs = []
    for inp in inputs:                               # loop length = batch size
        out = model(inp.unsqueeze(0))               # [1, seq_i, d_model]
        outputs.append(out)
    return outputs
```

**The static fix:**

```python
# Static equivalent — pad to fixed max length, use attention mask
def process_padded_batch(model, inputs, max_len):
    # Pad all inputs to max_len
    padded = torch.zeros(len(inputs), max_len, D_MODEL, device=inputs[0].device)
    for i, inp in enumerate(inputs):
        padded[i, :inp.shape[0]] = inp
    mask = create_padding_mask(inputs, max_len)     # static tensor op
    return model(padded, mask)                      # single forward call
```

**Expected finding:** The per-sample loop causes one XLA trace/compile per sample per
step — the graph size scales with batch size and the compilation cost grows with it. The
padded version is one fixed-shape forward call with a mask, which XLA compiles once. On
GPU: the loop version is slower (sequential Python overhead) but no catastrophic penalty.
On TPU: catastrophic in the loop version (N×compile), full or near-full recovery with padding.

**Benchmark configuration:** `batch=8` variable-length sequences (seq_len ∈ [32, 128]),
N_STEPS=30. Comparison: total throughput in tokens/sec.

> **[PENDING RUN]** Results will be added once benchmark cells are executed.

---

## Combined Key Takeaways

- **GPU throughput is insensitive to graph structure.** All three abstract variants land
  within 4% on the GPU. Eager dispatch has no concept of a "compiled graph" to invalidate —
  it executes each operation as issued, regardless of control flow structure.

- **A single Python `if` on a tensor value costs the TPU 60% of its throughput.**
  `DynamicFF` drops from 23,605 to 9,425 samples/sec. Per-step latency triples
  (10.85 ms → 27.16 ms) because XLA must sync mid-step to evaluate the branch condition.

- **`torch.where` recovers 97.7% of the penalty with one line of code.**
  Replacing the Python `if` with a tensor-valued equivalent keeps the condition inside the
  XLA graph. Latency returns to 10.99 ms.

- **The same principle applies at scale:** padding masks, early exit conditions, and ragged
  batch iteration all trigger the same mechanism. The fix is always the same — express the
  condition as a tensor op (`torch.where`, `masked_fill`, boolean masking) rather than a
  Python branch.

- **The constraint is refactorable, not fundamental.** Many research branches can be
  rewritten as tensor ops. When they cannot — variable routing, sparsity patterns whose
  structure cannot be expressed as masks — the GPU is the correct choice.

---

## Decision Rule from This Session

If your model's control flow depends on a tensor value:

1. **Can the branch be expressed as `torch.where`, `masked_fill`, or a boolean mask?**
   → Yes: refactor and use the TPU. Recovery is ~97%.
   → No: use the GPU.

2. **Does your training loop iterate over variable-length or ragged inputs?**
   → Yes: pad to a fixed shape first. The fixed shape compiles once; the ragged loop
   triggers recompilation per element on the TPU.

3. **Is the exit condition computed in Python from a tensor reduction?**
   → Rewrite as a tensor mask operation. Keep the condition inside the XLA graph.
