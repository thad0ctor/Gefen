# [Gefen: Optimized Stochastic Optimizer](https://arxiv.org/pdf/2606.13894)

Gefen is a drop-in replacement for the AdamW optimizer for memory-efficient
training. It keeps the familiar AdamW training recipe while dramatically
reducing optimizer-state memory: an 8x reduction in AdamW memory footprint, or
about 6.5 GiB saved per billion parameters, while maintaining AdamW-level
performance. The reduced memory footprint lets you train larger models or use
larger batch sizes and, as a result, achieve higher training throughput.
All it takes is changing two lines of code: import Gefen and replace the AdamW
optimizer constructor.

## Installation

Install from PyPI:

```bash
pip install gefen
```

Or install from source:

```bash
git clone https://github.com/ndvbd/Gefen
cd Gefen
pip install -e .
```

On the first CUDA run, Gefen builds its fused CUDA kernels with PyTorch JIT and
`nvcc`. This can take a few minutes. Later runs reuse the cached build for the
same Python, PyTorch, CUDA version, and Gefen source checkout.

This keeps the source install lightweight, but it requires a CUDA toolkit and
host compiler compatible with your PyTorch installation. In the future, we plan
to make this smoother with prebuilt wheels for common PyTorch/CUDA combinations.

## Quick Start

```python
import torch
from gefen import Gefen

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(128, 10).to(device)

# optimizer = torch.optim.AdamW(
optimizer = Gefen(  # Replace AdamW with Gefen:
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)

inputs = torch.randn(32, 128, device=device)
targets = torch.randint(0, 10, (32,), device=device)

logits = model(inputs)
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)

print('Finished successfully.')
```

### Learning rate (when porting an AdamW config)

Gefen matches AdamW's *interface*, but it needs a **lower learning rate** —
roughly **0.6× AdamW's** in our testing. **At its own optimal LR, Gefen matches
AdamW's loss and run-to-run reproducibility**, while keeping its optimizer-memory
advantage; reused at AdamW's LR unchanged, it over-steps.

Why: Gefen's second moment is a *block-shared* RMS of the gradient rather than
AdamW's per-element `√v`. Globally the two take similar-magnitude steps, but on a
few high-leverage tensors — most sharply the RMSNorm weights (`q_norm`/`k_norm`,
whose length equals the head dim, so the whole tensor is a *single shared block*) —
Gefen over-steps by ~1.5×. Those tensors set the stability ceiling, so the usable
LR is ~0.6× AdamW's. This is most pronounced on modern decoder architectures
(SwiGLU MLP + grouped-query attention, e.g. Qwen3).

Measured on Qwen3-4B full fine-tune (each optimizer at its own optimum, then the
naive drop-in):

| Learning rate | Gefen vs AdamW | run-to-run spread |
|---|---|---|
| ~0.6× AdamW's (Gefen's optimum) | **matches AdamW** (ties at 300 and 800 steps) | tight (≈ AdamW) |
| 1.0× AdamW's, but AdamW-LR too hot* | ~0.3–0.4 higher | ~5× AdamW |

\* Both optimizers preferred a much lower LR than a typical AdamW default in this
regime, so "1.0× AdamW's LR" above means an LR that is itself near AdamW's edge;
the gap is the *extra* over-step Gefen incurs there.

When porting an AdamW recipe, **scale the learning rate to ~0.6× and tune from
there.** An LR comfortable for AdamW can sit past Gefen's stability edge, and Gefen
does *not* tolerate raising the LR back up. The exact factor is set by the
architecture's RMSNorm/block structure, so confirm on your own model — the simplest
check is to compute, on a warmed AdamW state, `‖m̂/(√v̂+ε)‖ / ‖m̂/(√blockmean(v̂)+ε)‖`
for the norm-weight tensors; that ratio is the LR scale factor.

## Hugging Face Trainer

Until native `optim="gefen"` support is released in Transformers, pass Gefen to
the Trainer with `optimizer_cls_and_kwargs`:

```python
from gefen import Gefen
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=1e-3,
    weight_decay=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizer_cls_and_kwargs=(
        Gefen,
        {
            "lr": training_args.learning_rate,
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
            "fused": True,
        },
    ),
)
```

### Distributed Training

Gefen is fully compatible with standard distributed training setups, including PyTorch DDP, PyTorch FSDP (including FSDP2 with `fully_shard`), and all flavors of DeepSpeed ZeRO. Gefen can be used like any other PyTorch optimizer in these workflows, with either `fused=True` or `fused=False`.


### Extension: Gefen-Muon

Based on the Gefen paradigm, a simple extension is to add a pseudo-orthogonalization step on the first moment, as Muon does, while skipping the second moment. This version, which is based on the PyTorch Muon implementation, immediately reduces Muon's optimizer-state footprint by 4x: the first moments are quantized to 8-bit using Gefen's Hessian-block-diagonal-inspired partitioning exact quantization, while performance remains similar to Muon.

You can use it exactly as you use Muon, with a simple constructor name replacement:

```python
from gefen import GefenMuon

optimizer = GefenMuon(
    [muon_parameter for _, muon_parameter in muon_parameter_pairs],
    lr=lr,
)
```

Our experiments show similar performance to Muon, with x4 less optimizer memory.

## Testimonials

Have you tried Gefen and want to report your impressions privately or publicly?
We would be happy to hear about your experience. With your permission, we can
credit you and mention your work here.


## Citation

If you found this library useful, please consider citing our work:

```bibtex
@article{benedek2026gefen,
  title={Gefen: Optimized Stochastic Optimizer},
  author={Benedek, Nadav and Koren, Tomer and Fried, Ohad},
  journal={arXiv preprint arXiv:2606.13894},
  year={2026}
}
```

