# 🧬 Quantum Hodgkin-Huxley Language Model — Autonomous Research

A [Karpathy-style autonomous AI research](https://github.com/karpathy/autoresearch) setup designed to run on **RunPod with an H100 GPU**, with **Claude Code** as the autonomous research agent running on it.

The agent modifies training code, runs experiments (fixed 5-minute wall-clock budget each), evaluates results, keeps or discards changes, and repeats autonomously. The goal: drive a biophysically-inspired zero-attention language model as close as possible to transformer SOTA.

## 🔬 The Research Question

Can **Hodgkin-Huxley spiking neurons** — the same equations that describe real ion channel dynamics in biological neurons — replace attention entirely for language modeling?

The core architecture uses a multi-timescale HH state-space model with 4 receptor-type channels per dimension:

| Channel | Timescale | Biological Analogue |
|---------|-----------|---------------------|
| AMPA | ~1-3 tokens | Fast synaptic reactions |
| NMDA | ~30 tokens | Phrase-level memory |
| Calcium | ~150 tokens | Paragraph context |
| Neuromodulatory | ~600 tokens | Topic tracking |

All channels run full HH gating dynamics (alpha/beta rate functions, ion channel conductances) at biologically-motivated different speeds. Conv1d + SiLU output gate provide the nonlinearities that make SSMs work for language.

**SOTA target:** GPT baseline `val_bpb = 0.96` (optimized transformer with Flash Attention 3 + Muon optimizer, 126 experiments on the same hardware and time budget).

## 📁 Structure

```
PureQHH_LM_NeurIPS_def7.ipynb   — Standalone QHH-Transformer hybrid notebook (WikiText-103)
run_experiment                    — RunPod setup runbook

autoresearch/
  prepare.py                     — Data download, BPE tokenizer, dataloader, eval (fixed)
  train.py                       — Multi-Timescale HH-SSM model (agent modifies this)
  train_hybrid.py                — Hybrid HH-SSM + Attention variant
  train_original.py              — GPT baseline (nanochat-derived, Flash Attn 3 + Muon)
  program.md                     — Agent instructions and research directions
  analysis.ipynb                 — Experiment progress visualization
  pyproject.toml                 — Dependencies (PyTorch 2.9.1, CUDA 12.8)
```

## 🧪 Three Model Variants

- **`train.py`** — Pure HH-SSM. Zero attention. Multi-timescale Hodgkin-Huxley recurrence with Conv1d + SiLU gate (Mamba-style). Sequential O(T) per token.
- **`train_hybrid.py`** — Hybrid. Alternates HH-SSM layers with standard multi-head causal attention (pattern: HH, HH, Attn, ...). Inspired by Jamba/Griffin hybrids.
- **`train_original.py`** — GPT baseline. Full transformer with Flash Attention 3, RoPE, GQA, value embeddings, MuonAdamW optimizer, `torch.compile`.

## 🚀 Running on RunPod

```bash
# Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/autoresearch && uv sync

# Download data + train tokenizer (~2 min)
uv run prepare.py

# Autonomous research mode
curl -fsSL https://claude.ai/install.sh | sh
claude
# Prompt: "Read program.md and let's kick off a new experiment!"
```

## ⚙️ Key Constraints

- **fp32 required** for HH exponentials (overflow in bf16/fp16)
- **Fixed 5-min time budget** per experiment — makes results comparable regardless of architecture changes
- **Metric:** `val_bpb` (validation bits per byte) — lower is better, vocab-size-independent
- **Data:** climbmix-400b-shuffle (same as nanochat)

## 📚 Based On

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Autonomous research framework
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — GPT baseline training code
- Hodgkin & Huxley (1952) — Biophysical neuron dynamics
