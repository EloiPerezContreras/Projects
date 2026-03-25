# Autoresearch: Quantum Hodgkin-Huxley Language Model

## Context

This is **not** a standard GPT/Transformer training script. `train.py` implements a **Quantum Hodgkin-Huxley (QHH) Language Model** — a biophysically-inspired architecture where:

- **Attention is replaced** by recurrent spiking neuron dynamics. There are zero attention heads.
- Each "layer" contains a `QuantumHHNeuron` that maintains membrane voltage state across the sequence.
- Ion channel gating (Na⁺, K⁺, leak) follows the Hodgkin-Huxley equations with **Lindblad master equation** quantum corrections.
- Sequence memory = neuron state propagation (biologically plausible, not learned position-wise attention).

The primary metric is `val_bpb` (lower is better). Secondary metrics logged per run: `spike_rate`, `coherence`, and per-layer `layer_N_regime` (quantum-coherent / weak-quantum / near-classical). These are printed in `run.log` — record them in `results.tsv` alongside `val_bpb`.

## Tier 0 — The SOTA Target

**The ultimate goal is to match or close the gap with transformer SOTA on the same hardware and time budget.**

Before starting the QHH loop, the human will run the original GPT-based `train_original.py` once and record the transformer baseline:

```
GPT_BASELINE_BPB = 0.96  (optimized GPT on H100, 126 experiments)
```

**Your job as agent**: drive QHH `val_bpb` as close as possible to 0.96. Every experiment should be evaluated against this target. When recording results, note the gap: `gap = qhh_val_bpb - 0.96`.

**Why this is hard**: The QHH neuron is sequential (O(T) per token, no parallelism across time) and requires fp32 for HH exponentials. In 5 minutes, a GPT with flash attention processes 20-50× more tokens. To close the gap you likely need:
1. Aggressive throughput optimization (shorter sequences, fewer substeps, smaller model that trains faster)
2. Architectural simplifications that reduce cost per token without destroying capacity
3. Possibly radical changes (simpler neuron models, hybrid approaches)

**This is Phase 1 (discovery)**. The human will take the best configuration you find and run it with extended time budget for proper training. So optimize for the architecture that gives the best val_bpb *trajectory* in 5 minutes — a model that's clearly still improving at the 5-min mark is more valuable than one that plateaus early.

**Bio metrics matter for science**: If removing Lindblad quantum terms gives equal val_bpb → record that, it's a real negative result. If coherence correlates with val_bpb improvement → note it, it's evidence quantum dynamics help. Track these in results.tsv:
```
tag	val_bpb	peak_vram_mb	spike_rate	coherence	gap_to_gpt	notes
```

## Setup

1. Create the branch: `git checkout -b autoresearch/<tag>` from current master.
2. Read the in-scope files:
   - `README.md` — repository context
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. QHH model architecture, optimizer, training loop.
3. Verify data: `~/.cache/autoresearch/` should contain data shards and tokenizer. If not, tell the human to run `uv run prepare.py`.
4. Initialize `results.tsv` with header: `tag	val_bpb	peak_vram_mb	spike_rate	coherence	gap_to_gpt	notes`
5. Confirm and go.

## Experiment protocol

Each experiment runs on a single GPU with a fixed 5-minute wall-clock budget.

1. Tune `train.py` with an experimental idea by directly editing the code.
2. Run: `uv run train.py > run.log 2>&1`
3. Read results: `grep "^val_bpb:\|^peak_vram_mb:\|^spike_rate:\|^coherence:" run.log`
   - If empty → crashed. `tail -n 50 run.log` for traceback. Attempt fix (max 3 tries, then give up).
4. Record in `results.tsv` (untracked by git). Compute `gap_to_gpt = val_bpb - 0.96`.
5. If val_bpb improved → keep commit. If not → `git reset --hard` to previous state.

## Research directions (prioritized)

### Tier 1: Throughput — more training steps in 5 minutes (try first)
- **TRAIN_SEQ**: Currently 256. Try 128 (faster iteration, more steps in 5 min) or 64. The QHH neuron is O(T) sequential — halving seq len roughly doubles throughput.
- **Batch size**: Currently 32. Try 16 (less VRAM, more steps) or 64 (smoother gradients, fewer steps).
- **N_SUBSTEPS**: Currently 2. Try 1 (half the compute per token, cruder dynamics). This is the single biggest throughput lever.
- **DETACH_EVERY**: Currently 32 (truncated BPTT window). Try 16 (faster, less memory) or 8.
- **D_MODEL / DEPTH tradeoff**: Smaller models train faster. Try d=128/depth=2 vs d=256/depth=4 vs d=384/depth=3.

### Tier 2: Learning dynamics
- **Learning rate**: Currently 1e-3. Try 3e-3, 5e-4, 2e-3. The bio params get 0.5× multiplier.
- **BIO_LR_MULT**: Currently 0.5. Try 0.1 (more conservative bio params) or 1.0 (same as other params).
- **Grad clip**: Currently 3.0. QHH gradients can be spiky. Try 1.0 or 5.0.
- **Warmup**: Currently 100 steps. Try 50 or 200.
- **Dropout**: Currently 0.1. With 5-min training, dropout may hurt. Try 0.0 or 0.05.

### Tier 3: Architecture tweaks
- **Projection expansion ratio**: `proj` in QHHLayer uses 2× expansion. Try 4× (transformer-like) or 1× (minimal).
- **Remove the `mix` linear**: The cross-dimension mixing in the neuron may be redundant. Try removing it.
- **Output voltage normalization**: Currently `(V+100)/160*2-1`. Try just LayerNorm on raw voltage.
- **Add output gate**: Sigmoid gate on the voltage output (like LSTM's output gate).
- **Positional encoding**: Try learned positional embeddings instead of sinusoidal.

### Tier 4: Biophysics modifications
- **DT**: Currently 0.5. Try 1.0 (bigger steps, possibly unstable) or 0.25 (smoother but slower).
- **Conductance initialization**: `log_gNa`, `log_gK`, `log_gL` start at textbook values (120, 36, 0.3). Try different initializations.
- **Quantum parameters**: `omega_raw=1.0`, `eps_0=0.1`, `gamma=0.5` control the quantum-classical boundary. Try gamma=5.0 (more classical), or omega=5.0 (stronger quantum oscillations).
- **Spike threshold**: Currently -20.0 mV. Try -30 or -10.
- **Reset potential**: Currently -65.0 mV. Try -70 or -55.
- **Surrogate gradient**: Currently 1/(1+5|x|)². Try sigma=10 (sharper), sigma=2 (softer).

### Tier 5: Radical changes (high risk, high reward)
- **Remove Lindblad entirely**: Make the HH neuron fully classical. If val_bpb is equal or better → the quantum part is hurting (important scientific finding!). Record coherence=0 for comparison.
- **Replace HH with LIF neuron**: Leaky integrate-and-fire with learned time constant. Much cheaper per step → dramatically more training steps in 5 min. May close the GPT gap through sheer throughput.
- **Hybrid**: QHH for first 1-2 layers (captures biophysics), simple LIF or linear RNN for upper layers (fast).
- **Muon optimizer**: Replace AdamW with the Muon optimizer from nanochat. May help with the non-standard loss landscape.
- **Remove spiking entirely**: Make it a continuous-rate HH model. The spike_fn and reset add complexity — maybe continuous dynamics work better for language.

## Key constraints

- **fp32 requirement**: HH exponentials (`exp(-(V+65)/18)` etc.) OVERFLOW in bf16/fp16. The neuron forward pass must stay in fp32 via `autocast(enabled=False)`. Do NOT remove this.
- **VRAM**: Recurrent state is 10 tensors of [B, D]. With DETACH_EVERY, gradient memory is bounded.
- **NaN handling**: HH dynamics can produce NaN if V diverges. The `clamp(-100, 60)` on V is essential.
- **Eval wrapper**: `evaluate_bpb` expects `model(x) -> logits`. The `_EvalWrapper` class strips bio metrics. If you change the model's return signature, update the wrapper.

## Simplicity criterion

All else being equal, simpler is better. Removing something and getting equal or better results is a great outcome. The QHH neuron is already complex — if a simplification (fewer substeps, removing quantum terms, simpler spike mechanism) matches performance, that's a win worth keeping.

**Especially important**: if you find that a simpler neuron (LIF, classical HH without Lindblad) matches full QHH performance, KEEP that experiment and note it prominently. This is a scientifically meaningful result.

## What NOT to do

- Do NOT modify `prepare.py`.
- Do NOT add external dependencies (Flash Attention, etc.) — the QHH model doesn't use attention.
- Do NOT change the required output lines (`val_bpb:` and `peak_vram_mb:`).
- Do NOT try to add attention heads — the whole point is that this is a zero-attention architecture.
- Do NOT remove the secondary bio metric output lines — they're needed for Phase 2 analysis.
