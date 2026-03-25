"""
Autoresearch: Multi-Timescale Hodgkin-Huxley SSM Language Model.
Single-GPU, single-file.

Real HH biophysics with learnable timescale scaling — inspired by how
cortical neurons use different receptor types (AMPA, NMDA, calcium,
neuromodulatory) to process information across multiple timescales.

Each dimension has 4 channels at different speeds:
  Ch 0 "AMPA"    — fast reactions     (k=0.1,    τ ≈ 1-3 tokens)
  Ch 1 "NMDA"    — phrase memory      (k=0.01,   τ ≈ 30 tokens)
  Ch 2 "Calcium" — paragraph context  (k=0.002,  τ ≈ 150 tokens)
  Ch 3 "Neuromod"— topic tracking     (k=0.0005, τ ≈ 600 tokens)

All channels run full HH gating dynamics (alpha/beta rate functions,
ion channel conductances) — just at biologically-motivated different
speeds. Conv1d + SiLU output gate provide the nonlinearities that
make SSMs work for language.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
TIME_BUDGET = 180  # 3 min per experiment

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH        = 6        # number of layers
D_MODEL      = 384      # model dimension
N_CHANNEL    = 4        # receptor-type channels per dimension (AMPA/NMDA/Ca/Neuromod)
BATCH_SIZE   = 64       # reduced: 4-channel HH needs more VRAM per sample
TRAIN_SEQ    = 256      # training sequence length
DT           = 0.5      # base HH timestep

# Optimizer
LR           = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 100
GRAD_CLIP    = 1.0

# Derived
vocab_size   = 8192

# ---------------------------------------------------------------------------
# HH rate functions (the biophysical core)
# ---------------------------------------------------------------------------

_EPS = 1e-7

def alpha_m(V):
    x = V + 40.0
    return torch.where(x.abs() < 1e-4, torch.ones_like(V),
           0.1 * x / (1.0 - torch.exp((-x / 10.0).clamp(-20, 20)) + _EPS))

def beta_m(V):
    return 4.0 * torch.exp((-(V + 65.0) / 18.0).clamp(-20, 20))

def alpha_h(V):
    return 0.07 * torch.exp((-(V + 65.0) / 20.0).clamp(-20, 20))

def beta_h(V):
    return 1.0 / (1.0 + torch.exp((-(V + 35.0) / 10.0).clamp(-20, 20)))

def alpha_n(V):
    x = V + 55.0
    return torch.where(x.abs() < 1e-4, torch.full_like(V, 0.1),
           0.01 * x / (1.0 - torch.exp((-x / 10.0).clamp(-20, 20)) + _EPS))

def beta_n(V):
    return 0.125 * torch.exp((-(V + 65.0) / 80.0).clamp(-20, 20))

# ---------------------------------------------------------------------------
# Linear Scan
# ---------------------------------------------------------------------------

def linear_scan(gates, values):
    """
    Linear recurrence: h_t = gates_t * h_{t-1} + values_t, h_0 = 0.
    Sequential in T, element-wise per step.
    """
    B, T, D = gates.shape
    h = torch.zeros(B, D, device=gates.device, dtype=gates.dtype)
    outputs = []
    for t in range(T):
        h = gates[:, t] * h + values[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)

# ---------------------------------------------------------------------------
# Multi-Timescale HH-SSM Neuron
# ---------------------------------------------------------------------------

class MultiTimescaleHHNeuron(nn.Module):
    """
    Hodgkin-Huxley neuron with learnable timescale scaling.

    The HH rate functions (alpha, beta) provide input-dependent gating
    structure. The timescale factor k controls HOW SLOW these dynamics
    play out — like different receptor types in real cortex.

    For each of N_CHANNEL channels and 3 HH gate variables (m, h, n):
      A = exp(-dt * k_channel * (alpha + beta))
      B = m_inf * (1 - A)

    Then a voltage-like readout combines the gate trajectories with
    learned conductances.

    Conv1d (kernel=4) extracts local patterns before the scan.
    SiLU output gate provides multiplicative nonlinearity.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.n_ch = N_CHANNEL
        # inner = d * N_CHANNEL (one HH system per channel per dimension)
        inner = d * N_CHANNEL
        self.inner = inner

        self.norm = nn.LayerNorm(d)

        # V_eff projection: input -> effective voltage for HH rates
        # Per-dimension, per-channel voltage with spread initialization
        self.v_eff_proj = nn.Linear(d, inner)

        # Output gate branch (SiLU gating, like Mamba)
        self.z_proj = nn.Linear(d, inner)

        # Conv1d for local pattern extraction (causal, depthwise)
        self.conv1d = nn.Conv1d(inner, inner, kernel_size=4, padding=3, groups=inner)

        # Synaptic input current projection
        self.i_syn_proj = nn.Linear(d, inner)

        # Learnable timescale factors (log-space for stability)
        # Initialized to AMPA/NMDA/Calcium/Neuromodulatory timescales
        init_k = [0.1, 0.01, 0.002, 0.0005]
        log_k = torch.zeros(N_CHANNEL, d)
        for ch, k in enumerate(init_k):
            log_k[ch].fill_(math.log(k))
        self.log_k = nn.Parameter(log_k)  # [N_CHANNEL, d]

        # Learnable ion channel conductances (per channel, shared across d)
        self.log_gNa = nn.Parameter(torch.full((N_CHANNEL,), math.log(120.0)))
        self.log_gK  = nn.Parameter(torch.full((N_CHANNEL,), math.log(36.0)))
        self.log_gL  = nn.Parameter(torch.full((N_CHANNEL,), math.log(0.3)))
        self.register_buffer('ENa', torch.tensor(50.0))
        self.register_buffer('EK',  torch.tensor(-77.0))
        self.register_buffer('EL',  torch.tensor(-54.4))

        # V_eff bias: spread across voltage range per channel
        # So different channels sample different HH regimes
        v_bias = torch.zeros(N_CHANNEL, d)
        for ch in range(N_CHANNEL):
            v_bias[ch] = torch.linspace(-80.0 + ch * 10, -30.0 + ch * 5, d)
        self.v_bias = nn.Parameter(v_bias.reshape(inner))
        self.v_scale = nn.Parameter(torch.ones(inner) * 0.3)

        # State readout: combine channels back to d
        self.state_readout = nn.Linear(inner, d)

    def forward(self, x):
        """x: [B, T, D] -> output: [B, T, D], spike_rate, coherence"""
        B, T, D = x.shape
        NC = self.n_ch
        inner = self.inner

        normed = self.norm(x)

        # V_eff: input-predicted effective voltage for HH rates
        V_eff = self.v_eff_proj(normed) * self.v_scale + self.v_bias  # [B, T, inner]

        # Output gate branch
        z = self.z_proj(normed)                                        # [B, T, inner]

        # Synaptic input + Conv1d for local patterns
        I_syn = self.i_syn_proj(normed)                                # [B, T, inner]
        I_syn = self.conv1d(I_syn.transpose(1, 2))[:, :, :T].transpose(1, 2)
        I_syn = F.silu(I_syn)

        # --- HH dynamics with timescale scaling (fp32 for exponentials) ---
        with torch.amp.autocast('cuda', enabled=False):
            V_eff_f = V_eff.float()
            I_syn_f = I_syn.float()

            # HH rates — all [B, T, inner], computed in parallel
            am, bm = alpha_m(V_eff_f), beta_m(V_eff_f)
            ah, bh = alpha_h(V_eff_f), beta_h(V_eff_f)
            an, bn = alpha_n(V_eff_f), beta_n(V_eff_f)

            # Steady-state gates
            m_inf = am / (am + bm + _EPS)
            h_inf = ah / (ah + bh + _EPS)
            n_inf = an / (an + bn + _EPS)

            # Timescale factors k per channel, broadcast to [1, 1, inner]
            k = self.log_k.exp().reshape(1, 1, inner)  # [1, 1, N_CHANNEL*D]

            # Scaled gate dynamics: A = exp(-dt * k * (alpha + beta))
            # k makes the dynamics SLOWER — like NMDA vs AMPA
            A_m = torch.exp((-DT * k * (am + bm)).clamp(min=-20.0))
            B_m = m_inf * (1.0 - A_m)

            A_h = torch.exp((-DT * k * (ah + bh)).clamp(min=-20.0))
            B_h = h_inf * (1.0 - A_h)

            A_n = torch.exp((-DT * k * (an + bn)).clamp(min=-20.0))
            B_n = n_inf * (1.0 - A_n)

            # Batch all 3 gate scans
            A_gates = torch.cat([A_m, A_h, A_n], dim=-1)  # [B, T, 3*inner]
            B_gates = torch.cat([B_m, B_h, B_n], dim=-1)
            gate_traj = linear_scan(A_gates, B_gates)       # [B, T, 3*inner]
            m_traj = gate_traj[:, :, :inner]
            h_traj = gate_traj[:, :, inner:2*inner]
            n_traj = gate_traj[:, :, 2*inner:]

            # Voltage-like readout from gate trajectories
            # Use conductances to weight gate contributions
            gNa = self.log_gNa.exp()  # [N_CHANNEL]
            gK  = self.log_gK.exp()
            gL  = self.log_gL.exp()

            # Expand conductances to [1, 1, inner]
            gNa_e = gNa.repeat_interleave(D).reshape(1, 1, inner)
            gK_e  = gK.repeat_interleave(D).reshape(1, 1, inner)
            gL_e  = gL.repeat_interleave(D).reshape(1, 1, inner)

            # Total conductance from gate trajectories
            g_total = gNa_e * m_traj**3 * h_traj + gK_e * n_traj**4 + gL_e
            A_V = torch.exp((-DT * k * g_total).clamp(min=-20.0))

            # Voltage drive: synaptic input + reversal potential currents
            I_rev = (gNa_e * m_traj**3 * h_traj * self.ENa
                   + gK_e * n_traj**4 * self.EK
                   + gL_e * self.EL)
            B_V = (I_syn_f + I_rev) / (g_total + _EPS) * (1.0 - A_V)

            V_traj = linear_scan(A_V, B_V)  # [B, T, inner]

        # SiLU output gate — the critical nonlinearity
        out = V_traj * F.silu(z)              # [B, T, inner]

        # Combine channels back to model dimension
        out = self.state_readout(out)          # [B, T, D]

        # Bio diagnostics
        with torch.no_grad():
            spike = torch.sigmoid(5.0 * (V_traj - (-20.0)))
            sr = spike.mean()
            # Per-channel timescale diversity
            k_vals = self.log_k.exp()
            coh = k_vals.std() / (k_vals.mean() + _EPS)

        return out, sr, coh


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class HHSSMLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.neuron = MultiTimescaleHHNeuron(d)
        self.proj = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.post_norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.pre_norm(x)
        V, sr, coh = self.neuron(h)
        V = torch.nan_to_num(V, nan=0.0)
        out = self.post_norm(self.proj(V))
        return x + out, sr, coh


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Language Model
# ---------------------------------------------------------------------------

class HHSSM_LM(nn.Module):
    """
    Token -> Embedding + PosEnc -> [HH-SSM Layer x DEPTH] -> LM Head
    Zero attention. Multi-timescale HH state-space model.
    """
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.embedding = nn.Embedding(vocab_size, d)
        self.pos_enc = PositionalEncoding(d, max_len=max(TRAIN_SEQ, MAX_SEQ_LEN) + 1)
        self.layers = nn.ModuleList([HHSSMLayer(d) for _ in range(DEPTH)])
        self.out_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, input_ids):
        x = self.pos_enc(self.embedding(input_ids))
        total_sr = 0.0
        total_coh = 0.0
        for layer in self.layers:
            x, sr, coh = layer(x)
            total_sr += sr
            total_coh += coh
        logits = self.lm_head(self.out_norm(x))
        n = len(self.layers)
        avg_sr = total_sr / n if isinstance(total_sr, torch.Tensor) else torch.tensor(0.0)
        avg_coh = total_coh / n if isinstance(total_coh, torch.Tensor) else torch.tensor(0.0)
        return logits, avg_sr, avg_coh


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model):
    bio_names = {'log_k', 'log_gNa', 'log_gK', 'log_gL', 'v_bias', 'v_scale'}
    bio_params = []
    other_params = []
    for n, p in model.named_parameters():
        if any(k in n for k in bio_names):
            bio_params.append(p)
        else:
            other_params.append(p)
    return torch.optim.AdamW([
        {'params': other_params, 'weight_decay': WEIGHT_DECAY},
        {'params': bio_params,   'weight_decay': 0.0},
    ], lr=LR, betas=(0.9, 0.98), fused=True)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda")
    torch.manual_seed(42)

    model = HHSSM_LM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HH-SSM model: {n_params:,} params, depth={DEPTH}, d_model={D_MODEL}, channels={N_CHANNEL}")
    print(f"  train_seq={TRAIN_SEQ}, batch_size={BATCH_SIZE}")
    print(f"  Multi-timescale HH with Conv1d + SiLU gate.")

    # Print initial timescales
    with torch.no_grad():
        k_init = model.layers[0].neuron.log_k.exp()
        for ch in range(N_CHANNEL):
            k_val = k_init[ch].mean().item()
            rate = 6.7  # typical (alpha+beta) at resting potential
            A_val = math.exp(-DT * k_val * rate)
            tau = -1.0 / math.log(A_val + 1e-10) if A_val < 1.0 else float('inf')
            labels = ["AMPA", "NMDA", "Calcium", "Neuromod"]
            print(f"  Ch {ch} ({labels[ch]}): k={k_val:.4f}, A={A_val:.4f}, τ≈{tau:.0f} tokens")

    optimizer = build_optimizer(model)
    tokenizer = Tokenizer.from_directory()
    train_loader = make_dataloader(tokenizer, BATCH_SIZE, TRAIN_SEQ, "train")

    model.train()
    step = 0
    total_loss = 0.0
    total_tokens = 0
    total_sr = 0.0
    total_coh = 0.0
    n_batches = 0
    nan_batches = 0
    training_start = time.time()
    log_interval = 20

    print(f"\nTraining for {TIME_BUDGET}s ({TIME_BUDGET/60:.1f} min)...")
    print(f"{'step':>6} {'loss':>8} {'lr':>10} {'tok/s':>8} {'sr':>8} {'coh':>10} {'elapsed':>8}")

    for batch_idx, (x, y, epoch) in enumerate(train_loader):
        elapsed = time.time() - training_start
        if elapsed >= TIME_BUDGET:
            break

        x = x.to(device)
        y = y.to(device)

        if x.size(1) > TRAIN_SEQ:
            x = x[:, :TRAIN_SEQ]
            y = y[:, :TRAIN_SEQ]

        if step > 5 and elapsed > 0:
            est_total = int(step * TIME_BUDGET / elapsed)
        else:
            est_total = int(TIME_BUDGET * 3)
        lr = get_lr(step, est_total)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, sr, coh = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-1,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            nan_batches += 1
            step += 1
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            nan_batches += 1
            step += 1
            continue

        optimizer.step()
        optimizer.zero_grad()

        ntok = (y != -1).sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok
        total_sr += (sr.item() if isinstance(sr, torch.Tensor) else sr)
        total_coh += (coh.item() if isinstance(coh, torch.Tensor) else coh)
        n_batches += 1
        step += 1

        if step % log_interval == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            tok_per_sec = total_tokens / max(elapsed, 1e-6)
            avg_sr = total_sr / max(n_batches, 1)
            avg_coh = total_coh / max(n_batches, 1)
            print(f"{step:>6} {avg_loss:>8.4f} {lr:>10.2e} {tok_per_sec:>8.0f} {avg_sr:>8.4f} {avg_coh:>10.6f} {elapsed:>7.1f}s")

    total_training_time = time.time() - training_start
    avg_loss = total_loss / max(total_tokens, 1)
    avg_sr = total_sr / max(n_batches, 1)
    avg_coh = total_coh / max(n_batches, 1)
    print(f"\nTraining done: {step} steps, {total_tokens:,} tokens, {total_training_time:.1f}s")
    print(f"  avg train loss: {avg_loss:.4f}  spike_rate: {avg_sr:.4f}  coherence: {avg_coh:.6f}")
    if nan_batches > 0:
        print(f"  NaN batches: {nan_batches}")

    # Evaluate
    print("\nEvaluating...")
    model.eval()

    class _EvalWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, y=None, reduction='mean'):
            out = self.m(x)
            logits = out[0] if isinstance(out, tuple) else out
            if y is not None:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction=reduction)
            return logits

    eval_model = _EvalWrapper(model)
    import prepare as _prepare
    _orig_eval_tokens = _prepare.EVAL_TOKENS
    _prepare.EVAL_TOKENS = 4 * 524288
    val_bpb = evaluate_bpb(eval_model, tokenizer, BATCH_SIZE)
    _prepare.EVAL_TOKENS = _orig_eval_tokens
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    print(f"val_bpb: {val_bpb:.4f}")
    print(f"peak_vram_mb: {peak_vram:.0f}")
    print(f"spike_rate: {avg_sr:.4f}")
    print(f"coherence: {avg_coh:.6f}")
    print(f"train_loss: {avg_loss:.4f}")
    print(f"total_steps: {step}")
    print(f"tokens_processed: {total_tokens}")
    print(f"tok_per_sec: {total_tokens / max(total_training_time, 1):.0f}")

    # Per-layer timescale report
    labels = ["AMPA", "NMDA", "Calcium", "Neuromod"]
    for li, layer in enumerate(model.layers):
        n = layer.neuron
        k_vals = n.log_k.exp()  # [N_CHANNEL, D]
        gNa = n.log_gNa.exp().tolist()
        gK = n.log_gK.exp().tolist()
        parts = []
        for ch in range(N_CHANNEL):
            k_mean = k_vals[ch].mean().item()
            rate = 6.7
            A_val = math.exp(-DT * k_mean * rate)
            tau = -1.0 / math.log(A_val + 1e-10) if A_val < 0.9999 else 9999
            parts.append(f"{labels[ch]}(k={k_mean:.4f},τ={tau:.0f})")
        print(f"layer_{li}_timescales: {', '.join(parts)}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
