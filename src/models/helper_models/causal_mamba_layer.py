"""
CausalMambaLayer — Pure-PyTorch selective state space model.

Faithfully implements the architecture from:
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    arXiv:2312.00752 (2024)

Designed as a drop-in replacement for CausalTimeMLP:
    Interface: forward(x: [B, T, D]) → [B, T, D], strictly causal.

Why pure PyTorch instead of the official mamba-ssm package?
  The official package compiles CUDA triton kernels at install time and
  requires a CUDA-capable GPU.  This implementation runs on CPU (and GPU
  without triton) by replacing the hardware-aware scan with a sequential
  loop over T timesteps.  For T ≈ 60 and d_state = 16 the loop adds
  ~2–3 ms per batch on modern hardware — acceptable for research runs.

  When a GPU becomes available, swap this module for MambaSSM(...,
  use_fast_path=True) with identical kwargs.

Architecture (mirrors official mamba_ssm.modules.mamba_simple.Mamba):
  1.  LayerNorm(x)                             — pre-normalise
  2.  in_proj  : D → 2·d_inner                — gated split into x_ssm, z
  3.  conv1d   : depthwise, width d_conv       — local causal context
  4.  x_proj   : d_inner → dt_rank + 2·N      — selective params Δ_low, B, C
  5.  dt_proj  : dt_rank → d_inner            — up-project Δ
  6.  selective scan: h_t = Ā_t h_{t-1} + B̄_t x_t, y_t = C_t h_t
  7.  skip: y += D · x_ssm
  8.  gate: y = y * silu(z)
  9.  out_proj : d_inner → D                  — project back to model dim
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMambaLayer(nn.Module):
    """
    Pure-PyTorch selective SSM block.

    Parameters
    ----------
    d_model  : int   Feature dimension (= seq_hidden_units in CausalMixer /
                     CT). Input and output dim are identical.
    d_state  : int   SSM state size N (default 16 — matches Mamba paper
                     and official repo default).  Larger N = more memory
                     capacity; N=32 adds ~1 % parameters for a large gain
                     on long-range dependencies.
    d_conv   : int   Causal local-conv width (default 4 — same as official
                     repo).  Provides short-range locality before the SSM.
    expand   : int   Inner dimension multiplier: d_inner = expand × d_model.
                     Use expand=1 to match CausalTimeMLP's parameter count
                     footprint; expand=2 matches the official Mamba block.
    dt_min   : float Lower bound of Δ timescale after softplus init.
    dt_max   : float Upper bound of Δ timescale after softplus init.
    """

    def __init__(
        self,
        d_model:  int,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        expand:   int   = 1,
        dt_min:   float = 0.001,
        dt_max:   float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.d_inner  = int(expand * d_model)
        # dt_rank: rank of Δ low-rank projection (official repo: ceil(D/16))
        self.dt_rank  = math.ceil(d_model / 16)

        # ── Pre-norm ─────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)

        # ── 1. Gated input split ─────────────────────────────────────────────
        # Projects D → [x_ssm | z], each of size d_inner.
        # z gates the output (SiLU gate), x_ssm feeds the SSM + conv path.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # ── 2. Causal depthwise conv (local context) ─────────────────────────
        # Mirrors official: Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
        #                          padding=d_conv-1)
        # Left-padding by d_conv-1 gives a causal (no future leakage) window.
        # Trimming the last d_conv-1 output positions enforces strict causality.
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            groups       = self.d_inner,   # depthwise — no cross-channel mix
            padding      = d_conv - 1,     # causal left-pad
            bias         = True,
        )

        # ── 3. Selective parameter projections ───────────────────────────────
        # x → [Δ_low (dt_rank), B (d_state), C (d_state)]
        # Making B and C functions of x is the "selection mechanism":
        # each token chooses what to write into (B) and read from (C) the state.
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )

        # ── 4. Δ up-projection ───────────────────────────────────────────────
        # Low-rank Δ → full d_inner width.
        # Bias initialised so Δ spans [dt_min, dt_max] after softplus (mirrors
        # official repo's dt_init="random" scheme — equation A.3 in the paper).
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # softplus inverse: x = log(exp(dt) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # ── 5. A: log-parameterised diagonal SSM matrix ──────────────────────
        # A_n = -(n+1), n = 0,...,N-1  (HiPPO-inspired from official repo).
        # Stored as A_log so A = -exp(A_log) is always negative (stable decay).
        # Shape: [d_inner, d_state] — one state vector per feature channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0)
        A = A.expand(self.d_inner, -1)            # [d_inner, d_state]
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ── 6. D: skip connection ────────────────────────────────────────────
        # y_t += D * x_ssm_t  (learns how much input to pass through directly)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # ── 7. Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM forward pass (strictly causal, no future information).

        Args
        ----
        x : [B, T, D]

        Returns
        -------
        [B, T, D]
        """
        B, T, D = x.shape
        x = self.norm(x)                                    # pre-norm

        # ── Gated input split ────────────────────────────────────────────────
        xz    = self.in_proj(x)                             # [B, T, 2·d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                     # each [B, T, d_inner]

        # ── Causal local conv ────────────────────────────────────────────────
        # Conv1d expects [B, C, L]; trim right padding so output is causal.
        x_conv = x_ssm.transpose(1, 2)                     # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]             # trim → [B, d_inner, T]
        x_conv = F.silu(x_conv).transpose(1, 2)            # [B, T, d_inner]

        # ── Selective parameters: Δ, B, C — all from input ──────────────────
        # This is the core "selection mechanism": parameters are functions of
        # the current token, allowing the model to decide what to remember or
        # forget at each timestep (unlike fixed-weight LTI models).
        params    = self.x_proj(x_conv)                    # [B, T, dt_rank+2N]
        delta_low, B_sel, C_sel = params.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ: up-project low-rank → d_inner; softplus → strictly positive
        delta = F.softplus(self.dt_proj(delta_low))        # [B, T, d_inner]

        # ── SSM transition matrix A ──────────────────────────────────────────
        # A is kept negative (stable): Ā_t = exp(Δ_t ⊗ A) ∈ (0, 1)
        # Large Δ → Ā ≈ 0: model focuses on current input, resets state.
        # Small Δ → Ā ≈ 1: model retains past state, treats input as noise.
        A = -torch.exp(self.A_log.float())                 # [d_inner, d_state]

        # ── Sequential selective scan ────────────────────────────────────────
        # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
        # y_t = C_t · h_t
        # This loop is O(T · d_inner · d_state) on CPU.  With d_inner=32,
        # d_state=16, T=60: ~30K multiply-adds per sample — fast on CPU.
        h = torch.zeros(
            B, self.d_inner, self.d_state,
            device=x_conv.device,
            dtype=delta.dtype,
        )
        ys = []

        for t in range(T):
            d_t = delta[:, t, :]                           # [B, d_inner]

            # Discretise A: Ā_t = exp(Δ_t ⊗ A)  →  [B, d_inner, d_state]
            # Broadcasting: d_t [B, d_inner, 1] × A [1, d_inner, d_state]
            A_bar = torch.exp(
                d_t.unsqueeze(-1) * A.unsqueeze(0)
            )                                               # [B, d_inner, d_state]

            # Discretise B: B̄_t = Δ_t ⊗ B_t
            # d_t [B, d_inner, 1] × B_sel[:, t] [B, 1, d_state]
            B_bar = (
                d_t.unsqueeze(-1)
                * B_sel[:, t, :].unsqueeze(1)
            )                                               # [B, d_inner, d_state]

            # State update: h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_ssm_t
            h = (
                A_bar * h
                + B_bar * x_conv[:, t, :].unsqueeze(-1)    # [B, d_inner, 1]
            )                                               # [B, d_inner, d_state]

            # Output at step t: y_t = sum_n C_n · h_n (over state dim)
            y_t = (h * C_sel[:, t, :].unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                         # [B, T, d_inner]

        # ── Skip connection ──────────────────────────────────────────────────
        y = y + self.D * x_conv                            # D-weighted input bypass

        # ── SiLU output gate ─────────────────────────────────────────────────
        # z acts as a learned gate: amplify useful features, suppress noise.
        y = y * F.silu(z)                                  # [B, T, d_inner]

        # ── Output projection ────────────────────────────────────────────────
        return self.out_proj(y)                            # [B, T, D]
