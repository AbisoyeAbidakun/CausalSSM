import math

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with a learnable per-position residual.

    CT uses relative position embeddings baked into each attention block;
    CausalMixer had no positional information at all, so each timestep's
    representation was positionally ambiguous.  Adding PE before the first
    mixer block gives the model an absolute temporal reference that helps
    the time-mixing MLP learn phase-aligned patterns (e.g. "this is the
    3rd timestep of a 20-step sequence").

    Architecture (identical for all streams sharing this module):
        pe_fixed   : sinusoidal base, NOT trained (stable initialisation)
        pe_learned : nn.Parameter of shape [1, max_seq_len, d_model],
                     initialised to 0 so training starts from the pure
                     sinusoidal base and the model learns adjustments.

    The combined encoding is added to the input: x ← x + pe_fixed + pe_learned.

    Parameters
    ----------
    max_seq_len : int   maximum sequence length
    d_model     : int   feature dimension
    dropout     : float applied after adding the encoding (light regularisation)
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Fixed sinusoidal base (same formula as "Attention Is All You Need").
        position = torch.arange(max_seq_len).unsqueeze(1).float()       # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                # [d/2]
        pe = torch.zeros(1, max_seq_len, d_model)                       # [1, T, D]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe_fixed', pe)

        # Learnable residual — starts at zero so model can adapt without
        # disrupting the well-conditioned sinusoidal initialisation.
        self.pe_learned = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, D]
        Returns:
            [B, T, D] with positional information added.
        """
        T = x.size(1)
        pe = self.pe_fixed[:, :T, :] + self.pe_learned[:, :T, :]
        return self.dropout(x + pe)


class CausalTimeMLP(nn.Module):
    """
    Time-mixing MLP with causal constraint.

    Each timestep t can only use information from timesteps 0..t (no future
    leakage).  Analogous to CT's causal attention mask in utils_transformer.py
    but realised as a structured linear layer rather than softmax attention,
    giving O(T) memory instead of O(T²).

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length (same value as EDCT/CT max_seq_length).
    d_model : int
        Feature dimension of the stream being mixed (seq_hidden_units).
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        # Lower-triangular causal mask — shared across all feature dims.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask)

        # Two learnable time-axis projection matrices.
        self.w1 = nn.Parameter(torch.empty(max_seq_len, max_seq_len))
        self.w2 = nn.Parameter(torch.empty(max_seq_len, max_seq_len))
        nn.init.kaiming_uniform_(self.w1, a=0.01)
        nn.init.kaiming_uniform_(self.w2, a=0.01)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, D]
        Returns:
            [B, T, D]  — causal time-mixed features
        """
        T = x.size(1)
        x = self.norm(x)

        # Slice to actual sequence length and apply causal mask to BOTH
        # projection matrices so neither w1 nor w2 can propagate future
        # information into earlier timesteps (Bug 2 fix: w2 was previously
        # unmasked, allowing position t to receive signal from t+k via w2).
        w1 = self.w1[:T, :T] * self.mask[:T, :T]   # zero out future positions
        w2 = self.w2[:T, :T] * self.mask[:T, :T]   # ← now also causally masked

        # Mix across the time axis per feature channel: [B, D, T] @ [T, T]
        x = x.transpose(1, 2)           # [B, D, T]
        x = self.act(x @ w1.T)          # [B, D, T]
        x = x @ w2.T                    # [B, D, T]
        return x.transpose(1, 2)        # [B, T, D]


class CausalGatedMixer(nn.Module):
    """
    SCM-guided gated cross-stream mixer.

    Paper: "Causal State-Space Model for Causal Inference: Estimating
    Longitudinal Individual Treatment Effects", Section 4.3 ("The Causal
    Gated Mixer"). Implements the gated fusion
    BR_t = sigma(g_{A->Y})*a~_t (+) sigma(g_{Y->A})*y~_t (+) x~_t from that
    section, with g_{A->Y} initialised to 1 (sigma~=0.73, near-open — treatment
    causally influences outcomes) and g_{Y->A} initialised to -3 (sigma~=0.05,
    near-closed — the reverse direction is not structurally causal). Used by
    every proposed model (CSSD, CSSPD, CHSD, CHSPD) to fuse the per-stream
    encoder outputs into the balancing representation BR_t.

    Encodes the structural causal model directly in the architecture rather
    than relying on balanced representations alone.  Only causally-consistent
    information flows are permitted:

        treatments → outcomes   (direct causal effect)
        vitals     → outcomes   (confounder path)
        outcomes   → treatments (confounding / feedback)
        outcomes   → vitals     (limited reverse path)

    Edges t→v and v→t are omitted because they are absent from the SCM.
    Each allowed direction has a single learned gate scalar, initialised so
    that the two primary causal paths (t→o, v→o) are open and the feedback
    paths are gated closed at the start of training.

    Bug 4 fix: each stream is now pre-normalised with LayerNorm before
    participating in the cross-stream gated additions.  Without normalisation,
    the time-mixing step (CausalTimeMLP) can shift the scale of each stream
    differently, causing the gate magnitudes to become incomparable and
    producing gradient variance that grows with num_layer and sequence length.
    Pre-norm follows the same pattern as CausalTimeMLP and the standard
    Transformer pre-norm convention.

    Parameters
    ----------
    d_model    : int   feature dimension of each stream (seq_hidden_units).
    has_vitals : bool  whether the vitals stream is present.
    """

    def __init__(self, d_model: int, has_vitals: bool = True):
        super().__init__()
        self.has_vitals = has_vitals

        # Pre-norm LayerNorm for each active stream (Bug 4 fix).
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        if has_vitals:
            self.norm_v = nn.LayerNorm(d_model)

        # Feedback / confounding paths — near-closed at init (sigmoid(-3) ≈ 0.05).
        # Initialised to -3 so the Y→A reverse path is almost fully gated off,
        # reflecting the SCM prior that outcomes do not cause treatments.
        self.gate_o_to_t = nn.Parameter(torch.full((1,), -3.0))
        # Primary causal paths — start open (sigmoid(1) ≈ 0.73).
        self.gate_t_to_o = nn.Parameter(torch.ones(1))
        if has_vitals:
            self.gate_v_to_o = nn.Parameter(torch.ones(1))
            self.gate_o_to_v = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_t: torch.Tensor,
        x_o: torch.Tensor,
        x_v=None,
    ):
        """
        Args:
            x_t : [B, T, D]  treatment stream
            x_o : [B, T, D]  outcome stream
            x_v : [B, T, D] or None  vitals stream
        Returns:
            (x_t, x_o, x_v) with vitals, or (x_t, x_o) without vitals,
            after SCM-gated pre-norm mixing.
        """
        # Normalise inputs before mixing (pre-norm pattern; residuals are
        # the un-normalised originals).
        x_t_n = self.norm_t(x_t)
        x_o_n = self.norm_o(x_o)

        if self.has_vitals and x_v is not None:
            x_v_n = self.norm_v(x_v)
            # SCM-guided cross-stream additions (on normalised streams).
            x_t = x_t + torch.sigmoid(self.gate_o_to_t) * x_o_n
            x_o = (x_o
                   + torch.sigmoid(self.gate_t_to_o) * x_t_n
                   + torch.sigmoid(self.gate_v_to_o) * x_v_n)
            x_v = x_v + torch.sigmoid(self.gate_o_to_v) * x_o_n
            return x_t, x_o, x_v
        else:
            # No vitals: only the t↔o exchange paths apply.
            x_t = x_t + torch.sigmoid(self.gate_o_to_t) * x_o_n
            x_o = x_o + torch.sigmoid(self.gate_t_to_o) * x_t_n
            return x_t, x_o


class CausalMixerBlock(nn.Module):
    """
    Single mixing layer — replaces TransformerMultiInputBlock.

    Three steps per block:
      1. CausalTimeMLP per stream   — temporal mixing, causal, O(T) memory.
      2. CausalGatedMixer           — cross-stream mixing respecting the SCM.
      3. Shared feature-mixing FFN  — replaces all 6 cross-attention modules
                                      with a single concat→project→split MLP.

    Parameters
    ----------
    max_seq_len : int
    d_model     : int   hidden size (seq_hidden_units); same for all streams.
    fc_hidden   : int   inner width of the feature-mixing FFN.
    dropout     : float
    has_vitals  : bool
    """

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        fc_hidden: int,
        dropout: float,
        has_vitals: bool = True,
    ):
        super().__init__()
        self.has_vitals = has_vitals
        self.d_model = d_model

        # Step 1 — per-stream causal time mixing.
        self.time_mix_t = CausalTimeMLP(max_seq_len, d_model)
        self.time_mix_o = CausalTimeMLP(max_seq_len, d_model)
        if has_vitals:
            self.time_mix_v = CausalTimeMLP(max_seq_len, d_model)

        # Step 2 — SCM-guided gated cross-stream mixing (now with pre-norm,
        # requires d_model and has_vitals so LayerNorms can be sized correctly).
        self.causal_gate = CausalGatedMixer(d_model=d_model, has_vitals=has_vitals)

        # Step 3 — shared feature-mixing MLP over concatenated streams.
        n_streams = 3 if has_vitals else 2
        concat_dim = d_model * n_streams
        self.feature_mix = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, concat_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_o: torch.Tensor,
        x_v=None,
    ):
        """
        Args:
            x_t : [B, T, D]
            x_o : [B, T, D]
            x_v : [B, T, D] or None
        Returns:
            Tuple of updated (x_t, x_o) or (x_t, x_o, x_v).
        """
        # ── Step 1: causal temporal mixing ──────────────────────────────────
        x_t = x_t + self.time_mix_t(x_t)
        x_o = x_o + self.time_mix_o(x_o)
        if self.has_vitals and x_v is not None:
            x_v = x_v + self.time_mix_v(x_v)

        # ── Step 2: SCM-gated cross-stream mixing ────────────────────────────
        # Pre-normalisation and both vitals/no-vitals paths are now handled
        # inside CausalGatedMixer.forward (Bug 4 fix: LayerNorm added there).
        if self.has_vitals and x_v is not None:
            x_t, x_o, x_v = self.causal_gate(x_t, x_o, x_v)
        else:
            x_t, x_o = self.causal_gate(x_t, x_o)

        # ── Step 3: shared feature-mixing FFN ───────────────────────────────
        if self.has_vitals and x_v is not None:
            x_cat = torch.cat([x_t, x_o, x_v], dim=-1)
        else:
            x_cat = torch.cat([x_t, x_o], dim=-1)

        x_mixed = x_cat + self.feature_mix(x_cat)      # residual

        # Split concatenated representation back to per-stream tensors.
        parts = x_mixed.split(self.d_model, dim=-1)     # each [B, T, D]
        if self.has_vitals and x_v is not None:
            return parts[0], parts[1], parts[2]
        return parts[0], parts[1]
