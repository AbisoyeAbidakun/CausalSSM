"""
SSMMultiInputBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.helper_models.causal_mamba_layer import CausalMambaLayer
from src.models.helper_models.causal_mixer_block import CausalGatedMixer
from src.models.utils_transformer import PositionwiseFeedForward, TransformerMultiInputBlock, MultiHeadedAttention


class SSMMultiInputBlock(nn.Module):
    """
    Selective-SSM multi-stream block — drop-in for TransformerMultiInputBlock.

    Parameters
    ----------
    hidden              : int   Model hidden dimension (seq_hidden_units).
    feed_forward_hidden : int   Inner width of the per-stream FFN.
                                 seq_hidden_units × 4 by default.
    dropout             : float Dropout for the FFN layers.
    n_inputs            : int   2 (no vitals) or 3 (with vitals).
    d_state             : int   SSM state dimension N (default 16).
    d_conv              : int   Causal conv width in CausalMambaLayer (default 4).
    expand              : int   Inner-dim multiplier in CausalMambaLayer (default 1).
    """

    def __init__(
        self,
        hidden:               int,
        feed_forward_hidden:  int   = None,
        dropout:              float = 0.1,
        n_inputs:             int   = 2,
        d_state:              int   = 16,
        d_conv:               int   = 4,
        expand:               int   = 1,
        **kwargs,
    ):
        super().__init__()

        self.n_inputs   = n_inputs
        self.has_vitals = (n_inputs == 3)

        # feed_forward_hidden defaults to 4 × hidden (CT convention)
        ff_hidden = feed_forward_hidden if feed_forward_hidden is not None else 4 * hidden

        # ── Per-stream selective SSM (replaces self-attention per stream) ────
        # CausalMambaLayer: [B, T, D] → [B, T, D], strictly causal.
        # Δ_t, B_t, C_t are all input-dependent → content-adaptive temporal
        # mixing, unlike CausalTimeMLP's shared fixed parameter matrices.
        self.ssm_t = CausalMambaLayer(hidden, d_state, d_conv, expand)
        self.ssm_o = CausalMambaLayer(hidden, d_state, d_conv, expand)
        if self.has_vitals:
            self.ssm_v = CausalMambaLayer(hidden, d_state, d_conv, expand)

        # ── SCM-guided cross-stream mixing (replaces cross-attention) ────────
        # CausalGatedMixer encodes the causal graph in the architecture:
        #   t→o path: gate init = sigmoid(+1.0) ≈ 0.73 (near-open)
        #   o→t path: gate init = sigmoid(-3.0) ≈ 0.05 (near-closed)
        #   v→o path: gate init = sigmoid(+1.0) ≈ 0.73 (near-open)  [if vitals]
        #   o→v path: gate init = sigmoid(0.0)  = 0.50 (half-open)  [if vitals]
        # Pre-norm is applied inside CausalGatedMixer before each gated addition.
        self.causal_gate = CausalGatedMixer(d_model=hidden, has_vitals=self.has_vitals)

        # ── Per-stream feedforward
        self.feed_forwards = nn.ModuleList([
            PositionwiseFeedForward(d_model=hidden, d_ff=ff_hidden, dropout=dropout)
            for _ in range(n_inputs)
        ])

    def forward(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
    ):
        """
        Forward pass — identical signature to TransformerMultiInputBlock.forward.

        Args
        ----
        x_tov  : tuple of tensors — (x_t, x_o) or (x_t, x_o, x_v)
                 each [B, T, D]
        x_s    : static features [B, 1, D]
                 Injected into the FFN input (broadcast over T), matching CT.
        active_entries_treat_outcomes : [B, T, 1]
                 Active-entry mask for treatments and outcomes.
                 Used by TransformerMultiInputBlock for attention masking;
                 not needed by SSM (scan is inherently causal) but accepted
                 for interface compatibility.
        active_entries_vitals : [B, T, 1] or None
                 Same as above for the vitals stream.

        Returns
        -------
        (x_t, x_o) or (x_t, x_o, x_v) — updated stream tensors [B, T, D]
        """
        assert len(x_tov) == self.n_inputs, (
            f"SSMMultiInputBlock expected {self.n_inputs} inputs, got {len(x_tov)}"
        )

        x_v = None  # defined unconditionally so pre-mixer store is always valid
        if self.has_vitals:
            x_t, x_o, x_v = x_tov
        else:
            x_t, x_o = x_tov

        # ── Step 1: per-stream selective temporal mixing ──────────────────────
        # Residual connection: x = x + SSM(x)
        # The SSM internally applies LayerNorm before its projection (pre-norm),
        # matching the pre-norm convention used in CausalMixerBlock.
        x_t = x_t + self.ssm_t(x_t)
        x_o = x_o + self.ssm_o(x_o)
        if self.has_vitals:
            x_v = x_v + self.ssm_v(x_v)

        # Store pre-mixer embeddings for x_local capture in SSTCP
        self._pre_mixer_x_t = x_t          # [B, T, hidden]
        self._pre_mixer_x_o = x_o
        self._pre_mixer_x_v = x_v          # None if no vitals

        # ── Step 2: SCM-guided cross-stream exchange ──────────────────────────
        # CausalGatedMixer applies its own LayerNorm internally before mixing,
        # so inputs here are the (already-residually-updated) stream tensors.
        # The gate controls: which causal paths are open and at what strength.
        if self.has_vitals:
            x_t, x_o, x_v = self.causal_gate(x_t, x_o, x_v)
        else:
            x_t, x_o = self.causal_gate(x_t, x_o)

        # ── Step 3: per-stream feedforward with static feature injection ──────
        # CT adds x_s [B, 1, D] (static features, broadcast over T) to the
        # FFN input at every block.  This preserves that behaviour exactly.
        # PositionwiseFeedForward already applies LayerNorm + residual internally.
        out_t = self.feed_forwards[0](x_t + x_s)
        out_o = self.feed_forwards[1](x_o + x_s)

        if self.has_vitals:
            out_v = self.feed_forwards[2](x_v + x_s)
            return out_t, out_o, out_v

        return out_t, out_o


class MultiScaleSSMBlock(nn.Module):
    """
    Multi-scale temporal wrapper around SSMMultiInputBlock.

    Parameters
    ----------
    hidden              : int     Model hidden dimension (seq_hidden_units).
    feed_forward_hidden : int     Per-stream FFN inner width (default 4×hidden).
    dropout             : float   FFN dropout rate.
    n_inputs            : int     2 (no vitals) or 3 (with vitals).
    d_state             : int     SSM state dimension N (default 16).
    d_conv              : int     Causal conv width in CausalMambaLayer (default 4).
    expand              : int     Inner-dim multiplier (default 1).
    scales              : list    Temporal downsampling ratios (default [1, 2, 4]).
    **kwargs            :         Absorbed; allows passing CT attention kwargs
                                  (num_heads, head_size, …) without error.
    """

    def __init__(
        self,
        hidden:               int,
        feed_forward_hidden:  int   = None,
        dropout:              float = 0.1,
        n_inputs:             int   = 2,
        d_state:              int   = 16,
        d_conv:               int   = 4,
        expand:               int   = 1,
        scales:               list  = None,
        **kwargs,
    ):
        super().__init__()

        self.scales     = list(scales) if scales is not None else [1, 2, 4]
        self.n_inputs   = n_inputs
        self.has_vitals = (n_inputs == 3)

        # One independent SSMMultiInputBlock per temporal scale.
        # Scale-1 block processes full-resolution T; coarser blocks see fewer steps
        # so their SSM states capture longer-range context per update.
        self.scale_blocks = nn.ModuleList([
            SSMMultiInputBlock(
                hidden              = hidden,
                feed_forward_hidden = feed_forward_hidden,
                dropout             = dropout,
                n_inputs            = n_inputs,
                d_state             = d_state,
                d_conv              = d_conv,
                expand              = expand,
            )
            for _ in self.scales
        ])

        # Learned mixing weights — initialised to uniform (log-odds = 0).
        # After softmax the model can learn to weight scales dynamically.
        self.mix_logits = nn.Parameter(torch.zeros(len(self.scales)))

        # Placeholders for CSSPD._get_x_local compatibility.
        # Populated during forward() from the scale-1 block.
        self._pre_mixer_x_t = None
        self._pre_mixer_x_o = None
        self._pre_mixer_x_v = None

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
        """[B, T, D] → [B, floor(T/scale), D] via AvgPool1d."""
        if scale == 1:
            return x
        B, T, D = x.shape
        x = x.transpose(1, 2)                                  # [B, D, T]
        x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
        return x.transpose(1, 2)                               # [B, T', D]

    @staticmethod
    def _upsample(x: torch.Tensor, T_target: int) -> torch.Tensor:
        """[B, T_small, D] → [B, T_target, D] via nearest-neighbour."""
        if x.shape[1] == T_target:
            return x
        x = x.transpose(1, 2)                                  # [B, D, T]
        x = F.interpolate(x, size=T_target, mode='nearest')
        return x.transpose(1, 2)                               # [B, T_target, D]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
    ):
        """
        Multi-scale forward — identical signature to SSMMultiInputBlock.forward.

        Args
        ----
        x_tov  : tuple (x_t, x_o) or (x_t, x_o, x_v), each [B, T, D]
        x_s    : [B, 1, D]  static features (broadcast over T; no downsampling)
        active_entries_treat_outcomes : [B, T, 1]
        active_entries_vitals         : [B, T, 1] or None

        Returns
        -------
        (x_t, x_o) or (x_t, x_o, x_v) — aggregated multi-scale outputs [B, T, D]
        """
        if self.has_vitals:
            x_t, x_o, x_v = x_tov
        else:
            x_t, x_o = x_tov
            x_v = None

        T = x_t.shape[1]
        weights = F.softmax(self.mix_logits, dim=0)     # [K]

        # Initialise accumulators on the correct device/dtype
        acc_t = torch.zeros_like(x_t)
        acc_o = torch.zeros_like(x_o)
        acc_v = torch.zeros_like(x_v) if x_v is not None else None

        for i, (scale, block) in enumerate(zip(self.scales, self.scale_blocks)):
            # ── 1. Downsample each temporal stream ────────────────────────────
            xt_s  = self._downsample(x_t, scale)
            xo_s  = self._downsample(x_o, scale)
            xv_s  = self._downsample(x_v, scale)  if x_v is not None else None
            ae_s  = self._downsample(active_entries_treat_outcomes, scale)
            aev_s = self._downsample(active_entries_vitals, scale) \
                    if active_entries_vitals is not None else None

            x_tov_s = (xt_s, xo_s, xv_s) if self.has_vitals else (xt_s, xo_s)

            # ── 2. Run SSMMultiInputBlock at this temporal resolution ─────────
            # x_s (static) is NOT downsampled — it has no temporal dimension.
            out_s = block(x_tov_s, x_s, ae_s, aev_s)

            # Unpack per-scale outputs
            if self.has_vitals:
                out_t_s, out_o_s, out_v_s = out_s
            else:
                out_t_s, out_o_s = out_s
                out_v_s = None

            # ── 3. Upsample and weighted-accumulate ───────────────────────────
            w = weights[i]
            acc_t = acc_t + w * self._upsample(out_t_s, T)
            acc_o = acc_o + w * self._upsample(out_o_s, T)
            if acc_v is not None:
                acc_v = acc_v + w * self._upsample(out_v_s, T)

            # ── Store finest-scale pre-mixer embeddings (scale == 1) ──────────
            # SSMMultiInputBlock stores these DURING its forward as side effects.
            # can extract x_local at full temporal resolution — same as CSSPD.
            if scale == 1:
                self._pre_mixer_x_t = getattr(block, '_pre_mixer_x_t', None)
                self._pre_mixer_x_o = getattr(block, '_pre_mixer_x_o', None)
                self._pre_mixer_x_v = getattr(block, '_pre_mixer_x_v', None)

        if self.has_vitals:
            return acc_t, acc_o, acc_v
        return acc_t, acc_o


class HybridSSMAttentionMultiInputBlock(nn.Module):
    """
    Hybrid SSM-Attention block — Jamba / Griffin style.

    Interleaves cheap O(T) SSM layers for local feature extraction with
    a single sparse O(T²) attention layer for selective global recall.

    Architecture per compound block
    ────────────────────────────────
      SSM sub-block 1: CausalMambaLayer(t) + CausalMambaLayer(o)
                       → CausalGatedMixer → FFN × 2
      SSM sub-block 2: CausalMambaLayer(t) + CausalMambaLayer(o)
                       → CausalGatedMixer → FFN × 2
        ⋮   (ssm_per_attn total)
      Attention block: CausalSelfAttn(t) + CausalSelfAttn(o)
                       + CausalCrossAttn(t→o) + CausalCrossAttn(o→t)
                       → FFN × 2


    Why this helps for causal inference
    ────────────────────────────────────
      • SSM layers handle O(T) local feature extraction and temporal mixing,
        preserving the computational efficiency.
      • The single attention layer provides global recall — it can directly
        associate treatment assignments at time t with outcomes at t+k
        across the full sequence.  For T=59, one attention layer adds only
        T² = 3,481 operations per sequence — negligible cost, maximal gain.
      • The CausalGatedMixer inside each SSM block retains SCM-guided causal
        path gating for cheap inter-stream mixing between attention layers.
      • The attention layer is TransformerMultiInputBlock: causally
        masked, with correct o→t and t→o cross-attention, directly reusing
        the representation structure learned.

    Interface compatibility
    ───────────────────────
      • Same forward signature as SSMMultiInputBlock / TransformerMultiInputBlock.
      • Exposes `_pre_mixer_x_{t,o,v}` from the last SSM sub-block.

    Parameters
    ----------
    hidden              : int    Model hidden dimension (seq_hidden_units).
    feed_forward_hidden : int    Inner width for each FFN (default 4 × hidden).
    dropout             : float  Dropout for all FFN layers.
    n_inputs            : int    2 (no vitals) or 3 (with vitals).
    d_state             : int    SSM state dimension N (default 16).
    d_conv              : int    Causal conv width in CausalMambaLayer (default 4).
    expand              : int    Inner-dim multiplier in CausalMambaLayer (default 1).
    ssm_per_attn        : int    SSM sub-blocks before each attention block (default 2).
    attn_heads          : int    Number of attention heads (default 2).
    attn_dropout        : float  Attention dropout (default 0.0).
    **kwargs            :        Absorbed; allows attention kwargs without error.
    """

    def __init__(
        self,
        hidden:               int,
        feed_forward_hidden:  int   = None,
        dropout:              float = 0.1,
        n_inputs:             int   = 2,
        d_state:              int   = 16,
        d_conv:               int   = 4,
        expand:               int   = 1,
        ssm_per_attn:         int   = 2,
        attn_heads:           int   = 2,
        attn_dropout:         float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.n_inputs    = n_inputs
        self.has_vitals  = (n_inputs == 3)
        self.ssm_per_attn = ssm_per_attn

        ff_hidden = feed_forward_hidden if feed_forward_hidden is not None else 4 * hidden

        # ── SSM sub-layers ────────────────────────────────────────────────────
        # Each is a complete SSMMultiInputBlock: CausalMambaLayer × streams +
        # CausalGatedMixer + PositionwiseFeedForward.
        self.ssm_blocks = nn.ModuleList([
            SSMMultiInputBlock(
                hidden              = hidden,
                feed_forward_hidden = ff_hidden,
                dropout             = dropout,
                n_inputs            = n_inputs,
                d_state             = d_state,
                d_conv              = d_conv,
                expand              = expand,
            )
            for _ in range(ssm_per_attn)
        ])

        # ── Attention sub-layer ───────────────────────────────────────────────
        #   • Causal self-attention per stream (one_direction=True)
        #   • Causal cross-attention t→o and o→t
        #   • PositionwiseFeedForward per stream
        # head_size=None → defaults to hidden // attn_heads.
        # With hidden=32, attn_heads=2: head_size=16, output=32 = hidden. ✓
        self.attn_block = TransformerMultiInputBlock(
            hidden              = hidden,
            attn_heads          = attn_heads,
            head_size           = None,     # hidden // attn_heads
            feed_forward_hidden = ff_hidden,
            dropout             = dropout,
            attn_dropout        = attn_dropout,
            n_inputs            = n_inputs,
            final_layer         = False,
            disable_cross_attention = False,
        )

        # ── Placeholders for CSSPD._get_x_local compatibility ───────────────
        # Populated from the last SSM sub-block during forward().
        self._pre_mixer_x_t = None
        self._pre_mixer_x_o = None
        self._pre_mixer_x_v = None

    def forward(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
    ):
        """
        Hybrid forward — SSM sub-layers then sparse attention layer.

        Args
        ----
        x_tov  : tuple (x_t, x_o) or (x_t, x_o, x_v), each [B, T, D]
        x_s    : [B, 1, D]  static features (broadcast over T; unchanged)
        active_entries_treat_outcomes : [B, T, 1]
        active_entries_vitals         : [B, T, 1] or None

        Returns
        -------
        (x_t, x_o) or (x_t, x_o, x_v) — updated stream tensors [B, T, D]
        """
        assert len(x_tov) == self.n_inputs, (
            f'HybridSSMAttentionMultiInputBlock expected {self.n_inputs} inputs, '
            f'got {len(x_tov)}'
        )

        # ── Step 1: sequential SSM sub-layers ────────────────────────────────
        for idx, ssm_block in enumerate(self.ssm_blocks):
            x_tov = ssm_block(
                x_tov,
                x_s,
                active_entries_treat_outcomes,
                active_entries_vitals,
            )
            # Capture pre-mixer embeddings from the LAST SSM sub-layer.
            # These are the representations *before* CausalGatedMixer mixing,
            # which is exactly what CSSPD._get_x_local needs for LIM.
            if idx == self.ssm_per_attn - 1:
                self._pre_mixer_x_t = getattr(ssm_block, '_pre_mixer_x_t', None)
                self._pre_mixer_x_o = getattr(ssm_block, '_pre_mixer_x_o', None)
                self._pre_mixer_x_v = getattr(ssm_block, '_pre_mixer_x_v', None)

        # ── Step 2: sparse global attention layer ────────────────────────────
        # TransformerMultiInputBlock applies causal masking internally via
        # one_direction=True in all MultiHeadedAttention calls.
        # x_s is added inside feed_forwards as a build_br.
        x_tov = self.attn_block(
            x_tov,
            x_s,
            active_entries_treat_outcomes,
            active_entries_vitals,
        )

        return x_tov


class CausallyConstrainedHybridBlock(nn.Module):
    """
    Causally-Constrained Hybrid (CCH) Block.

    Designed from first principles for the Individual Treatment Effect (ITE)
    setting, motivated directly by the backdoor adjustment criterion.

    The central insight
    ───────────────────
    Treatment assignment is the *confounding source* in longitudinal causal
    inference.  For the sequential ignorability assumption to hold —

        Y_t(ā) ⊥ A_t | H_t      (Pearl, 2009; Robins, 1986)

    — the balanced representation BR_t must faithfully encode the full
    treatment history A_{1:t-1}.  A selective-state-space model compresses
    A_{1:t-1} into a fixed-size state h_t via learned gates Δ_t, B_t, C_t.
    If the SSM gates decide that early treatment assignments are less
    relevant than recent ones — which is precisely what content-adaptive
    gating does — then BR_t loses information about A_{1:t-1}, the domain
    confusion objective cannot detect the residual association, and the
    resulting balanced representation is insufficient for deconfounding.

    Full causal self-attention on the treatment stream prevents this
    compression loss: at step t, each treatment representation directly
    attends to *all* past treatment assignments A_1, …, A_{t-1}, without
    any intermediate lossy compression.  This is theoretically necessary;
    it is also provably sufficient for the domain confusion objective to
    detect and remove the A_t → BR_t association.

    Outcome and vitals streams do not require global access:
    • Outcomes Y_t are predicted targets, not confounders.  Their sequential
      dynamics (responses to treatment changes) are well-captured by SSMs.
    • Vitals X_t confound *through* A_t.  Conditioning on the attended
      treatment history A_{1:t-1} already accounts for this path.

    Architecture contrast
    ─────────────────────
      HybridSSMAttentionBlock (Jamba/Griffin):
          All streams → full attention at every N-th layer.
          Motivation: computational efficiency.  No causal structure.

      CausallyConstrainedHybridBlock (this, CCH):
          Treatment stream  → causal self-attention  O(T²)   ← theoretically required
          Outcome stream    → CausalMambaLayer       O(T)    ← efficient, correct
          Vitals stream     → CausalMambaLayer       O(T)    ← efficient, correct
          Cross-stream      → CausalGatedMixer       O(T)    ← SCM-guided, unchanged

    This is NOT Jamba.  Jamba applies attention uniformly by position in the
    stack.  CCH applies attention by *causal role in the identification
    problem* — treatment gets attention because it is the confounding source
    and because the backdoor criterion requires full history encoding; outcomes
    and vitals get SSMs because sequential dynamics suffice.

    The enriched treatment representation (full-history attention) flows to
    the outcome stream through the CausalGatedMixer's open t→o gate, providing
    richer treatment context than an SSM-compressed version would.

    Computational cost
    ──────────────────
    With n_inputs=2 (treatment, outcome), the O(T²) cost is incurred for
    exactly one of two streams.  For T=59: 3,481 attention ops vs 59×d_inner×N
    SSM ops per stream.  Net overhead vs pure SSM: small.

    Interface
    ─────────
    Same forward signature as SSMMultiInputBlock / TransformerMultiInputBlock —
    drop-in replacement in transformer_blocks.

    Stores _pre_mixer_x_{t,o,v} from before the CausalGatedMixer so that
    CSSPD._get_x_local works without any modifications.

    Parameters
    ----------
    hidden              : int    d_model (seq_hidden_units).
    feed_forward_hidden : int    FFN inner width (default 4 × hidden).
    dropout             : float  FFN dropout.
    n_inputs            : int    2 (no vitals) or 3 (with vitals).
    d_state             : int    SSM state dimension N (default 16).
    d_conv              : int    SSM causal conv width (default 4).
    expand              : int    SSM inner-dim multiplier (default 1).
    attn_heads          : int    Treatment self-attention heads (default 2).
    attn_dropout        : float  Treatment attention dropout (default 0.0).
    **kwargs            :        Absorbed (unused attention kwargs).
    """

    def __init__(
        self,
        hidden:               int,
        feed_forward_hidden:  int   = None,
        dropout:              float = 0.1,
        n_inputs:             int   = 2,
        d_state:              int   = 16,
        d_conv:               int   = 4,
        expand:               int   = 1,
        attn_heads:           int   = 2,
        attn_dropout:         float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.n_inputs   = n_inputs
        self.has_vitals = (n_inputs == 3)

        ff_hidden = feed_forward_hidden if feed_forward_hidden is not None else 4 * hidden

        # ── Treatment stream: causal self-attention ───────────────────────────
        # head_size=None → hidden // attn_heads.  With hidden=32, heads=2:
        # head_size=16, output dim = 2×16 = 32 = hidden. ✓
        # MultiHeadedAttention.forward includes residual + LayerNorm internally:
        #   return layer_norm(attention_output + query)
        # So x_t = attn_t(x_t, x_t, x_t, ...) is already residual-connected.
        self.attn_t = MultiHeadedAttention(
            num_heads  = attn_heads,
            d_model    = hidden,
            head_size  = None,
            dropout    = attn_dropout,
            final_layer= False,
        )

        # ── Outcome stream: O(T) selective SSM ───────────────────────────────
        # CausalMambaLayer returns only the delta (residual added externally).
        self.ssm_o = CausalMambaLayer(hidden, d_state, d_conv, expand)

        # ── Vitals stream: O(T) selective SSM (if present) ───────────────────
        if self.has_vitals:
            self.ssm_v = CausalMambaLayer(hidden, d_state, d_conv, expand)

        # ── SCM-guided cross-stream mixer ─────────────────────────────────────
        # O(T) — gates are scalar learned parameters, not attention weights.
        # The open t→o gate allows the enriched treatment representation
        # (post-attention) to inform the outcome stream at each timestep.
        self.causal_gate = CausalGatedMixer(d_model=hidden, has_vitals=self.has_vitals)

        # ── Per-stream feedforward ─────────────────────────────────────────────
        self.feed_forwards = nn.ModuleList([
            PositionwiseFeedForward(d_model=hidden, d_ff=ff_hidden, dropout=dropout)
            for _ in range(n_inputs)
        ])

        # ── Placeholders for CSSPD._get_x_local compatibility ───────────────
        # Populated during forward() from the state BEFORE CausalGatedMixer.
        # For CSSPD, x_local is the per-stream embedding before cross-stream
        # mixing — this semantic is preserved whether the stream is processed
        # by SSM or attention.
        self._pre_mixer_x_t = None
        self._pre_mixer_x_o = None
        self._pre_mixer_x_v = None

    def forward(
        self,
        x_tov,
        x_s,
        active_entries_treat_outcomes,
        active_entries_vitals=None,
    ):
        """
        Causally-constrained forward pass.

        Args
        ----
        x_tov  : tuple (x_t, x_o) or (x_t, x_o, x_v), each [B, T, D]
        x_s    : [B, 1, D]  static features, broadcast over T.
        active_entries_treat_outcomes : [B, T, 1]
        active_entries_vitals         : [B, T, 1] or None

        Returns
        -------
        (x_t, x_o) or (x_t, x_o, x_v) — updated stream tensors [B, T, D]
        """
        assert len(x_tov) == self.n_inputs, (
            f'CausallyConstrainedHybridBlock expected {self.n_inputs} inputs, '
            f'got {len(x_tov)}'
        )

        x_v = None
        if self.has_vitals:
            x_t, x_o, x_v = x_tov
        else:
            x_t, x_o = x_tov

        B, T, _ = x_t.shape

        # ── Step 1: stream-asymmetric temporal mixing ─────────────────────────

        # Treatment: causal self-attention.
        # Causal mask: position t can attend to all positions s ≤ t.
        # one_direction=True enforces the lower-triangular mask inside Attention.
        # MultiHeadedAttention.forward adds the residual and LayerNorm:
        #   return LayerNorm(Attention(x_t) + x_t)
        attn_mask = active_entries_treat_outcomes.repeat(1, 1, T).unsqueeze(1)
        x_t = self.attn_t(x_t, x_t, x_t, attn_mask, one_direction=True)

        # Outcome: selective SSM (O(T)).
        # CausalMambaLayer includes its own pre-norm; residual added here.
        x_o = x_o + self.ssm_o(x_o)

        # Vitals: selective SSM (O(T)).
        if self.has_vitals:
            x_v = x_v + self.ssm_v(x_v)

        # Store per-stream embeddings BEFORE cross-stream mixing.
        # CSSPD._get_x_local reads these to construct x_local for LIM.
        self._pre_mixer_x_t = x_t          # attended treatment embedding
        self._pre_mixer_x_o = x_o          # SSM outcome embedding
        self._pre_mixer_x_v = x_v          # SSM vitals embedding (or None)

        # ── Step 2: SCM-guided cross-stream exchange (O(T)) ───────────────────
        # The enriched treatment representation x_t (full-history attention)
        # flows to x_o through the open t→o gate.
        # Gate magnitudes are scalar learned parameters — O(T) cost.
        if self.has_vitals:
            x_t, x_o, x_v = self.causal_gate(x_t, x_o, x_v)
        else:
            x_t, x_o = self.causal_gate(x_t, x_o)

        # ── Step 3: per-stream feedforward with static feature injection ──────
        out_t = self.feed_forwards[0](x_t + x_s)
        out_o = self.feed_forwards[1](x_o + x_s)

        if self.has_vitals:
            out_v = self.feed_forwards[2](x_v + x_s)
            return out_t, out_o, out_v

        return out_t, out_o
