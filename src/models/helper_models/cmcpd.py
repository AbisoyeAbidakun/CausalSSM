"""CausalMixerCPCDirect (CMCPD) — CMCP with a Direct Parallel Multi-step Decoder

This variant targets the structural bottleneck of all prior CM/CMC/CMCP models:
the GRU autoregressive rollout accumulates prediction error multiplicatively
across steps.  Each step t's prediction error becomes step t+1's input noise.

CMCPD eliminates the rollout entirely.  Instead, τ independent prediction heads
each receive the full future treatment context and directly regress the outcome
at their specific horizon depth in a single forward pass.

Four improvements over CMCP (all independent and testable):
────────────────────────────────────────────────────────────

  1. Direct Parallel Multi-step Heads (structural fix)
     ───────────────────────────────────────────────────
     GRU rollout: y_{t+k} = f(ŷ_{t+k-1}, a_{t+k}, h_{t+k-1})
       Error at step k feeds into step k+1 → multiplicative accumulation.
     Parallel:    y_{t+k} = head_k(trunk(BR_t, trt_ctx_k))
       No recursion → no error accumulation → no train/inference gap.
       Teacher forcing is structurally impossible (and not needed).
       Training and inference use identical computation graphs.

  2. Larger Representational Capacity
     ────────────────────────────────────
     br_size: 24 → 48, num_layer: 3 → 4, seq_hidden_units: 24 → 48.
     The encoder needs more capacity to capture the 6-step pharmacokinetic
     dynamics.  This is safe to combine with the parallel decoder because
     gradient isolation (br.detach()) is preserved in the training loss.

  3. Treatment Sequence Encoder per Head (Approach 3)
     ──────────────────────────────────────────────────
     Integrated inside ParallelMultiStepDecoder:
     For step k, the treatment context is the CUMULATIVE MEAN of the first
     k+1 treatment embeddings [a_t, a_{t+1}, ..., a_{t+k}].  This gives
     each head causally correct, step-specific treatment information rather
     than FiLM's single-treatment scale/shift.

  4. Reduced Adversarial Balancing Strength (Approach 4)
     ──────────────────────────────────────────────────────
     Domain confusion with high alpha actively erases patient-specific
     covariate information from BR.  At coeff=1.0 (moderate confounding),
     this erasure is over-aggressive.  Setting alpha_rate lower and
     balancing via `balancing_weight` in the config gives the encoder more
     room to retain prognostically useful covariate information.
     The LIM head already counteracts this from the other direction.

Running independently
──────────────────────
  python3 runnables/train_multi.py +backbone=cmcpd \\
      "+backbone/cmcpd_hparams/cancer_sim_domain_conf='1'" \\
      dataset=cancer_sim_basic exp.balancing=domain_confusion

Model hierarchy:
  CM     — CausalMixerBlock backbone + GRU decoder
  CMC    — CM + FiLM + CAETC autoencoding + conditioning loss
  CMCP   — CMC + CPC (temporal contrastive) + Local InfoMax (LIM)
  CMCPD  — CMCP + ParallelMultiStepDecoder (replaces GRU) ← this file

All CMCP losses (CPC, LIM, recon, cond) are fully inherited.
The ONLY changes are in the decoder architecture and the multi-step loss.
"""

import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import Dataset

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.helper_models.cmcp import CausalMixerCPC

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Component — Parallel Multi-step Decoder  (Approaches 1 + 3)
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelMultiStepDecoder(nn.Module):
    """
    Direct parallel multi-step outcome predictor.

    Paper: "Causal State-Space Model for Causal Inference: Estimating
    Longitudinal Individual Treatment Effects", Section 4.5 ("The Parallel
    Multi-Step Decoder"). Implements
    Y_hat_{t+tau}(a_bar) = g_tau(BR_t_perp, TrtEnc(a_bar_{t+1:t+tau})) from
    that section: BR_t_perp is a stop-gradient copy of the balancing
    representation, and TrtEnc produces the two-component treatment
    embedding (cumulative + step-specific, see below) that the paper argues
    resolves treatment-ordering ambiguity. Eliminates the O(epsilon^tau)
    autoregressive rollout error of sequential decoders (Taieb et al.,
    2014). Used by all four proposed models (CSSD, CSSPD, CHSD, CHSPD).

    Replaces the GRU autoregressive rollout with τ independent prediction
    heads — one per future timestep k = 0 .. τ-1.

    No recursion, no teacher forcing, no train/inference gap.

    Treatment Sequence Encoder (integrated, Approach 3)
    ────────────────────────────────────────────────────
    Head k predicts the outcome τ steps ahead conditioned on TWO causal
    treatment signals concatenated together:

        trt_history_k = mean( TreatmentMLP(a_{t+0}), ..., TreatmentMLP(a_{t+k}) )
                                                       ∈ ℝ^{trt_enc_dim}
        trt_current_k = TreatmentMLP(a_{t+k})         ∈ ℝ^{trt_enc_dim}

    trt_history_k encodes the cumulative treatment context — how the patient
    has been treated on average from t to t+k (captures pharmacokinetic history).

    trt_current_k encodes the treatment given specifically at step k — critical
    for distinguishing sequences with the same mean but different ordering.
    Example: [high_chemo, no_drug] and [no_drug, high_chemo] have the same
    cumulative mean at step 2 but very different clinical implications.

    Without trt_current, heads k and k' cannot distinguish these cases.
    Adding it eliminates the ordering ambiguity at negligible parameter cost
    (only widens the first trunk Linear by trt_enc_dim columns).

    Architecture per step k
    ────────────────────────
        ctx_k   = concat(BR_t, trt_history_k, trt_current_k)
                                               [br_size + 2*trt_enc_dim]
        h_k     = trunk(ctx_k)                 [hidden]
        y_{t+k} = head_k(h_k)                 [dim_outcome]

    Parameters
    ----------
    br_size       : balanced representation size (encoder output dim)
    dim_treatments: treatment vector dimensionality
    tau           : projection horizon τ (number of prediction steps)
    trt_enc_dim   : per-step treatment embedding width
    hidden        : trunk hidden layer width
    dim_outcome   : outcome dimensionality
    dropout       : dropout applied inside the trunk
    """

    def __init__(
        self,
        br_size:        int,
        dim_treatments: int,
        tau:            int,
        trt_enc_dim:    int = 32,
        hidden:         int = 128,
        dim_outcome:    int = 1,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.tau         = tau
        self.dim_outcome = dim_outcome

        # ── Per-step treatment embedding (shared across all k) ──────────────
        # Simple Linear → GELU → LayerNorm.  The LayerNorm keeps the embedding
        # on the same scale as the BR regardless of dim_treatments magnitude.
        self.trt_step_mlp = nn.Sequential(
            nn.Linear(dim_treatments, trt_enc_dim),
            nn.GELU(),
            nn.LayerNorm(trt_enc_dim),
        )

        # ── Shared trunk MLP ────────────────────────────────────────────────
        # Two-layer MLP.  Shared parameters reduce total param count and force
        # the step-specific information to be routed entirely through the heads.
        # LayerNorm after the first activation stabilises gradient norms when
        # br_size is large (48+).
        #
        # Input width is br_size + 2*trt_enc_dim:
        #   [BR | trt_history_k | trt_current_k]
        # The extra trt_enc_dim columns (vs the old br_size + trt_enc_dim layout)
        # carry the step-specific treatment embedding, resolving ordering ambiguity.
        self.trunk = nn.Sequential(
            nn.Linear(br_size + 2 * trt_enc_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # ── τ independent output heads ──────────────────────────────────────
        # Each head specialises to the output distribution at its horizon depth.
        # NOT shared: step 1 and step 6 have very different variance profiles
        # in cancer_sim (step 6 is far more uncertain), so independent weights
        # let each head learn the appropriate output scale.
        self.heads = nn.ModuleList([
            nn.Linear(hidden, dim_outcome)
            for _ in range(tau)
        ])

        # Near-zero initialisation prevents large output heads from dominating
        # the backbone gradient in the first few epochs via lambda_ms.
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        br_last:            torch.Tensor,   # [B, br_size]
        future_treatments:  torch.Tensor,   # [B, τ_actual, dim_treatments]
    ) -> torch.Tensor:
        """
        Predict outcomes at all future steps in one parallel pass.

        Returns
        -------
        torch.Tensor : [B, n_steps, dim_outcome]
          where n_steps = min(τ_actual, self.tau)
        """
        B, tau_actual, _ = future_treatments.shape
        n_steps = min(tau_actual, self.tau)
        device  = br_last.device
        dtype   = br_last.dtype

        # Per-step treatment embeddings: [B, n_steps, trt_enc_dim]
        trt_emb = self.trt_step_mlp(future_treatments[:, :n_steps, :])

        # ── Treatment history: causal cumulative mean per step k ────────────
        #   trt_history[:, k, :] = mean(trt_emb[:, 0:k+1, :])
        # Encodes what treatments have accumulated from step 0 to step k.
        # Computed via cumsum — O(n_steps) with no Python loop.
        trt_cumsum  = torch.cumsum(trt_emb, dim=1)                    # [B, n_steps, enc]
        step_counts = torch.arange(
            1, n_steps + 1, device=device, dtype=dtype
        ).view(1, -1, 1)                                               # [1, n_steps, 1]
        trt_history = trt_cumsum / step_counts                         # [B, n_steps, enc]

        # ── Treatment current: step-specific embedding ──────────────────────
        #   trt_current[:, k, :] = TreatmentMLP(a_{t+k})
        # Provides head k with the exact treatment at its target step,
        # resolving ordering ambiguity that trt_history alone cannot distinguish.
        # E.g.: [high_chemo, no_drug] vs [no_drug, high_chemo] share the same
        # cumulative mean at step 2 but have different trt_current at step 2.
        trt_current = trt_emb                                          # [B, n_steps, enc]

        # ── Expand BR and concatenate all three context streams ─────────────
        # ctx[:, k, :] = [BR | trt_history_k | trt_current_k]
        br_exp = br_last.unsqueeze(1).expand(-1, n_steps, -1)          # [B, n_steps, br]
        ctx    = torch.cat([br_exp, trt_history, trt_current], dim=-1) # [B, n, br+2*enc]

        # Shared trunk — Linear layers broadcast over the time dimension.
        h = self.trunk(ctx)   # [B, n_steps, hidden]

        # Per-step independent heads.
        outputs = []
        for k in range(n_steps):
            outputs.append(self.heads[k](h[:, k, :]))   # [B, dim_outcome]

        return torch.stack(outputs, dim=1)   # [B, n_steps, dim_outcome]


# ═══════════════════════════════════════════════════════════════════════════════
# CausalMixerCPCDirect — Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class CausalMixerCPCDirect(CausalMixerCPC):
    """
    CausalMixerCPCDirect (CMCPD) — CMCP with all four targeted improvements.

    Inherits from CausalMixerCPC:
      ✓ CausalMixerBlock backbone  (O(T) mixer architecture)
      ✓ FiLM treatment conditioning (residual scale+shift)
      ✓ PartialAutoencoderHeads     (CAETC reconstruction)
      ✓ Treatment conditioning loss
      ✓ CPCHead     (temporal contrastive predictive coding)
      ✓ LocalInfoMaxHead (Local Deep InfoMax)
      ✓ build_br override (captures pre-mixer input for InfoMax)

    Replaces:
      ✗ GRUMultiStepDecoder (autoregressive, error-accumulating)
      ✓ ParallelMultiStepDecoder (direct parallel, zero accumulation)

    Overrides only three methods:
      _init_specific            : swaps GRU for ParallelMultiStepDecoder
      _compute_direct_multi_step_loss : simplified parallel training loss
      get_autoregressive_predictions  : direct prediction (no GRU rollout)

    All other behaviour (training loop, CPC/LIM, FiLM, CAETC, logging)
    is inherited from CausalMixerCPC without modification.

    Comparison target
    ─────────────────
    Run CMCP and CMCPD side-by-side with identical encoder configs (br_size,
    num_layer, seq_hidden_units, lambda_cpc, lambda_lim, etc.) to isolate the
    decoder contribution.  The cmcpd hparams config also tests a larger encoder
    (br_size=48, num_layer=4) to test Approach 2 simultaneously.
    """

    model_type = 'multi'
    possible_model_types = {'multi'}

    def __init__(
        self,
        args: DictConfig,
        dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
        autoregressive: bool = None,
        has_vitals: bool = None,
        projection_horizon: int = None,
        bce_weights: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(
            args, dataset_collection, autoregressive,
            has_vitals, projection_horizon, bce_weights, **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation — swap GRU for ParallelMultiStepDecoder
    # ─────────────────────────────────────────────────────────────────────────

    def _init_specific(self, sub_args: DictConfig):
        """
        Calls the full CMCP._init_specific (which calls CMC._init_specific,
        which calls CM._init_specific), then REPLACES self.direct_head
        (set to GRUMultiStepDecoder by CM._init_specific) with the
        ParallelMultiStepDecoder.

        Using the same attribute name (self.direct_head) means:
          - No changes needed in inherited forward / training_step code.
          - The CPC/LIM heads remain unchanged.
          - The CAETC heads remain unchanged.
          - Only the decoder object is swapped.
        """
        # Let the full parent chain initialise everything (including GRU)
        super()._init_specific(sub_args)

        if not hasattr(self, 'br_size') or self.br_size is None:
            return

        try:
            # ── Parallel decoder config ────────────────────────────────────
            trt_enc_dim = int(getattr(sub_args, 'trt_enc_dim', 32))
            dec_hidden  = int(getattr(sub_args, 'dec_hidden',  128))
            dec_dropout = float(getattr(sub_args, 'dec_dropout', 0.1))

            # REPLACE the GRU decoder that CM._init_specific just built.
            # The parent already set self.projection_horizon and
            # self.dim_treatments / self.dim_outcome.
            self.direct_head = ParallelMultiStepDecoder(
                br_size        = self.br_size,
                dim_treatments = self.dim_treatments,
                tau            = self.projection_horizon,
                trt_enc_dim    = trt_enc_dim,
                hidden         = dec_hidden,
                dim_outcome    = self.dim_outcome,
                dropout        = dec_dropout,
            )

            n_params = sum(p.numel() for p in self.direct_head.parameters())
            logger.info(
                f'CausalMixerCPCDirect: ParallelMultiStepDecoder initialised '
                f'(tau={self.projection_horizon}, trt_enc_dim={trt_enc_dim}, '
                f'hidden={dec_hidden}, params={n_params:,}). '
                f'GRUMultiStepDecoder replaced.'
            )

        except Exception as e:
            logger.warning(f'CausalMixerCPCDirect decoder init failed: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-step training loss — parallel version (no teacher forcing)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_direct_multi_step_loss(
        self,
        br: torch.Tensor,
        batch: dict,
        teacher_forcing_p: float = 0.0,   # ignored — kept for API compatibility
    ):
        """
        Parallel multi-step auxiliary loss for the ParallelMultiStepDecoder.

        Structural simplification vs the GRU version
        ──────────────────────────────────────────────
        • No teacher forcing: the parallel decoder has no autoregressive
          feedback loop, so teacher_forcing_p is structurally irrelevant.
          (The arg is accepted to keep the inherited training_step call
          signature compatible without any changes there.)

        • No FiLM seed alignment: the parallel decoder does not have a GRU
          hidden state that needs treatment-specific seeding — it conditions
          on the treatment sequence directly via the TreatmentMLP.  The FiLM
          alignment patch (in cm.py) is not needed here.

        • Same gradient isolation: br_last is detached before the decoder
          to prevent the multi-step loss from corrupting the backbone encoder
          (same rationale as the GRU version).

        • Same multi-split sampling: 8 random split points per batch, averaged.

        • Same per-step discount: controlled by ms_step_discount in config.

        Args
        ────
        br               : [B, T, br_size]  balanced representations (detach below)
        batch            : training batch dict
        teacher_forcing_p: ignored (API compatibility)

        Returns
        ───────
        Scalar loss averaged over valid splits, or None if sequence too short.
        """
        tau = self.projection_horizon
        if tau is None or tau <= 0:
            return None

        B, T, _ = br.shape
        if T <= tau:
            return None

        max_split = T - tau
        n_splits  = min(8, max_split)
        split_pts = torch.randperm(max_split)[:n_splits].tolist()

        total_loss = br.new_zeros(())
        n_valid    = 0

        gamma = float(getattr(self.hparams.exp, 'ms_step_discount', 1.0))
        if gamma < 1.0:
            # Pre-compute discount weights once per call.
            discounts = torch.tensor(
                [gamma ** k for k in range(tau)],
                dtype=br.dtype, device=br.device,
            )  # [τ]
        else:
            discounts = None

        for s in split_pts:
            # ── Gradient isolation ─────────────────────────────────────────
            # Detach the BR snapshot: multi-step loss trains only the decoder,
            # not the backbone.  λ_ms can safely be large (3.5+) without risk
            # of encoder/decoder gradient conflict.
            br_last    = br[:, s, :].detach()                           # [B, br_size]
            future_trt = batch['current_treatments'][:, s:s + tau, :]  # [B, τ, dim_t]
            future_out = batch['outputs'][:, s:s + tau, :]             # [B, τ, dim_o]
            future_mask= batch['active_entries'][:, s:s + tau, :]      # [B, τ, 1]

            # ── Random horizon curriculum ──────────────────────────────────
            # Sample τ' ∈ [2, τ] uniformly so all horizon depths get balanced
            # gradient signal.  Avoids over-specialising to short horizons.
            tau_prime = torch.randint(2, tau + 1, (1,)).item() if tau >= 3 else tau

            future_trt_h  = future_trt[:, :tau_prime, :]
            future_out_h  = future_out[:, :tau_prime, :]
            future_mask_h = future_mask[:, :tau_prime, :]

            if future_mask_h.sum() == 0:
                continue

            # ── Parallel prediction (no rollout) ───────────────────────────
            pred = self.direct_head(
                br_last, future_trt_h,
            )  # [B, τ', dim_o]

            mse = F.mse_loss(pred, future_out_h, reduce=False)  # [B, τ', dim_o]

            # ── Per-step discount (if gamma < 1.0) ────────────────────────
            if discounts is not None:
                mse = mse * discounts[:tau_prime].view(1, -1, 1)

            # Mask and average.
            loss_s = (future_mask_h * mse).sum() / future_mask_h.sum()
            total_loss = total_loss + loss_s
            n_valid   += 1

        return total_loss / max(n_valid, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference — direct parallel prediction (no GRU rollout)
    # ─────────────────────────────────────────────────────────────────────────

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        Direct τ-step counterfactual prediction via ParallelMultiStepDecoder.

        Replaces CMC's GRU rollout with a single parallel forward pass.
        No autoregression → no error accumulation → no teacher-forcing gap.

        For each sample i:
          1. Get BR at the last observed timestep (split - 1).
          2. Get the τ counterfactual treatments from [split, split + τ).
          3. Call ParallelMultiStepDecoder(BR_last, future_treatments).
          4. Store the τ predicted outcomes.

        Note: FiLM seeding (applied in CMC for the GRU's init_hidden) is NOT
        applied here.  The ParallelMultiStepDecoder conditions directly on the
        full treatment sequence via TreatmentMLP + cumulative context, which
        subsumes and extends what FiLM seeding provided to the GRU.
        """
        logger.info(
            f'CausalMixerCPCDirect: direct parallel prediction for '
            f'{dataset.subset_name} (no autoregressive rollout).'
        )
        tau = self.hparams.dataset.projection_horizon

        all_br = torch.tensor(self.get_representations(dataset))   # [N, T, br_size]
        future_treatments = torch.tensor(
            dataset.data['current_treatments']
        ).to(torch.get_default_dtype())   # [N, T_total, dim_treatments]

        splits = dataset.data['future_past_split']   # [N]
        predicted_outputs = np.zeros((len(dataset), tau, self.dim_outcome))

        with torch.no_grad():
            for i in range(len(dataset)):
                split = int(splits[i])

                # BR at the last observed timestep.
                br_last = all_br[i, split - 1, :].unsqueeze(0)   # [1, br_size]

                # Counterfactual treatment sequence for the τ-step window.
                fut_trt = future_treatments[
                    i, split:split + tau, :
                ].unsqueeze(0)   # [1, τ, dim_treatments]

                # Single parallel forward pass — no rollout needed.
                pred = self.direct_head(br_last, fut_trt)   # [1, τ, dim_outcome]

                predicted_outputs[i] = pred.squeeze(0).cpu().numpy()

        return predicted_outputs
