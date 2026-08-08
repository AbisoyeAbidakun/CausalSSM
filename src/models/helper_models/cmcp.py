"""CausalMixerCPC (CMCP) — CausalMixerCAETC + Contrastive Predictive Coding
                            + Local Deep InfoMax

Model hierarchy:
    CM   — MLP-Mixer backbone + GRU decoder
    CMC  — CM + FiLM conditioning + partial autoencoding + conditioning loss
    CMCP — CMC + CPC + Local InfoMax  ← this file

Two innovations are added on top of CMC (Nguyen et al. 2026):

  1. Contrastive Predictive Coding (CPC)
     ─────────────────────────────────────
     Motivation (NeurIPS 2024 — Causal Contrastive Learning for Counterfactual
     Regression Over Time, arxiv:2406.00535):
     The backbone is trained so that the balanced representation at timestep t
     can predict the BR at timestep t+k, for k = 1, …, K.  This is the
     multi-step temporal predictive objective that is missing from all prior
     causal inference models (CT, CRN, CMC).

     Architecture:
       • context_proj : BR → cpc_dim   (same projection for all k)
       • target_proj  : BR → cpc_dim   (separate projection for targets)
       • W_k : cpc_dim → cpc_dim       (K bilinear predictors, one per step)

     Loss (InfoNCE with in-batch negatives):
       For each anchor t and step k, the model predicts f_{t+k} = W_k(c_t)
       and must identify the TRUE future representation z_{t+k} among all B
       batch items' representations at t+k:

         L_CPC = -E_{t,k} log [ exp(f_{t+k} · z_{t+k}) /
                                 Σ_j exp(f_{t+k} · z^j_{t+k}) ]

     Why it beats CMC/CAETC:
       CAETC's partial autoencoding forces reconstruction of Y_t, A_t, X_t
       at the SAME timestep — it's a local constraint.  CPC forces the model
       to capture information that is predictive ACROSS multiple future steps.
       This directly trains the representation to support the GRU decoder's
       multi-step rollout, which is the current bottleneck (6-step RMSE
       degrades significantly vs. 1-step in CM/CMC).

  2. Local Deep InfoMax (LIM)
     ─────────────────────────
     Motivation (Hjelm et al. 2019 — Learning Deep Representations by Mutual
     Information Maximisation; AMDIM 2019):
     Maximises the mutual information between the local input embedding at t
     and the balanced representation at t:

         L_LIM = I(x_local_t ; BR_t)

     estimated via a bilinear InfoNCE critic:

         T(x, z) = x · W_im · z

     Positive pairs : (x_local_t, BR_t) from the same timestep and batch item
     Negative pairs : (x_local_t, BR_{t'}) where t' ≠ t (time-shuffled)

     Why it beats CMC/CAETC:
       CAETC enforces invertibility via RECONSTRUCTION (decoder heads).
       LIM enforces it variationally without decoder heads — it is a tighter
       lower bound on MI and does not require the decoder to be expressive
       enough to perfectly invert the encoder.  It also cannot be fooled by
       the encoder memorising a compressed reconstruction, which decoder-based
       methods can.  Critically, LIM explicitly counteracts the over-erasure
       of domain-confusion training by penalising any drop in MI between inputs
       and representations, so covariate information is always preserved.

Training loss structure
───────────────────────
  L = L_factual     (MSE outcome, from CM)
    + L_domain      (BCE domain confusion × α, from CM)
    + λ_ms    × L_multistep   (GRU multi-step, from CM)
    + λ_recon × L_recon       (partial autoencoding, from CMC)
    + λ_cond  × L_cond        (FiLM conditioning, from CMC)
    + λ_cpc   × L_CPC         (contrastive prediction, NEW)
    + λ_lim   × L_LIM         (local InfoMax, NEW)

Usage
─────
  python train.py +backbone=cmcp \\
      "+backbone/cmcp_hparams/cancer_sim_domain_conf='1'" \\
      dataset=cancer_sim_basic exp.balancing=domain_confusion

References
──────────
  [1] CPC: Oord et al. 2018 — Representation Learning with Contrastive
      Predictive Coding. arXiv:1807.03748
  [2] DIM/InfoMax: Hjelm et al. 2019 — Learning Deep Representations by
      Mutual Information Maximisation. ICLR 2019.
  [3] Causal Contrastive Learning: NeurIPS 2024, arXiv:2406.00535
  [4] CAETC / CMC: Nguyen et al. 2026
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
from src.models.helper_models.cmc import CausalMixerCAETC

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Component 1 — Contrastive Predictive Coding Head
# ═══════════════════════════════════════════════════════════════════════════════

class CPCHead(nn.Module):
    """
    Contrastive Predictive Coding (CPC) head for temporal self-supervision.

    Forces the balanced representation at timestep t to be predictive of the
    representation at timestep t+k for k = 1, …, K.

    Architecture
    ────────────
      context_proj : br_size → cpc_dim   (projects anchor BR to CPC space)
      target_proj  : br_size → cpc_dim   (projects future BR to CPC space)
      W_k          : cpc_dim → cpc_dim   (K bilinear predictors, k=1..K)

    InfoNCE loss with in-batch negatives
    ─────────────────────────────────────
    For each sampled anchor t and prediction step k:
      • Compute prediction: f = W_k(context_proj(BR_t))        [B, cpc_dim]
      • Compute targets   : z = target_proj(BR_{t+k})          [B, cpc_dim]
      • Score matrix      : S = f · z^T                        [B, B]
      • Positive on diagonal; all off-diagonal entries negative
      • InfoNCE = -mean log softmax(S)[diagonal]

    Negative sampling: all B batch items' future representations at t+k are
    used as negatives.  This is efficient and requires no extra forward passes.

    Efficiency: n_anchors random timesteps are sampled per batch (default 8),
    keeping cost to O(B² × K × n_anchors) regardless of sequence length T.

    Parameters
    ----------
    br_size    : balanced representation size (input dim)
    cpc_dim    : CPC projection dimension (typically br_size or br_size // 2)
    K          : number of prediction steps (1 to K)
    n_anchors  : random anchor steps to sample per batch
    """

    def __init__(
        self,
        br_size: int,
        cpc_dim: int,
        K: int = 3,
        n_anchors: int = 8,
        dim_treatments: int = 0,
    ):
        super().__init__()
        self.K = K
        self.cpc_dim = cpc_dim
        self.n_anchors = n_anchors
        self.dim_treatments = dim_treatments

        # Context and target projections — LayerNorm keeps projections
        # on a consistent scale so temperature is implicitly controlled.
        self.context_proj = nn.Sequential(
            nn.Linear(br_size, cpc_dim),
            nn.LayerNorm(cpc_dim),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(br_size, cpc_dim),
            nn.LayerNorm(cpc_dim),
        )

        # One bilinear predictor per step k.  Each W_k is a square matrix;
        # bias=False keeps the space symmetric around zero, which stabilises
        # the InfoNCE score distribution.
        self.W_k = nn.ModuleList([
            nn.Linear(cpc_dim, cpc_dim, bias=False)
            for _ in range(K)
        ])

        # Initialise W_k as near-identity so the loss starts at a well-
        # conditioned value (if W_k = I then S = f · z^T which is already
        # a meaningful cosine-like score).
        for W in self.W_k:
            nn.init.eye_(W.weight)

        # ── Treatment-conditioned CPC (improvement) ───────────────────────────
        # Problem with treatment-unaware CPC: in cancer_sim, BR_{t+k} depends
        # heavily on which treatments were given in [t, t+k).  An unaware
        # predictor must average over all treatment paths, making the task
        # easier but less aligned with counterfactual prediction.
        #
        # Fix: add one linear treatment projection per step k.  The mean
        # treatment over the [t, t+k) window is projected into cpc_dim and
        # ADDED to the context before W_k — the predictor now answers
        # "given this context AND these intervening treatments, what will BR
        # look like k steps ahead?"  This is exactly the counterfactual
        # structure the model must generalise at inference.
        #
        # dim_treatments=0 disables this path (backward-compatible with CM).
        if dim_treatments > 0:
            self.trt_proj_k = nn.ModuleList([
                nn.Linear(dim_treatments, cpc_dim, bias=False)
                for _ in range(K)
            ])
        else:
            self.trt_proj_k = None

    def compute_loss(
        self,
        br: torch.Tensor,                      # [B, T, br_size]
        active_entries: torch.Tensor,           # [B, T, 1]
        treatments: torch.Tensor = None,        # [B, T, dim_treatments]  optional
    ) -> torch.Tensor:
        """
        Compute the treatment-conditioned CPC InfoNCE loss.

        If `treatments` is provided and dim_treatments > 0, the bilinear
        predictor is additionally conditioned on the mean treatment over the
        k-step prediction window, making CPC treatment-specific.

        Returns a scalar averaged over K steps and sampled anchors.
        """
        B, T, _ = br.shape
        device = br.device

        # Project the full BR sequence once.
        z_c = self.context_proj(br)   # [B, T, cpc_dim]
        z_t = self.target_proj(br)    # [B, T, cpc_dim]

        total_loss = br.new_zeros(())
        n_valid_steps = 0

        for k_idx in range(self.K):
            k = k_idx + 1           # prediction step (1-indexed)
            max_anchor = T - k
            if max_anchor <= 0:
                continue

            # Sample n_anchors random valid anchor timesteps.
            n = min(self.n_anchors, max_anchor)
            anchor_times = torch.randperm(max_anchor, device=device)[:n]

            step_loss = br.new_zeros(())
            n_valid_anchors = 0

            for t in anchor_times.tolist():
                # Mask check: both anchor and future must be active in
                # at least one batch item to form a valid pair.
                anchor_mask = active_entries[:, t,   0]   # [B]  binary
                future_mask = active_entries[:, t+k, 0]   # [B]  binary
                valid_mask  = (anchor_mask * future_mask)  # [B]  binary

                if valid_mask.sum() < 2:
                    # Need at least 2 valid samples for in-batch negatives.
                    continue

                # Bilinear prediction: W_k transforms context → future space.
                c_t  = z_c[:, t,   :]        # [B, cpc_dim]  context at t
                f_tk = z_t[:, t+k, :]        # [B, cpc_dim]  future at t+k

                # ── Treatment conditioning ────────────────────────────────────
                # Mean treatment over the k-step prediction window [t, t+k).
                # Projected into cpc_dim and added to the context so W_k
                # predicts the treatment-specific future, not the marginal.
                if (self.trt_proj_k is not None
                        and treatments is not None
                        and t + k <= T):
                    trt_window = treatments[:, t:t + k, :]  # [B, k, dim_t]
                    trt_mean   = trt_window.mean(dim=1)      # [B, dim_t]
                    c_t = c_t + self.trt_proj_k[k_idx](trt_mean)  # [B, cpc_dim]

                pred = self.W_k[k_idx](c_t)  # [B, cpc_dim]  predicted future

                # Restrict to valid batch items.
                idx   = valid_mask.nonzero(as_tuple=True)[0]   # [N]
                pred  = pred[idx]    # [N, cpc_dim]
                f_tk  = f_tk[idx]    # [N, cpc_dim]

                N = idx.size(0)
                if N < 2:
                    continue

                # [N, N] score matrix: row i predicts future of item i,
                # column j is the actual future of item j.
                # Positive = diagonal; negatives = off-diagonal.
                scores = torch.matmul(pred, f_tk.T)    # [N, N]

                # Divide by sqrt(cpc_dim) — acts as temperature=1 normalisation,
                # keeping logit scale stable as cpc_dim grows.
                scores = scores / (self.cpc_dim ** 0.5)

                labels = torch.arange(N, device=device)
                step_loss = step_loss + F.cross_entropy(scores, labels)
                n_valid_anchors += 1

            if n_valid_anchors > 0:
                total_loss = total_loss + step_loss / n_valid_anchors
                n_valid_steps += 1

        return total_loss / max(n_valid_steps, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2 — Local Deep InfoMax Head
# ═══════════════════════════════════════════════════════════════════════════════

class LocalInfoMaxHead(nn.Module):
    """
    Local Deep InfoMax (LIM) head for representation invertibility.

    Maximises I(x_local_t ; BR_t) — the mutual information between the
    pre-mixer local input embedding at timestep t and the balanced
    representation at the same timestep.  This prevents the adversarial
    domain-confusion training from stripping covariate information from BR.

    Estimator: InfoNCE (a lower bound on MI; tighter than JSD in practice)
    ─────────────────────────────────────────────────────────────────────────
      Positive pairs : (x_local_t, BR_t)  same batch item, same timestep
      Negative pairs : (x_local_t, BR_t') same batch item, DIFFERENT timestep
                       (time-shuffle negatives within each sequence)

    Score function : bilinear T(x, z) = x_proj(x) · z_proj(z)^T

    Why time-shuffle negatives?
      Domain-confusion strips the TEMPORAL content from BR (it pushes BR
      to look the same regardless of treatment history).  Time-shuffle
      negatives specifically target this: if BR_t and BR_{t'} are identical,
      the model cannot score (x_t, BR_t) higher than (x_t, BR_{t'}), and
      the InfoNCE loss cannot be minimised.  This forces the BR to retain
      timestep-specific information.

    Parameters
    ----------
    input_dim   : dimension of local input embedding (= seq_hidden_units)
    br_size     : balanced representation size
    lim_dim     : projection dimension for the bilinear critic
    n_negs      : number of negative timesteps to sample per anchor
    """

    def __init__(
        self,
        input_dim: int,
        br_size: int,
        lim_dim: int,
        n_negs: int = 8,
    ):
        super().__init__()
        self.lim_dim = lim_dim
        self.n_negs  = n_negs

        # Project local input → critic space.
        self.x_proj = nn.Sequential(
            nn.Linear(input_dim, lim_dim),
            nn.LayerNorm(lim_dim),
        )
        # Project BR → critic space.
        self.z_proj = nn.Sequential(
            nn.Linear(br_size, lim_dim),
            nn.LayerNorm(lim_dim),
        )

    def compute_loss(
        self,
        x_local: torch.Tensor,       # [B, T, input_dim]
        br: torch.Tensor,             # [B, T, br_size]
        active_entries: torch.Tensor, # [B, T, 1]
    ) -> torch.Tensor:
        """
        Compute the Local InfoMax InfoNCE loss.

        For each valid anchor timestep t, TWO complementary InfoNCE terms are
        averaged to produce a richer negative set:

          Term 1 — Cross-patient (in-batch) negatives:
            • Positive : (x_local_{i,t}, BR_{i,t})  same patient i, same t
            • Negatives: (x_local_{i,t}, BR_{j,t})  different patient j, same t
            Score matrix [N, N]; positive = diagonal.
            Forces the encoder to retain patient-specific covariate information
            at each timestep — patients with different covariate histories must
            have distinguishable BRs even at the same calendar time.

          Term 2 — Time-shuffle (within-patient) negatives (existing):
            • Positive : (x_local_{i,t}, BR_{i,t})   same patient, same t
            • Negatives: (x_local_{i,t}, BR_{i,t'})  same patient, different t
            Forces the encoder to retain timestep-specific temporal dynamics.

        The two terms address orthogonal failure modes:
          - Domain confusion may strip patient identity  → Term 1 catches this
          - Domain confusion may strip temporal content  → Term 2 catches this
        Averaging them gives a balanced gradient that protects both dimensions.

        Returns a scalar loss.
        """
        B, T, _ = x_local.shape
        device   = x_local.device

        # Project inputs and representations.
        px = self.x_proj(x_local)   # [B, T, lim_dim]
        pz = self.z_proj(br)        # [B, T, lim_dim]

        mask = active_entries.squeeze(-1)  # [B, T] binary

        # Sample n_anchors anchor timesteps (same budget as CPC).
        n_anchors = min(self.n_negs, T)
        anchor_times = torch.randperm(T, device=device)[:n_anchors].tolist()

        total_loss  = x_local.new_zeros(())
        n_valid_t   = 0

        for t in anchor_times:
            anchor_valid = mask[:, t]              # [B] binary
            if anchor_valid.sum() < 2:
                continue

            idx = anchor_valid.nonzero(as_tuple=True)[0]  # [N]
            N   = idx.size(0)

            # Anchor features and positive representations.
            x_t  = px[idx, t, :]    # [N, lim_dim]  local input at t
            z_t  = pz[idx, t, :]    # [N, lim_dim]  BR at t (positive)

            anchor_loss = x_local.new_zeros(())
            n_terms = 0

            # ── Term 1: Cross-patient in-batch negatives ─────────────────────
            # Score matrix S_ij = x_t_i · z_t_j / sqrt(d).
            # Positive = diagonal; all N-1 off-diagonal entries = negatives.
            # This is identical to CPC's in-batch scoring but applied at the
            # input→BR level rather than BR→future-BR level.
            if N >= 2:
                scores_cross = torch.matmul(x_t, z_t.T) / (self.lim_dim ** 0.5)  # [N, N]
                labels_cross = torch.arange(N, device=device)
                anchor_loss  = anchor_loss + F.cross_entropy(scores_cross, labels_cross)
                n_terms += 1

            # ── Term 2: Time-shuffle within-patient negatives (existing) ─────
            # Positive at index 0; n_negs time-shuffled BRs from same patient.
            neg_pool = [s for s in range(T) if s != t]
            n_sample = min(self.n_negs, len(neg_pool))
            neg_times = torch.randperm(len(neg_pool), device=device)[:n_sample].tolist()
            neg_t_list = [neg_pool[i] for i in neg_times]

            if neg_t_list:
                z_negs = torch.stack(
                    [pz[idx, nt, :] for nt in neg_t_list], dim=1
                )  # [N, n_negs, lim_dim]

                # z_all: [N, 1 + n_negs, lim_dim] — positive first
                z_all = torch.cat([z_t.unsqueeze(1), z_negs], dim=1)

                scores_time = torch.bmm(
                    x_t.unsqueeze(1),         # [N, 1, lim_dim]
                    z_all.transpose(1, 2),    # [N, lim_dim, 1 + n_negs]
                ).squeeze(1) / (self.lim_dim ** 0.5)  # [N, 1 + n_negs]

                labels_time = torch.zeros(N, dtype=torch.long, device=device)
                anchor_loss = anchor_loss + F.cross_entropy(scores_time, labels_time)
                n_terms += 1

            if n_terms > 0:
                total_loss = total_loss + anchor_loss / n_terms
                n_valid_t  += 1

        return total_loss / max(n_valid_t, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# CausalMixerCPC — Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class CausalMixerCPC(CausalMixerCAETC):
    """
    CausalMixerCPC (CMCP) — the full model combining all innovations.

    Inherits from CausalMixerCAETC (CMC):
      ✓ CausalMixerBlock backbone  (O(T) complexity)
      ✓ GRU autoregressive decoder (multi-step counterfactual rollout)
      ✓ LearnablePositionalEncoding (per stream)
      ✓ FiLM treatment conditioning (residual scale+shift of BR)
      ✓ PartialAutoencoderHeads     (F^A, F^Y, F^X reconstruction)
      ✓ Treatment conditioning loss (counterfactual FiLM supervision)

    Adds:
      + CPCHead        (temporal contrastive predictive coding)
      + LocalInfoMaxHead (local mutual information maximisation)
      + build_br override: captures pre-mixer input embeddings (_x_local)
        for the InfoMax loss without an extra forward pass.

    Training loss:
      L = L_factual + L_domain
        + λ_ms    × L_multistep
        + λ_recon × L_recon
        + λ_cond  × L_cond
        + λ_cpc   × L_CPC      ← CPC contrastive prediction
        + λ_lim   × L_LIM      ← local InfoMax MI maximisation

    Recommended starting weights (cancer_sim_domain_conf/1):
      λ_ms=0.5, λ_recon=0.1, λ_cond=0.05, λ_cpc=0.1, λ_lim=0.05
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
        # _x_local will be populated inside build_br during each forward pass.
        self._x_local = None

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation — adds CPC + InfoMax on top of CMC
    # ─────────────────────────────────────────────────────────────────────────

    def _init_specific(self, sub_args: DictConfig):
        """Calls CMC._init_specific then appends CPC and InfoMax heads."""
        super()._init_specific(sub_args)

        if not hasattr(self, 'br_size') or self.br_size is None:
            return

        try:
            d = self.seq_hidden_units

            # ── CPC head ──────────────────────────────────────────────────────
            # CPC projection dimension: same as br_size by default.
            # Reducing to br_size // 2 also works if you need fewer params.
            cpc_dim   = self.br_size
            K_steps   = int(getattr(sub_args, 'cpc_k_steps',   3))
            n_anchors = int(getattr(sub_args, 'cpc_n_anchors', 8))

            self.cpc_head = CPCHead(
                br_size        = self.br_size,
                cpc_dim        = cpc_dim,
                K              = K_steps,
                n_anchors      = n_anchors,
                dim_treatments = self.dim_treatments,
            )

            # ── Local InfoMax head ────────────────────────────────────────────
            # Input to InfoMax is the pre-mixer average of stream embeddings.
            # Each stream is projected to d = seq_hidden_units, so x_local ∈ ℝ^d.
            lim_dim = self.br_size
            n_negs  = int(getattr(sub_args, 'lim_n_negs', 8))

            self.lim_head = LocalInfoMaxHead(
                input_dim = d,
                br_size   = self.br_size,
                lim_dim   = lim_dim,
                n_negs    = n_negs,
            )

            logger.info(
                f'CausalMixerCPC: CPCHead(K={K_steps}, n_anchors={n_anchors}) + '
                f'LocalInfoMaxHead(lim_dim={lim_dim}, n_negs={n_negs}) initialised.'
            )

        except Exception as e:
            logger.warning(f'CausalMixerCPC extra init failed: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # build_br override — captures pre-mixer input embeddings for InfoMax
    # ─────────────────────────────────────────────────────────────────────────

    def build_br(
        self,
        prev_treatments,
        vitals,
        prev_outputs,
        static_features,
        active_entries,
        fixed_split=None,
    ):
        """
        Overrides CM.build_br to additionally store self._x_local.

        self._x_local [B, T, d] is the mean of the stream embeddings
        AFTER static injection and positional encoding but BEFORE the mixer
        blocks.  It is used by LocalInfoMaxHead as the local input signal
        whose MI with BR is maximised.

        The BR computation itself is identical to the parent class.
        """
        import torch
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:
            for i in range(len(active_entries)):
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        # ── Input projections ─────────────────────────────────────────────────
        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None

        # ── Static feature injection (gated) ─────────────────────────────────
        x_s = self.static_input_transformation(static_features.unsqueeze(1))
        x_t = x_t + torch.sigmoid(self.gate_static_t) * x_s
        x_o = x_o + torch.sigmoid(self.gate_static_o) * x_s
        if x_v is not None:
            x_v = x_v + torch.sigmoid(self.gate_static_v) * x_s

        # ── Positional encodings ──────────────────────────────────────────────
        x_t = self.pos_enc_t(x_t)
        x_o = self.pos_enc_o(x_o)
        if x_v is not None:
            x_v = self.pos_enc_v(x_v)

        # ── Capture pre-mixer average for InfoMax ─────────────────────────────
        # Computed BEFORE the mixer blocks so it represents the raw local
        # information available at each timestep, not the globally-mixed BR.
        if x_v is not None:
            self._x_local = (x_t + x_o + x_v) / 3   # [B, T, d]
        else:
            self._x_local = (x_t + x_o) / 2          # [B, T, d]

        # Detach from graph: InfoMax should maximise MI between the FIXED
        # input representation and the learned BR, not backprop through
        # the input projection again (which would create a trivial collapse
        # where both projections learn the same thing).
        self._x_local = self._x_local.detach()

        # ── N × CausalMixerBlock (unchanged from parent) ─────────────────────
        for block in self.mixer_blocks:
            if self.has_vitals:
                x_t, x_o, x_v = block(x_t, x_o, x_v)
            else:
                x_t, x_o = block(x_t, x_o)

        # ── Stream pooling (unchanged from parent) ────────────────────────────
        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    sp = int(fixed_split[i])
                    x[i, :sp] = (x_o[i, :sp] + x_t[i, :sp] + x_v[i, :sp]) / 3
                    x[i, sp:] = (x_o[i, sp:] + x_t[i, sp:]) / 2
            else:
                x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        br     = self.br_treatment_outcome_head.build_br(output)
        return br

    # ─────────────────────────────────────────────────────────────────────────
    # Training step — extends CMC with CPC + InfoMax losses
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        Full training step with all losses:
          CM    : L_factual + L_domain
          CMC   : + λ_ms × L_ms + λ_recon × L_recon + λ_cond × L_cond
          CMCP  : + λ_cpc × L_CPC + λ_lim × L_LIM
        """
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:  # representation + outcome update

            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
            else:
                treatment_pred, outcome_pred, br = self(batch)

            # self._x_local was set inside build_br during self(batch) above.
            x_local = self._x_local  # [B, T, d]  (detached)

            # ── Factual MSE loss ──────────────────────────────────────────────
            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

            # ── Domain balancing loss ─────────────────────────────────────────
            if self.balancing == 'grad_reverse':
                bce_loss = self.bce_loss(
                    treatment_pred, batch['current_treatments'].to(torch.get_default_dtype()), kind='predict'
                )
            elif self.balancing == 'domain_confusion':
                bce_loss = self.bce_loss(
                    treatment_pred, batch['current_treatments'].to(torch.get_default_dtype()), kind='confuse'
                )
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            else:
                raise NotImplementedError()

            bce_loss = (
                batch['active_entries'].squeeze(-1) * bce_loss
            ).sum() / batch['active_entries'].sum()
            mse_loss = (
                batch['active_entries'] * mse_loss
            ).sum() / batch['active_entries'].sum()

            loss = bce_loss + mse_loss

            # ── GRU multi-step auxiliary loss (from CM, with scheduled TF) ─────
            # Bug fix: previously called _compute_direct_multi_step_loss without
            # teacher_forcing_p, so the GRU trained at a fixed p=0.5 for all 300
            # epochs — never closing the train/inference gap (inference is p=0).
            # Now mirrors CM's training_step: anneal p from tf_init→tf_min each
            # epoch so the GRU is progressively exposed to its own predictions.
            lambda_ms = float(getattr(self.hparams.exp, 'lambda_ms', 0.2))
            if lambda_ms > 0.0:
                tf_p    = self._get_teacher_forcing_p(self.current_epoch)
                ms_loss = self._compute_direct_multi_step_loss(
                    br, batch, teacher_forcing_p=tf_p
                )
                if ms_loss is not None:
                    loss = loss + lambda_ms * ms_loss
                    self.log(f'{self.model_type}_train_ms_loss', ms_loss,
                             on_epoch=True, on_step=False, sync_dist=True)
                    self.log(f'{self.model_type}_train_tf_p', tf_p,
                             on_epoch=True, on_step=False, sync_dist=True)

            # ── CAETC: partial autoencoding loss (from CMC) ───────────────────
            lambda_recon = float(getattr(self.hparams.exp, 'lambda_recon', 0.1))
            if lambda_recon > 0.0 and hasattr(self, 'autoencoder_heads'):
                vitals   = batch.get('vitals', None) if self.has_vitals else None
                delta_a  = float(getattr(self.hparams.exp, 'delta_a',  0.1))
                delta_x  = float(getattr(self.hparams.exp, 'delta_x',  0.1))
                recon_loss = self.autoencoder_heads.compute_reconstruction_loss(
                    br=br,
                    treatments=batch['current_treatments'],
                    outcomes=batch['outputs'],
                    active_entries=batch['active_entries'],
                    vitals=vitals,
                    delta_a=delta_a,
                    delta_x=delta_x,
                )
                loss = loss + lambda_recon * recon_loss
                self.log(f'{self.model_type}_train_recon_loss', recon_loss,
                         on_epoch=True, on_step=False, sync_dist=True)

            # ── CAETC: FiLM conditioning loss (from CMC) ──────────────────────
            lambda_cond = float(getattr(self.hparams.exp, 'lambda_cond', 0.05))
            if lambda_cond > 0.0 and hasattr(self, 'film_layer'):
                n_cf_arms    = int(getattr(self.hparams.exp, 'n_cf_arms',    4))
                label_smooth = float(getattr(self.hparams.exp, 'label_smooth', 0.1))
                cond_loss = self._compute_conditioning_loss(
                    br=br,
                    treatments=batch['current_treatments'],
                    active_entries=batch['active_entries'],
                    n_cf_arms=n_cf_arms,
                    label_smooth=label_smooth,
                )
                loss = loss + lambda_cond * cond_loss
                self.log(f'{self.model_type}_train_cond_loss', cond_loss,
                         on_epoch=True, on_step=False, sync_dist=True)

            # ── Contrastive warmup ────────────────────────────────────────────
            # At epoch 0, the CPC and LIM heads are randomly initialised and
            # their gradients are noisy — adding them at full weight early on
            # can destabilise the factual MSE loss before the encoder has
            # converged.  Linear warm-up ramps the contrastive signal in
            # gradually so the encoder sees a clean factual loss for the
            # critical early phase.
            #
            # warmup_epochs is configurable (default 100):
            #   epoch 0         → warmup=0.0  (contrastive losses fully off)
            #   epoch warmup/2  → warmup=0.5  (half strength)
            #   epoch warmup    → warmup=1.0  (full lambda_cpc / lambda_lim)
            #
            # Why 100 rather than 50:
            #   When cross-patient LIM negatives are active, the InfoNCE loss
            #   starts at log(B)≈4.85 rather than log(n_negs+1)≈2.2.  A faster
            #   ramp (50 epochs) causes the full-strength gradient to arrive
            #   before the factual encoder has converged, degrading factual RMSE.
            #   100 epochs (≈29% of 350) gives the encoder time to stabilise.
            warmup_epochs = int(getattr(self.hparams.exp, 'warmup_epochs', 100))
            warmup = min(1.0, self.current_epoch / max(warmup_epochs, 1))

            # ── CMCP: CPC temporal contrastive loss ───────────────────────────
            # CPC forces the BR at t to be predictive of BR at t+k.
            # This trains the backbone to capture multi-step dynamics that the
            # GRU decoder can exploit — directly targeting n-step RMSE.
            # Treatment conditioning is active when dim_treatments > 0:
            #   the mean treatment over [t, t+k) is projected into cpc_dim and
            #   added to the context before W_k, so predictions are
            #   treatment-specific rather than marginalised over arms.
            lambda_cpc = float(getattr(self.hparams.exp, 'lambda_cpc', 0.1))
            if lambda_cpc > 0.0 and warmup > 0.0 and hasattr(self, 'cpc_head'):
                cpc_loss = self.cpc_head.compute_loss(
                    br,
                    batch['active_entries'],
                    treatments=batch['current_treatments'],
                )
                loss = loss + warmup * lambda_cpc * cpc_loss
                self.log(f'{self.model_type}_train_cpc_loss', cpc_loss,
                         on_epoch=True, on_step=False, sync_dist=True)

            # ── CMCP: Local InfoMax MI maximisation ───────────────────────────
            # Maximises I(x_local_t; BR_t) to counteract the MI-reducing effect
            # of domain-confusion adversarial training.  Prevents BR from
            # dropping covariate information needed for accurate outcomes.
            # Uses pre-mixer embeddings (x_local) captured inside build_br.
            # Cross-patient negatives are now included alongside time-shuffle
            # negatives, protecting both patient-identity and temporal content.
            lambda_lim = float(getattr(self.hparams.exp, 'lambda_lim', 0.05))
            if (lambda_lim > 0.0
                    and warmup > 0.0
                    and hasattr(self, 'lim_head')
                    and x_local is not None):
                lim_loss = self.lim_head.compute_loss(
                    x_local, br, batch['active_entries']
                )
                loss = loss + warmup * lambda_lim * lim_loss
                self.log(f'{self.model_type}_train_lim_loss', lim_loss,
                         on_epoch=True, on_step=False, sync_dist=True)

            self.log(f'{self.model_type}_train_contrastive_warmup', warmup,
                     on_epoch=True, on_step=False, sync_dist=True)

            self.log(f'{self.model_type}_train_loss',     loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha',
                     self.br_treatment_outcome_head.alpha,
                     on_epoch=True, on_step=False, sync_dist=True)
            return loss

        elif optimizer_idx == 1:  # domain-classifier update (unchanged)
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

            bce_loss = self.bce_loss(
                treatment_pred, batch['current_treatments'].to(torch.get_default_dtype()), kind='predict'
            )
            if self.balancing == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss

            bce_loss = (
                batch['active_entries'].squeeze(-1) * bce_loss
            ).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            return bce_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Inference — inherited from CMC (FiLM-conditioned GRU seed)
    # ─────────────────────────────────────────────────────────────────────────
    # get_autoregressive_predictions is inherited from CausalMixerCAETC.
    # It already seeds the GRU with FiLM(BR_last, first_cf_treatment),
    # which benefits directly from the CPC-trained representation quality.

    # ─────────────────────────────────────────────────────────────────────────
    # Hyperparameter search interface (inherited from CM via CMC)
    # ─────────────────────────────────────────────────────────────────────────
    # set_hparams is inherited unchanged from CausalMixer.
