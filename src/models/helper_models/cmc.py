"""CausalMixerA (CMC) — CausalMixer augmented with three innovations

  1. FiLM Treatment Conditioning
     ──────────────────────────────
     Instead of naively concatenating the balanced representation BR with the
     current treatment A_t before the outcome head, we apply Feature-wise
     Linear Modulation (FiLM):

         F^C(Φ, a) = Φ ⊙ (1 + R^ξ(a)) + R^β(a)

     R^ξ and R^β are linear projections from treatment space to BR space.
     The residual formulation (1 + scale) starts as identity and learns
     treatment-specific gain and bias, giving far more expressive power than
     concatenation for tailoring outcome predictions per treatment arm.

  2. Partial Autoencoding
     ─────────────────────
     The balanced representation is additionally asked to reconstruct the
     inputs it was built from:

         L^R = δ_Y · MSE(F^Y(BR_t), Y_t)
             + δ_A · BCE(F^A(BR_t), A_t)
             + δ_X · MSE(F^X(BR_t), X_t)   [if vitals present]

     Domain-confusion adversarial training can strip covariate information
     from BR, causing representation collapse (Huang et al. 2024).  The
     autoencoding loss counteracts this by requiring BR to retain enough
     information to reconstruct each input stream.

  3. Treatment-Specific Conditioning Loss
     ──────────────────────────────────────
     The FiLM layer is optimised for ALL treatment arms, not just the observed
     one.  For each timestep t we generate counterfactual treatment vectors
     a^c, apply FiLM, and enforce that F^A correctly reconstructs a^c:

         L^C = E_{a^c} [ BCE(F^A(F^C(BR_t, a^c)), a^c) ]

     This prevents FiLM from collapsing to identity and forces it to produce
     arm-distinguishable BR modulations even for unseen counterfactuals.

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
from src.models.helper_models.cm import CausalMixer, GRUMultiStepDecoder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Component 1 — FiLM Treatment Conditioning Layer
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMConditioningLayer(nn.Module):
    """Feature-wise Linear Modulation for treatment conditioning.

    Implements F^C from CAETC (Table 7c):

        F^C(Φ, a) = Φ ⊙ ξ(a) + β(a)

    with a residual formulation so the module initialises as identity:

        ξ(a) = 1 + R^ξ(a),   β(a) = R^β(a)

    where R^ξ, R^β : ℝ^{dim_treatments} → ℝ^{br_size} are linear projections
    initialised to near-zero so the initial forward pass is approximately
    Φ ⊙ 1 + 0 = Φ (identity), giving stable early training.

    Parameters
    ----------
    br_size        : size of the balanced representation (input/output dim)
    dim_treatments : dimensionality of the treatment vector
    """

    def __init__(self, br_size: int, dim_treatments: int):
        super().__init__()

        # Scale projection — maps treatment vector → per-feature scale residual.
        # Small initialisation ensures |R^ξ(a)| ≈ 0 at init → ξ ≈ 1.
        self.scale_proj = nn.Linear(dim_treatments, br_size, bias=True)
        nn.init.normal_(self.scale_proj.weight, std=0.01)
        nn.init.zeros_(self.scale_proj.bias)

        # Shift projection — maps treatment vector → per-feature shift.
        # Initialised to exactly zero → β ≈ 0 at init.
        self.shift_proj = nn.Linear(dim_treatments, br_size, bias=False)
        nn.init.zeros_(self.shift_proj.weight)

    def forward(
        self,
        br: torch.Tensor,          # [..., br_size]
        treatment: torch.Tensor,   # [..., dim_treatments]
    ) -> torch.Tensor:
        """
        Args:
            br        : balanced representation [..., br_size]
            treatment : treatment vector [..., dim_treatments]
        Returns:
            FiLM-conditioned representation [..., br_size]
        """
        # Residual scale and shift — same shape as br.
        scale = 1.0 + self.scale_proj(treatment)   # [..., br_size]
        shift = self.shift_proj(treatment)          # [..., br_size]
        return br * scale + shift


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2 — Partial Autoencoder Heads
# ═══════════════════════════════════════════════════════════════════════════════

class PartialAutoencoderHeads(nn.Module):
    """Lightweight reconstruction heads for partial autoencoding regularisation.

    Implements F^A, F^Y, F^X from CAETC (Table 7a-b):

        F^A : BR → A_t  (treatment reconstructor) — Linear→ELU→Linear→Sigmoid
        F^Y : BR → Y_t  (outcome  reconstructor)  — Linear→ELU→Linear
        F^X : BR → X_t  (vitals  reconstructor)   — Linear→ELU→Linear  [optional]

    The hidden layer width defaults to half the BR size (CAETC uses a small
    two-layer MLP) — large enough to reconstruct inputs, small enough not to
    compete with the outcome head for representational capacity.

    Reconstruction loss:
        L^R = δ_Y · MSE(F^Y(BR), Y)
            + δ_A · BCE(F^A(BR), A)          ← binary treatments
            + δ_X · MSE(F^X(BR), X)          [if has_vitals]

    Parameters
    ----------
    br_size        : balanced representation size
    dim_treatments : treatment dimension
    dim_outcome    : outcome dimension
    dim_vitals     : vitals dimension (0 = no vitals head created)
    hidden_ratio   : hidden dim = max(br_size * hidden_ratio, 16)
    """

    def __init__(
        self,
        br_size: int,
        dim_treatments: int,
        dim_outcome: int,
        dim_vitals: int = 0,
        hidden_ratio: float = 0.5,
    ):
        super().__init__()
        hidden = max(int(br_size * hidden_ratio), 16)

        # F^A — treatment reconstructor
        # Sigmoid at output: each treatment dimension is binary in our datasets.
        self.f_a = nn.Sequential(
            nn.Linear(br_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim_treatments),
            nn.Sigmoid(),
        )

        # F^Y — outcome reconstructor
        self.f_y = nn.Sequential(
            nn.Linear(br_size, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim_outcome),
        )

        # F^X — vitals reconstructor (optional)
        self.f_x = (
            nn.Sequential(
                nn.Linear(br_size, hidden),
                nn.ELU(),
                nn.Linear(hidden, dim_vitals),
            )
            if dim_vitals > 0 else None
        )

    def compute_reconstruction_loss(
        self,
        br: torch.Tensor,             # [B, T, br_size]
        treatments: torch.Tensor,     # [B, T, dim_t]
        outcomes: torch.Tensor,       # [B, T, dim_o]
        active_entries: torch.Tensor, # [B, T, 1]
        vitals: torch.Tensor = None,  # [B, T, dim_v]  or None
        delta_a: float = 0.1,
        delta_x: float = 0.1,
        delta_y: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the partial-autoencoding reconstruction loss.

        Returns a scalar loss averaged over active timesteps.
        """
        mask = active_entries  # [B, T, 1]
        n_active = mask.sum().clamp(min=1.0)

        # F^Y outcome reconstruction loss
        y_pred = self.f_y(br)                                        # [B, T, dim_o]
        loss_y = delta_y * (mask * F.mse_loss(y_pred, outcomes, reduce=False)).sum() / n_active

        # F^A treatment reconstruction loss (BCE for binary treatments)
        a_pred = self.f_a(br)                                        # [B, T, dim_t]
        loss_a = delta_a * (
            mask * F.binary_cross_entropy(a_pred, treatments.clamp(0.0, 1.0), reduction='none')
        ).sum() / n_active

        loss = loss_y + loss_a

        # F^X vitals reconstruction loss (optional)
        if self.f_x is not None and vitals is not None:
            x_pred = self.f_x(br)                                    # [B, T, dim_v]
            loss_x = delta_x * (mask * F.mse_loss(x_pred, vitals, reduce=False)).sum() / n_active
            loss = loss + loss_x

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# CausalMixerAug — Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class CausalMixerAug(CausalMixer):
    """
    CausalMixerAug (CMC) — CausalMixer + Augmentation innovations.

    Architecture modifications over CausalMixer
    ────────────────────────────────────────────
    Forward pass:
        1. build_br  →  BR [B, T, br_size]              (unchanged)
        2. build_treatment(BR) → treatment_pred          (unchanged)
        3. FiLMConditioningLayer(BR, A_t) → BR_film      (NEW)
        4. build_outcome(BR_film, A_t) → outcome_pred    (uses FiLM-BR)

    Additional training losses:
        L_total = L_factual + L_domain
                + λ_recon × L_recon          (partial autoencoding)
                + λ_cond  × L_cond           (treatment conditioning)
                + λ_ms    × L_multistep      (GRU decoder, from CM)

    Multi-step inference:
        GRU decoder hidden state initialised from FiLM(BR_last, last_treatment)
        rather than raw BR, so the decoder starts from a treatment-conditioned
        representation.

    Configuration (exp section)
    ──────────────────────────
        lambda_recon : float = 0.1   weight for partial autoencoding loss
        lambda_cond  : float = 0.05  weight for treatment conditioning loss
        delta_a      : float = 0.1   weight on treatment reconstruction in L^R
        delta_x      : float = 0.1   weight on vitals reconstruction in L^R
        n_cf_arms    : int   = 4     number of counterfactual arms per step
        label_smooth : float = 0.1   label smoothing for conditioning BCE loss
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
        # CausalMixer.__init__ calls _init_specific internally.
        super().__init__(
            args, dataset_collection, autoregressive,
            has_vitals, projection_horizon, bce_weights, **kwargs,
        )
        # save_hyperparameters is already called in CausalMixer.__init__.

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation — adds Augmentation components on top of CM
    # ─────────────────────────────────────────────────────────────────────────

    def _init_specific(self, sub_args: DictConfig):
        """Calls parent _init_specific then appends Augmentation components."""
        super()._init_specific(sub_args)

        # Guard: if parent init aborted due to missing hparams, do the same.
        if not hasattr(self, 'br_size') or self.br_size is None:
            return

        try:
            # ── FiLM conditioning layer ───────────────────────────────────────
            # Applied to BR before the outcome head. By conditioning the BR on
            # the treatment before predicting Y, the model can learn treatment-
            # specific transformations rather than relying on concatenation.
            self.film_layer = FiLMConditioningLayer(
                br_size=self.br_size,
                dim_treatments=self.dim_treatments,
            )

            # ── Partial autoencoder heads ─────────────────────────────────────
            # F^A + F^Y + F^X — reconstruct inputs from the balanced repr.
            # This regularisation prevents adversarial domain-confusion from
            # stripping covariate information (representation invertibility).
            self.autoencoder_heads = PartialAutoencoderHeads(
                br_size=self.br_size,
                dim_treatments=self.dim_treatments,
                dim_outcome=self.dim_outcome,
                dim_vitals=self.dim_vitals if self.has_vitals else 0,
            )

            # ── FiLM-conditioned GRU decoder ──────────────────────────────────
            # Replaces CM's direct_head so the GRU hidden state is seeded from
            # a treatment-conditioned BR rather than raw BR.
            # (direct_head was already created by super()._init_specific, so we
            # just keep the same object — the FiLM conditioning happens in
            # get_autoregressive_predictions before seeding the GRU.)

            logger.info('CausalMixerAugmentation: FiLM + AutoencoderHeads initialised.')

        except Exception as e:
            logger.warning(f'CausalMixerAugmentation extra init failed: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass — adds FiLM conditioning to outcome prediction
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch, detach_treatment=False):
        """
        Same as CausalMixer.forward but applies FiLM before the outcome head.

        Returns (treatment_pred, outcome_pred, br, film_br) where:
            br      : raw balanced representation [B, T, br_size]
            film_br : FiLM-conditioned BR [B, T, br_size]

        For compatibility with the parent training_step, we still return only
        (treatment_pred, outcome_pred, br) — film_br is stored as self._film_br
        and accessed by training_step.
        """
        fixed_split = batch.get('future_past_split', None)

        # Vitals-augmentation during training (mirrors CT exactly).
        if (self.training
                and self.hparams.model.multi.augment_with_masked_vitals
                and self.has_vitals):
            assert fixed_split is None
            fixed_split = torch.empty(2 * len(batch['active_entries'])).type_as(
                batch['active_entries']
            )
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len
                fixed_split[len(batch['active_entries']) + i] = torch.randint(
                    0, int(seq_len) + 1, (1,)
                ).item()
            for k, v in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries']

        br = self.build_br(
            prev_treatments, vitals, prev_outputs,
            static_features, active_entries, fixed_split,
        )

        # Build treatment prediction on raw BR (same as CM).
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)

        # FiLM-condition the BR with the current treatment before outcome pred.
        # This replaces the implicit conditioning via concatenation in the
        # BRTreatmentOutcomeHead.build_outcome method.
        film_br = self.film_layer(br, curr_treatments)          # [B, T, br_size]

        # Build outcome from FiLM-conditioned BR.
        outcome_pred = self.br_treatment_outcome_head.build_outcome(film_br, curr_treatments)

        # Store film_br for training_step (avoids re-computing).
        self._film_br = film_br
        self._raw_br = br

        return treatment_pred, outcome_pred, br

    # ─────────────────────────────────────────────────────────────────────────
    # CAETC Treatment Conditioning Loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_conditioning_loss(
        self,
        br: torch.Tensor,             # [B, T, br_size]
        treatments: torch.Tensor,     # [B, T, dim_t]
        active_entries: torch.Tensor, # [B, T, 1]
        n_cf_arms: int = 4,
        label_smooth: float = 0.1,
    ) -> torch.Tensor:
        """
        Treatment-specific conditioning loss L^C.

        For each of n_cf_arms counterfactual treatment arms, apply FiLM to BR
        and check whether F^A correctly reconstructs the counterfactual
        treatment.  This forces FiLM to produce treatment-distinguishable
        modulations, not just a fixed identity transform.

        Counterfactual arms are generated by rolling the treatment tensor
        within the batch (batch-level permutation), giving pseudo-counterfactual
        treatments that are guaranteed to be from the same data distribution
        but typically differ from the observed treatment.

        Args:
            br             : raw balanced representations [B, T, br_size]
            treatments     : observed treatment vectors  [B, T, dim_t]
            active_entries : mask [B, T, 1]
            n_cf_arms      : number of counterfactual perturbations to generate
            label_smooth   : label smoothing ε for conditioning BCE loss
        Returns:
            Scalar conditioning loss.
        """
        mask = active_entries  # [B, T, 1]
        n_active = mask.sum().clamp(min=1.0)
        total_loss = br.new_zeros(())

        for k in range(1, n_cf_arms + 1):
            # Roll by k positions in the batch dimension → counterfactual
            # treatment from a different patient, guaranteed ≠ observed
            # (with high probability for k < B).
            a_cf = torch.roll(treatments, shifts=k, dims=0)  # [B, T, dim_t]

            # Apply FiLM with counterfactual treatment.
            film_cf = self.film_layer(br.detach(), a_cf)       # [B, T, br_size]

            # F^A should reconstruct the counterfactual treatment a_cf.
            a_cf_pred = self.autoencoder_heads.f_a(film_cf)    # [B, T, dim_t]

            # Smooth target: ε/K + (1 - ε) * a_cf
            target = (label_smooth / max(self.dim_treatments, 1)) + (1.0 - label_smooth) * a_cf.clamp(0.0, 1.0)

            bce = F.binary_cross_entropy(a_cf_pred, target, reduction='none')  # [B, T, dim_t]
            # Average over treatment dims then mask.
            bce = bce.mean(dim=-1, keepdim=True)                # [B, T, 1]
            total_loss = total_loss + (mask * bce).sum() / n_active

        return total_loss / n_cf_arms

    # ─────────────────────────────────────────────────────────────────────────
    # Training step — extends CM training_step with CAETC losses
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        Extends CausalMixer.training_step with:
          • partial autoencoding loss  (λ_recon)
          • treatment conditioning loss (λ_cond)

        All CM losses (MSE factual, BCE domain, GRU multi-step) are inherited
        unchanged.  The CAETC components are simply added on top.
        """
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:  # representation + outcome update
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
            else:
                treatment_pred, outcome_pred, br = self(batch)

            # ── Standard CM losses ────────────────────────────────────────────
            mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

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

            # ── GRU multi-step auxiliary loss (from CM) ───────────────────────
            lambda_ms = float(getattr(self.hparams.exp, 'lambda_ms', 0.2))
            if lambda_ms > 0.0:
                ms_loss = self._compute_direct_multi_step_loss(br, batch)
                if ms_loss is not None:
                    loss = loss + lambda_ms * ms_loss
                    self.log(
                        f'{self.model_type}_train_ms_loss', ms_loss,
                        on_epoch=True, on_step=False, sync_dist=True,
                    )

            # ── CAETC: Partial autoencoding loss ──────────────────────────────
            lambda_recon = float(getattr(self.hparams.exp, 'lambda_recon', 0.1))
            if lambda_recon > 0.0 and hasattr(self, 'autoencoder_heads'):
                vitals = batch.get('vitals', None) if self.has_vitals else None
                delta_a = float(getattr(self.hparams.exp, 'delta_a', 0.1))
                delta_x = float(getattr(self.hparams.exp, 'delta_x', 0.1))

                recon_loss = self.autoencoder_heads.compute_reconstruction_loss(
                    br=br,
                    treatments=batch['current_treatments'],
                    outcomes=batch['outputs'],
                    active_entries=batch['active_entries'],
                    vitals=vitals,
                    delta_a=delta_a,
                    delta_x=delta_x,
                    delta_y=1.0,
                )
                loss = loss + lambda_recon * recon_loss
                self.log(
                    f'{self.model_type}_train_recon_loss', recon_loss,
                    on_epoch=True, on_step=False, sync_dist=True,
                )

            # ── CAETC: Treatment conditioning loss ────────────────────────────
            lambda_cond = float(getattr(self.hparams.exp, 'lambda_cond', 0.05))
            if lambda_cond > 0.0 and hasattr(self, 'film_layer'):
                n_cf_arms   = int(getattr(self.hparams.exp, 'n_cf_arms', 4))
                label_smooth = float(getattr(self.hparams.exp, 'label_smooth', 0.1))

                cond_loss = self._compute_conditioning_loss(
                    br=br,
                    treatments=batch['current_treatments'],
                    active_entries=batch['active_entries'],
                    n_cf_arms=n_cf_arms,
                    label_smooth=label_smooth,
                )
                loss = loss + lambda_cond * cond_loss
                self.log(
                    f'{self.model_type}_train_cond_loss', cond_loss,
                    on_epoch=True, on_step=False, sync_dist=True,
                )

            self.log(f'{self.model_type}_train_loss',     loss,     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha,
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
    # Multi-step inference — FiLM-conditioned GRU seed
    # ─────────────────────────────────────────────────────────────────────────

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        GRU decoder τ-step counterfactual prediction with FiLM-conditioned seed.

        Extends CM.get_autoregressive_predictions:
          • GRU hidden state is seeded from FiLM(BR_last, first_cf_treatment)
            rather than raw BR_last.
          • This means the decoder starts from a representation already
            conditioned on the first counterfactual treatment arm, giving the
            GRU a better-calibrated starting point for the rollout.
        """
        logger.info(f'CMC direct multi-step prediction for {dataset.subset_name}.')
        tau = self.hparams.dataset.projection_horizon

        all_br = torch.tensor(self.get_representations(dataset))  # [N, T, br_size]

        future_treatments = torch.tensor(
            dataset.data['current_treatments']
        ).to(torch.get_default_dtype())  # [N, T_total, dim_treatments]

        all_outputs = torch.tensor(
            dataset.data['outputs']
        ).to(torch.get_default_dtype())  # [N, T_total, dim_outcome]

        splits = dataset.data['future_past_split']  # [N]
        predicted_outputs = np.zeros((len(dataset), tau, self.dim_outcome))

        with torch.no_grad():
            for i in range(len(dataset)):
                split = int(splits[i])

                # Raw BR at the last observed timestep.
                br_last = all_br[i, split - 1, :].unsqueeze(0)  # [1, br_size]

                # Counterfactual treatments for the prediction window.
                fut_trt = future_treatments[i, split:split + tau, :].unsqueeze(0)
                # [1, τ, dim_treatments]

                # FiLM-condition BR_last with the first counterfactual treatment.
                # This gives the GRU decoder a head-start: its initial hidden
                # state already encodes the treatment-specific representation
                # for the first future step.
                if fut_trt.size(1) > 0 and hasattr(self, 'film_layer'):
                    first_cf_trt = fut_trt[:, 0, :]               # [1, dim_t]
                    br_seed = self.film_layer(br_last, first_cf_trt)
                else:
                    br_seed = br_last

                # Seed the GRU with the last observed factual outcome.
                last_out = all_outputs[i, split - 1, :].unsqueeze(0)  # [1, dim_o]

                # Autoregressive inference (future_outputs=None → model feeds
                # its own predictions from step t into step t+1).
                pred = self.direct_head(
                    br_seed, fut_trt,
                    future_outputs=None,
                    last_output=last_out,
                )  # [1, τ, dim_outcome]
                predicted_outputs[i] = pred.squeeze(0).cpu().numpy()

        return predicted_outputs

    # ─────────────────────────────────────────────────────────────────────────
    # Hyperparameter search interface (inherits from CM, same grid)
    # ─────────────────────────────────────────────────────────────────────────
    # set_hparams is inherited unchanged from CausalMixer.


# Backward/forward-compat alias: cmcp.py and package docs refer to this class
# as CausalMixerCAETC (its architecture name); the class body above is
# CausalMixerAug. Keep both names resolvable without duplicating the class.
CausalMixerCAETC = CausalMixerAug
