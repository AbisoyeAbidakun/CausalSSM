"""
CSSD — SST + Direct Parallel Decoder.

Same ParallelMultiStepDecoder as SSTD; only the encoder base changed from
SST (CT-derived) to SST (SSMM-derived) — see sst.py

Inheritance chain
─────────────────
CSSD → SST → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel → LightningModule

Override surface
─────────────────
  _init_specific          adds self.direct_head after the full SST init
  training_step           adds λ_ms × L_ms on optimizer_idx==0
  _compute_direct_multi_step_loss   parallel decoder loss (ported from CMCPD)
  get_autoregressive_predictions    single parallel pass (ported from CMCPD)

Everything else — domain confusion, BRTreatmentOutcomeHead, build_br,
validation/test loops — is inherited from SST/SSMM without modification.
"""

import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import DataLoader, Dataset

from src.models.sst import SST
from src.models.helper_models.cmcpd import ParallelMultiStepDecoder
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class CSSD(SST):
    """
    CSSD — SST encoder + ParallelMultiStepDecoder.

    Additional hyperparameters (all have sensible defaults):
        trt_enc_dim  : int   per-step treatment embedding width (default 32)
        dec_hidden   : int   trunk hidden width                 (default 128)
        dec_dropout  : float trunk dropout                      (default 0.1)

    Training loss:
        L = L_factual + L_domain + λ_ms × L_ms

    Config keys consumed from exp:
        lambda_ms        (default 3.5)   multi-step loss weight
        ms_step_discount (default 1.0)   per-step horizon discount γ
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
        bce_weights: np.array = None,
        **kwargs,
    ):
        super().__init__(
            args,
            dataset_collection,
            autoregressive,
            has_vitals,
            projection_horizon,
            bce_weights,
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_specific(self, sub_args: DictConfig):
        """
        Extends CSSD._init_specific by adding the ParallelMultiStepDecoder.

        Step 1: super()._init_specific(sub_args)
            Runs CSSD → SSMM → EDSSM full chain:
              • Input projections
              • BRTreatmentOutcomeHead
              • SSMMultiInputBlock stack (self.sequence_blocks)

        Step 2: Build ParallelMultiStepDecoder (self.direct_head)
            Reads trt_enc_dim, dec_hidden, dec_dropout from sub_args.
            Uses self.br_size, self.dim_treatments, self.dim_outcome,
                 self.fc_hidden_units, self.projection_horizon from SSMM init.
        """
        try:
            super()._init_specific(sub_args)  # full CSSD chain

            if self.seq_hidden_units is None:
                raise MissingMandatoryValue()

            trt_enc_dim = int(getattr(sub_args, 'trt_enc_dim', 32))
            dec_hidden  = int(getattr(sub_args, 'dec_hidden',  128))
            dec_dropout = float(getattr(sub_args, 'dec_dropout', 0.1))

            tau = self.projection_horizon
            if tau is None or tau <= 0:
                logger.warning(
                    'CSSD: projection_horizon not set — direct_head not built. '
                    'Multi-step loss will be skipped.'
                )
                return

            self.direct_head = ParallelMultiStepDecoder(
                br_size        = self.br_size,
                dim_treatments = self.dim_treatments,
                tau            = tau,
                trt_enc_dim    = trt_enc_dim,
                hidden         = dec_hidden,
                dim_outcome    = self.dim_outcome,
                dropout        = dec_dropout,
            )

            logger.info(
                f'CSSD: added ParallelMultiStepDecoder '
                f'(tau={tau}, br_size={self.br_size}, '
                f'trt_enc_dim={trt_enc_dim}, dec_hidden={dec_hidden})'
            )

        except MissingMandatoryValue:
            logger.warning('CSSD not fully initialised — mandatory args missing.')

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        optimizer_idx == 0 : encoder + decoder update
            L = bce_loss + mse_loss  (SSMM base)
              + lambda_ms × L_ms     (parallel decoder auxiliary)

        optimizer_idx == 1 : domain classifier update (unchanged from SSMM)
        """
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:
            # ── Single forward pass — capture br for decoder ─────────────────
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
            else:
                treatment_pred, outcome_pred, br = self(batch)

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

            # Mask and average (same convention as SSMM)
            bce_loss = (
                batch['active_entries'].squeeze(-1) * bce_loss
            ).sum() / batch['active_entries'].sum()
            mse_loss = (
                batch['active_entries'] * mse_loss
            ).sum() / batch['active_entries'].sum()

            loss = bce_loss + mse_loss

            self.log(f'{self.model_type}_train_loss',     loss,     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha',
                     self.br_treatment_outcome_head.alpha,
                     on_epoch=True, on_step=False, sync_dist=True)

            # ── Parallel decoder auxiliary loss ───────────────────────────────
            lambda_ms = float(getattr(self.hparams.exp, 'lambda_ms', 3.5))
            if lambda_ms > 0.0 and hasattr(self, 'direct_head'):
                ms_loss = self._compute_direct_multi_step_loss(br, batch)
                if ms_loss is not None:
                    loss = loss + lambda_ms * ms_loss
                    self.log(f'{self.model_type}_train_ms_loss', ms_loss,
                             on_epoch=True, on_step=False, sync_dist=True)

            return loss

        elif optimizer_idx == 1:
            # Domain classifier update — unchanged from SSMM
            return super().training_step(batch, batch_ind, optimizer_idx)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-step loss (ported from CMCPD._compute_direct_multi_step_loss)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_direct_multi_step_loss(
        self,
        br: torch.Tensor,
        batch: dict,
        teacher_forcing_p: float = 0.0,   # ignored — kept for API compat
    ):
        """
        Parallel multi-step auxiliary loss for ParallelMultiStepDecoder.

        Structural properties
        ─────────────────────
        • No teacher forcing — ParallelMultiStepDecoder has no autoregressive
          feedback, so teacher_forcing_p is structurally irrelevant.
        • Gradient isolation — br_last is detached so the multi-step loss
          trains only the decoder heads, not the SSM encoder.  lambda_ms can
          be large (3.5+) without risk of encoder/decoder gradient conflict.
        • Random split sampling — 8 random split points per batch, averaged.
        • Random horizon curriculum — τ' ∈ [2, τ] sampled per split.
        • Per-step discount — controlled by ms_step_discount (default 1.0).

        Args
        ────
        br               : [B, T, br_size]  balanced representations
        batch            : training batch dict
        teacher_forcing_p: ignored

        Returns
        ───────
        Scalar loss averaged over valid splits, or None if T ≤ τ.
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
            discounts = torch.tensor(
                [gamma ** k for k in range(tau)],
                dtype=br.dtype, device=br.device,
            )
        else:
            discounts = None

        for s in split_pts:
            # ── Gradient isolation ─────────────────────────────────────────
            br_last    = br[:, s, :].detach()                           # [B, br_size]
            future_trt = batch['current_treatments'][:, s:s + tau, :]  # [B, τ, dim_t]
            future_out = batch['outputs'][:, s:s + tau, :]             # [B, τ, dim_o]
            future_mask= batch['active_entries'][:, s:s + tau, :]      # [B, τ, 1]

            # ── Random horizon curriculum ──────────────────────────────────
            tau_prime = torch.randint(2, tau + 1, (1,)).item() if tau >= 3 else tau

            future_trt_h  = future_trt[:, :tau_prime, :]
            future_out_h  = future_out[:, :tau_prime, :]
            future_mask_h = future_mask[:, :tau_prime, :]

            if future_mask_h.sum() == 0:
                continue

            # ── Parallel prediction (no rollout) ───────────────────────────
            pred = self.direct_head(br_last, future_trt_h)  # [B, τ', dim_o]

            mse = F.mse_loss(pred, future_out_h, reduce=False)  # [B, τ', dim_o]

            # ── Per-step discount ──────────────────────────────────────────
            if discounts is not None:
                mse = mse * discounts[:tau_prime].view(1, -1, 1)

            loss_s = (future_mask_h * mse).sum() / future_mask_h.sum()
            total_loss = total_loss + loss_s
            n_valid   += 1

        return total_loss / max(n_valid, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Inference — direct parallel prediction (replaces rollout)
    # ─────────────────────────────────────────────────────────────────────────

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        Direct τ-step counterfactual prediction via ParallelMultiStepDecoder.

        Replaces SSMM's get_autoregressive_predictions (τ sequential encoder
        passes) with a single parallel forward pass per sample.

        For each sample i:
          1. Get BR at position split (encodes history through outputs[split-1]).
          2. Get the τ counterfactual treatments from [split, split + τ).
          3. Call ParallelMultiStepDecoder(BR_last, future_treatments).
          4. Store the τ predicted outcomes.
        """
        if not hasattr(self, 'direct_head'):
            logger.warning(
                'SSTD_Lean.get_autoregressive_predictions: direct_head not built '
                '(projection_horizon was not set). Falling back to SSMM rollout.'
            )
            return super().get_autoregressive_predictions(dataset)

        logger.info(
            f'SSTD_Lean: direct parallel prediction for {dataset.subset_name} '
            f'(no autoregressive rollout).'
        )

        tau = self.hparams.dataset.projection_horizon

        all_br = torch.tensor(self.get_representations(dataset))  # [N, T, br_size]
        future_treatments = torch.tensor(
            dataset.data['current_treatments']
        ).to(torch.get_default_dtype())  # [N, T_total, dim_treatments]

        splits = dataset.data['future_past_split']   # [N]
        predicted_outputs = np.zeros((len(dataset), tau, self.dim_outcome))

        with torch.no_grad():
            for i in range(len(dataset)):
                split = int(splits[i])

                # BR at the last observed timestep.
                # split = future_past_split = sequence_lengths - tau
                #   = the first counterfactual timestep index.
                # br[split] encodes history through outputs[split-1]
                #   (prev_outputs[split] = outputs[split-1] in the encoder).
                # Training convention: decoder(br[s], trt[s:s+tau]) → outputs[s:s+tau].
                # So br[split] + trt[split:split+tau] → outputs[split:split+tau]. ✓
                # Using split-1 would give br encoding through outputs[split-2],
                #   causing all predictions to be one step behind. DON'T use split-1.
                br_last = all_br[i, split, :].unsqueeze(0)   # [1, br_size]

                # Counterfactual treatment sequence for the τ-step window
                fut_trt = future_treatments[
                    i, split:split + tau, :
                ].unsqueeze(0)   # [1, τ, dim_treatments]

                # Single parallel forward pass — no rollout needed
                pred = self.direct_head(br_last, fut_trt)   # [1, τ, dim_outcome]

                predicted_outputs[i] = pred.squeeze(0).cpu().numpy()

        return predicted_outputs
