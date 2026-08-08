"""

CSSPD — Causal State-Space model with Predictive regularisation and Direct decoder.

CSSPD = CSSD + CPC (Contrastive Predictive Coding) + LIM (Local Information
Maximisation).


  Phase 1 — SSMMultiInputBlock encoder          (from SST)
      O(T) selective-scan temporal mixing per stream.
      CausalGatedMixer SCM-guided cross-stream exchange.

  Phase 2 — ParallelMultiStepDecoder            (from CSSD)
      τ independent heads conditioned on a detached BR snapshot.
      Zero error accumulation.  Single inference pass (not τ rollouts).

  Phase 3 — CPC + LIM regularisation            (from SSTCP)
      CPC  forces BR_t to predict BR_{t+k} — trains encoder to capture
           dynamics that the decoder can exploit across the full horizon.
      LIM  maximises I(x_local_t ; BR_t) — prevents domain-confusion from
           stripping covariate information needed for accurate outcomes.

Inheritance chain
─────────────────
CSSPD → CSSD → SST → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel → LightningModule

Override surface
─────────────────
  _init_specific   : CSSD full chain, then add CPCHead + LocalInfoMaxHead
  training_step    : single forward pass; all four loss terms in one block
  _get_x_local     : pre-cross-mixer stream average from last SSM block

Inherited without modification:
  _compute_direct_multi_step_loss  (CSSD) — 8-split parallel decoder loss
  get_autoregressive_predictions   (CSSD) — direct τ-step parallel inference
  Everything else from SSMM / EDSSM / LightningModule

Training loss
─────────────
  L = bce_loss + mse_loss              (SSMM base)
    + λ_ms   × L_ms                    (parallel decoder; br_last detached)
    + warmup × λ_cpc × L_CPC          (contrastive prediction)
    + warmup × λ_lim × L_LIM          (local InfoMax)

Gradient isolation notes
────────────────────────
  • L_ms  uses br_last.detach() — trains ONLY the decoder heads; λ_ms can be
    large (3.5+) without polluting the encoder gradient.
  • L_LIM uses x_local.detach() — trains LIM heads and the BR projection;
    does NOT create a trivial collapse through the input projection.
  • L_CPC backpropagates into the encoder (that is its purpose); kept small
    via λ_cpc and warmed up linearly to avoid destabilising the factual loss.
	Quick run
	---------
	  python3 runnables/train_multi.py -m \\
	      +dataset=cancer_sim +backbone=csspd \\
	      "+backbone/csspd_hparams/cancer_sim_domain_conf=0" \\
	      exp.seed=10,101,1010,10101,101010

	See REPLICATION.md for full experiment commands.
"""

import logging
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from src.models.cssd import CSSD
from src.models.helper_models.cmcp import CPCHead, LocalInfoMaxHead
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class CSSPD(CSSD):
    """
    CSSPD — full model: SST encoder + CPC + LIM + ParallelMultiStepDecoder.

    Inherits from CSSD (which provides the parallel decoder), adds
    CPCHead and LocalInfoMaxHead for self-supervised representation
    regularisation.

    Additional hyperparameters beyond CSSD (all optional with sensible defaults):
        cpc_k_steps   : int    CPC prediction steps K              (default  3)
        cpc_n_anchors : int    anchor timesteps sampled per batch   (default  8)
        lim_n_negs    : int    LIM negative timesteps per anchor    (default  8)

    Training loss:
        L = L_factual + L_domain
          + λ_ms   × L_ms                  (parallel decoder)
          + warmup × λ_cpc × L_CPC         (contrastive prediction)
          + warmup × λ_lim × L_LIM         (local InfoMax)

    Config keys consumed from exp (beyond CSSD):
        lambda_cpc    (default 0.3)    CPC loss weight
        lambda_lim    (default 0.1)    LIM loss weight
        warmup_epochs (default 60)     linear warmup duration for CPC/LIM
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
        Extends SSTD_Lean._init_specific by adding CPCHead and LocalInfoMaxHead.

        Step 1: super()._init_specific(sub_args)
            Runs SSTD_Lean → SST_Lean → SSMM → EDSSM full chain:
              • Input projections
              • BRTreatmentOutcomeHead
              • SSMMultiInputBlock stack (self.sequence_blocks)
              • ParallelMultiStepDecoder (self.direct_head)

        Step 2: Build CPCHead
            Reads cpc_k_steps, cpc_n_anchors from sub_args.
            Treatment-conditioned: mean treatment over [t, t+k) added to context.

        Step 3: Build LocalInfoMaxHead
            input_dim = seq_hidden_units (pre-cross-mixer embedding space).
        """
        try:
            super()._init_specific(sub_args)  # SSTD_Lean full chain

            if self.seq_hidden_units is None:
                raise MissingMandatoryValue()

            # ── CPC head ────────────────────────────────────────────────────
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

            # ── LIM head ────────────────────────────────────────────────────
            # input_dim matches the pre-cross-mixer embedding space (seq_hidden_units).
            lim_dim = self.br_size
            n_negs  = int(getattr(sub_args, 'lim_n_negs', 8))

            self.lim_head = LocalInfoMaxHead(
                input_dim = self.seq_hidden_units,
                br_size   = self.br_size,
                lim_dim   = lim_dim,
                n_negs    = n_negs,
            )

            logger.info(
                f'SSTCPD_Lean: CPCHead(K={K_steps}, n_anchors={n_anchors}, '
                f'cpc_dim={cpc_dim}) + '
                f'LocalInfoMaxHead(input_dim={self.seq_hidden_units}, '
                f'lim_dim={lim_dim}, n_negs={n_negs}) added to SSTD_Lean backbone.'
            )

        except MissingMandatoryValue:
            logger.warning('SSTCPD_Lean not fully initialised — mandatory args missing.')

    # ─────────────────────────────────────────────────────────────────────────
    # x_local extraction  (same logic as SSTCP)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_x_local(self) -> torch.Tensor:
        """
        Extract pre-cross-mixer stream embeddings from the last SSM block.

        Returns
        -------
        x_local : [B, T, seq_hidden_units]  detached, or None before first forward.

        The LAST SSMMultiInputBlock stores its pre-CausalGatedMixer embeddings as:
            _pre_mixer_x_t, _pre_mixer_x_o, _pre_mixer_x_v (None if no vitals)
        These are averaged across active streams to form x_local.
        Detaching prevents the LIM objective from collapsing the encoder into a
        trivial identity (both projections learning the same representation).
        """
        last_block = self.sequence_blocks[-1]

        if not hasattr(last_block, '_pre_mixer_x_t'):
            return None

        x_t = last_block._pre_mixer_x_t   # [B, T, hidden]
        x_o = last_block._pre_mixer_x_o   # [B, T, hidden]
        x_v = last_block._pre_mixer_x_v   # [B, T, hidden] or None

        if x_v is not None:
            x_local = (x_t + x_o + x_v) / 3.0
        else:
            x_local = (x_t + x_o) / 2.0   # [B, T, hidden]

        return x_local.detach()

    # ─────────────────────────────────────────────────────────────────────────
    # Training — combines SSTD_Lean + SSTCP in a single forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        All loss terms computed from a single forward pass:

        optimizer_idx == 0 : encoder + decoder + CPC/LIM update
            L = bce_loss + mse_loss              (SSMM base)
              + λ_ms   × L_ms                    (parallel decoder; br detached)
              + warmup × λ_cpc × L_CPC          (CPC; backprops into encoder)
              + warmup × λ_lim × L_LIM          (LIM; x_local detached)

        optimizer_idx == 1 : domain classifier update (unchanged from SSMM)

        Gradient flow summary
        ─────────────────────
          Encoder    ← bce + mse + warmup×(λ_cpc×CPC + λ_lim×LIM)
          Decoder    ← λ_ms × L_ms  only  (br_last detached → encoder protected)
          CPC heads  ← L_CPC
          LIM heads  ← L_LIM  (x_local detached → encoder not touched by LIM)
          Domain clf ← optimizer_idx==1 path (separate backward pass)

        The λ_ms decoder signal is large (default 3.5) but never flows back
        through the encoder, so it cannot interfere with CPC/LIM training.
        """
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:
            # ── Single forward pass ──────────────────────────────────────────
            # Side effects: sequence_blocks[-1]._pre_mixer_x_{t,o,v} updated.
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
            else:
                treatment_pred, outcome_pred, br = self(batch)

            # x_local: pre-cross-mixer average from last SSM block (detached)
            x_local = self._get_x_local()   # [B, T, seq_hidden_units] or None

            # ── SSMM base losses ─────────────────────────────────────────────
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

            self.log(f'{self.model_type}_train_loss',     loss,     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha',
                     self.br_treatment_outcome_head.alpha,
                     on_epoch=True, on_step=False, sync_dist=True)

            # ── Parallel decoder loss (SSTD_Lean) ────────────────────────────
            # br_last is detached inside _compute_direct_multi_step_loss, so
            # this loss trains ONLY the decoder heads — encoder is unaffected.
            lambda_ms = float(getattr(self.hparams.exp, 'lambda_ms', 3.5))
            if lambda_ms > 0.0 and hasattr(self, 'direct_head'):
                ms_loss = self._compute_direct_multi_step_loss(br, batch)
                if ms_loss is not None:
                    loss = loss + lambda_ms * ms_loss
                    self.log(f'{self.model_type}_train_ms_loss', ms_loss,
                             on_epoch=True, on_step=False, sync_dist=True)

            # ── Contrastive warmup ───────────────────────────────────────────
            # Linear ramp 0 → 1 over warmup_epochs.
            # CPC/LIM start at full strength at epoch warmup_epochs.
            # The parallel decoder loss is NOT warmed up: it is encoder-isolated
            # (br detached) so it never destabilises the factual representation.
            warmup_epochs = int(getattr(self.hparams.exp, 'warmup_epochs', 60))
            warmup = min(1.0, self.current_epoch / max(warmup_epochs, 1))
            self.log(f'{self.model_type}_train_contrastive_warmup', warmup,
                     on_epoch=True, on_step=False, sync_dist=True)

            # ── CPC contrastive prediction loss (SSTCP) ──────────────────────
            # Backpropagates into the encoder — training it to capture dynamics
            # that are predictive across k steps.  This makes the BR richer for
            # both the decoder heads and the autoregressive counterfactual path.
            lambda_cpc = float(getattr(self.hparams.exp, 'lambda_cpc', 0.3))
            if lambda_cpc > 0.0 and warmup > 0.0 and hasattr(self, 'cpc_head'):
                cpc_loss = self.cpc_head.compute_loss(
                    br,
                    batch['active_entries'],
                    treatments=batch['current_treatments'],
                )
                loss = loss + warmup * lambda_cpc * cpc_loss
                self.log(f'{self.model_type}_train_cpc_loss', cpc_loss,
                         on_epoch=True, on_step=False, sync_dist=True)

            # ── LIM local InfoMax loss (SSTCP) ────────────────────────────────
            # x_local is detached: LIM trains its own projection heads + the BR
            # projection without creating a trivial encoder identity collapse.
            # Cross-patient + time-shuffle negatives protect both axes of info.
            lambda_lim = float(getattr(self.hparams.exp, 'lambda_lim', 0.1))
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

            return loss

        elif optimizer_idx == 1:
            # Domain classifier update — identical to SSMM
            return super().training_step(batch, batch_ind, optimizer_idx)
