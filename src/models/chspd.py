"""
CHSPD — Causally-Constrained Hybrid + CPC + LIM + Parallel Decoder.

Inheritance chain
─────────────────
CHSPD → CSSPD → CSSD → SST → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel → LightningModule

Override surface
─────────────────
  _init_specific : CSSPD full chain, then build sequence_blocks
                   with CausallyConstrainedHybridBlock

Everything else — training_step (all 4 losses), _get_x_local,
_compute_direct_multi_step_loss, get_autoregressive_predictions —
is inherited from CSSPD / CSSD without modification.

New hyperparameters (beyond CSSPD)
──────────────────────────────────────────
    cch_attn_heads   : int   Treatment self-attention heads (default 2)
    cch_attn_dropout : float Treatment attention dropout    (default 0.0)
"""

import logging
from typing import Union

import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from src.models.csspd import CSSPD
from src.models.utils_ssm import CausallyConstrainedHybridBlock
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class CHSPD(CSSPD):
    """
    CHSPD — CCH encoder + CPC + LIM + ParallelMultiStepDecoder.

    Inherits from CSSPD (all four loss terms, CPC/LIM heads, parallel
    decoder, direct inference).  Only sequence_blocks are replaced with
    CausallyConstrainedHybridBlock.

    Additional hyperparameters (beyond CSSPD):
        cch_attn_heads   : int   Treatment self-attention heads (default 2)
        cch_attn_dropout : float Treatment attention dropout    (default 0.0)

    All CSSPD hyperparameters — cpc_k_steps, cpc_n_anchors, lim_n_negs,
    lambda_cpc, lambda_lim, warmup_epochs, lambda_ms — are inherited.
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
        Extends CSSPD._init_specific by replacing sequence_blocks with
        CausallyConstrainedHybridBlock.

        Step 1: super()._init_specific(sub_args)
            Runs CSSPD → CSSD → SST → SSMM → EDSSM full chain:
              • Input projections
              • BRTreatmentOutcomeHead
              • SSMMultiInputBlock stack (immediately replaced below)
              • ParallelMultiStepDecoder (self.direct_head)
              • CPCHead (self.cpc_head)
              • LocalInfoMaxHead (self.lim_head)

        Step 2–3: Read CCH hyperparameters, build sequence_blocks.
            All auxiliary heads (cpc_head, lim_head, direct_head) remain.
        """
        try:
            # ── Step 1: full CSSPD chain ───────────────────────────────
            super()._init_specific(sub_args)

            if self.seq_hidden_units is None:
                raise MissingMandatoryValue()

            # ── Step 2: CCH-specific hyperparameters ─────────────────────────
            d_state      = int(getattr(sub_args, 'ssm_d_state',         16))
            d_conv       = int(getattr(sub_args, 'ssm_d_conv',           4))
            expand       = int(getattr(sub_args, 'ssm_expand',           1))
            attn_heads   = int(getattr(sub_args, 'cch_attn_heads',       2))
            attn_dropout = float(getattr(sub_args, 'cch_attn_dropout', 0.0))

            logger.info(
                f'CHSPD: replacing {self.num_layer} SSMMultiInputBlock(s) with '
                f'CausallyConstrainedHybridBlock '
                f'(attn_heads={attn_heads}, d_state={d_state}, '
                f'd_conv={d_conv}, expand={expand}). '
                f'Treatment stream: O(T²) attention (backdoor-motivated); '
                f'Outcome/vitals: O(T) SSM.'
            )

            # ── Step 3: build sequence_blocks ─────────────────────────────────
            self.sequence_blocks = nn.ModuleList([
                CausallyConstrainedHybridBlock(
                    hidden              = self.seq_hidden_units,
                    feed_forward_hidden = self.seq_hidden_units * 4,
                    dropout             = self.dropout_rate,
                    n_inputs            = self.n_inputs,
                    d_state             = d_state,
                    d_conv              = d_conv,
                    expand              = expand,
                    attn_heads          = attn_heads,
                    attn_dropout        = attn_dropout,
                )
                for _ in range(self.num_layer)
            ])

        except MissingMandatoryValue:
            logger.warning('CHSPD not fully initialised — mandatory args missing.')
