"""
CHSD — Causally-Constrained Hybrid encoder + Direct Parallel Decoder.

Replaces CSSD's uniform SSMMultiInputBlock stack with the
CausallyConstrainedHybridBlock (CCH), which assigns each stream's temporal
mixing operator based on its causal role in the identification problem:

    Treatment stream  →  causal self-attention  (O(T²))
    Outcome stream    →  CausalMambaLayer        (O(T))
    Vitals stream     →  CausalMambaLayer        (O(T))   [if present]
    Cross-stream      →  CausalGatedMixer        (O(T))

Inheritance chain
─────────────────
CHSD → CSSD → SST → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel → LightningModule

Override surface
─────────────────
  _init_specific : CSSD full chain, then rebuild sequence_blocks
                   with CausallyConstrainedHybridBlock

All other methods — training_step, _compute_direct_multi_step_loss,
get_autoregressive_predictions — are inherited from CSSD unchanged.

New hyperparameters
───────────────────
    cch_attn_heads   : int   Treatment attention heads (default 2)
    cch_attn_dropout : float Treatment attention dropout (default 0.0)
"""

import logging
from typing import Union

import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from src.models.cssd import CSSD
from src.models.utils_ssm import CausallyConstrainedHybridBlock
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class CHSD(CSSD):
    """
    CHSD — Causally-Constrained Hybrid encoder + ParallelMultiStepDecoder.

    Inherits CSSD's parallel decoder, training loop, and inference path.
    Replaces sequence_blocks with CausallyConstrainedHybridBlock.

    Additional hyperparameters (beyond CSSD):
        cch_attn_heads   : int   Treatment self-attention heads (default 2)
        cch_attn_dropout : float Treatment attention dropout    (default 0.0)

    The SSM encoder hyperparameters (ssm_d_state, ssm_d_conv, ssm_expand),
    decoder hyperparameters (trt_enc_dim, dec_hidden, dec_dropout), and
    training loss hyperparameters (lambda_ms, ms_step_discount) are all
    inherited from CSSD without change.
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
        Extends CSSD._init_specific by replacing sequence_blocks with
        CausallyConstrainedHybridBlock.

        Step 1: super()._init_specific(sub_args)
            Runs CSSD → SST → SSMM → EDSSM full chain:
              • Input projections
              • BRTreatmentOutcomeHead
              • SSMMultiInputBlock stack (immediately replaced below)
              • ParallelMultiStepDecoder (self.direct_head)

        Step 2: Read CCH-specific hyperparameters.

        Step 3: Rebuild sequence_blocks with CausallyConstrainedHybridBlock.
            num_layer CCH blocks, each with:
              • Treatment: MultiHeadedAttention (O(T²), causal)
              • Outcome:   CausalMambaLayer     (O(T))
              • Vitals:    CausalMambaLayer     (O(T))   [if present]
              • Mixer:     CausalGatedMixer     (O(T))
        """
        try:
            # ── Step 1: full CSSD chain ─────────────────────────────────
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
                f'CHSD: replacing {self.num_layer} SSMMultiInputBlock(s) with '
                f'CausallyConstrainedHybridBlock '
                f'(attn_heads={attn_heads}, d_state={d_state}, '
                f'd_conv={d_conv}, expand={expand}). '
                f'Treatment stream: O(T²) attention; '
                f'Outcome/vitals: O(T) SSM.'
            )

            # ── Step 3: rebuild sequence_blocks ───────────────────────────────
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
            logger.warning('CHSD not fully initialised — mandatory args missing.')