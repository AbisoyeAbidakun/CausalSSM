"""
SST_Lean — Selective-State-Space Model, on the SSM-native base.

Same architecture as SST (CausalMambaLayer per stream + CausalGatedMixer
cross-stream + PositionwiseFeedForward), but built on SSMM instead of CT:
SSMM's `_init_specific` never constructs an attention block stack, so there
is nothing to build-then-discard here — `self.sequence_blocks` is built
exactly once.

Inheritance chain
─────────────────
SST_Lean → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel → LightningModule

Override strategy
─────────────────
SSMM.__init__ calls self._init_specific(args.model.multi), which (via MRO)
dispatches to SST_Lean._init_specific:
  1. calls super()._init_specific(sub_args)  — runs SSMM's init (input
     projections, BRTreatmentOutcomeHead). No blocks are built here.
  2. builds self.sequence_blocks with SSMMultiInputBlock using SSM kwargs
     extracted from sub_args (ssm_d_state, ssm_d_conv, ssm_expand).

Not state_dict-compatible with SST: no input_transformation, no
self_positional_encoding_k/v/cross_*/absolute keys, and the block-stack
key changes from `transformer_blocks.*` to `sequence_blocks.*`. Load SST
checkpoints into this class with strict=False if needed.
"""

import logging
from typing import Union

from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import numpy as np
import torch.nn as nn

from src.models.ssmm import SSMM
from src.models.utils_ssm import SSMMultiInputBlock
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class SST(SSMM):
    """
    Selective-State-Space Model — SSM-native, no attention
    scaffolding to build and discard.

    Additional hyperparameters (all optional with sensible defaults):
        ssm_d_state : int   SSM state size N      (default 16)
        ssm_d_conv  : int   Local-conv width       (default 4)
        ssm_expand  : int   Inner-dim multiplier   (default 1)

    All other sub_args are forwarded to SSMM._init_specific unchanged.
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
        # SSMM.__init__ calls self._init_specific(args.model.multi), which
        # dispatches to SST_Lean._init_specific (Python MRO) below.
        super().__init__(
            args,
            dataset_collection,
            autoregressive,
            has_vitals,
            projection_horizon,
            bce_weights,
            **kwargs,
        )
        # save_hyperparameters is already called by SSMM.__init__ — no need to repeat.

    def _init_specific(self, sub_args: DictConfig):
        """
        Override of SSMM._init_specific.

        Runs SSMM's full initialisation (input projections,
        BRTreatmentOutcomeHead) then builds self.sequence_blocks with
        SSMMultiInputBlock stacks.

        Args
        ----
        sub_args : DictConfig
            model.multi hyperparameters — same dict SSMM receives.
            SST_Lean reads three extra optional keys:
                ssm_d_state (int, default 16)
                ssm_d_conv  (int, default  4)
                ssm_expand  (int, default  1)
        """
        try:
            # ── Step 1: run SSMM's full _init_specific ────────────────────────
            super()._init_specific(sub_args)

            # Guard: if SSMM raised MissingMandatoryValue internally, some attrs
            # will be absent; check for the most fundamental one.
            if self.seq_hidden_units is None:
                raise MissingMandatoryValue()

            # ── Step 2: read SSM-specific hyperparameters ─────────────────────
            # Use getattr with defaults so existing CT-style configs work
            # unmodified (extra keys are simply ignored).
            d_state = int(getattr(sub_args, 'ssm_d_state', 16))
            d_conv  = int(getattr(sub_args, 'ssm_d_conv',  4))
            expand  = int(getattr(sub_args, 'ssm_expand',  1))

            logger.info(
                f'SST_Lean: building {self.num_layer} SSMMultiInputBlock(s) '
                f'(d_state={d_state}, d_conv={d_conv}, expand={expand})'
            )

            # ── Step 3: build sequence_blocks with SSMMultiInputBlock ─────────
            self.sequence_blocks = nn.ModuleList([
                SSMMultiInputBlock(
                    hidden              = self.seq_hidden_units,
                    feed_forward_hidden = self.seq_hidden_units * 4,
                    dropout             = self.dropout_rate,
                    n_inputs            = self.n_inputs,
                    d_state             = d_state,
                    d_conv              = d_conv,
                    expand              = expand,
                )
                for _ in range(self.num_layer)
            ])

        except MissingMandatoryValue:
            logger.warning(
                f'SST_Lean not fully initialised — some mandatory args are missing. '
                f'(Expected if running hyperparameter search.)'
            )
