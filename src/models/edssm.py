"""
EDSSM — Encoder-Decoder SSM base.

Internal component of the SSM model family. Sits between BRCausalModel
(time_varying_model.py) and SSMM in the inheritance chain:

    CSSPD → CSSD → SST → SSMM → EDSSM → BRCausalModel → TimeVaryingCausalModel

Provides the BRTreatmentOutcomeHead initialisation and the per-stream input
projection layers shared by all SSM-family models (CSSD, CSSPD, CHSD, CHSPD).
"""

import logging
from typing import Union

import numpy as np
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch import nn

from src.models.time_varying_model import BRCausalModel
from src.models.utils import BRTreatmentOutcomeHead
from src.data import RealDatasetCollection, SyntheticDatasetCollection

logger = logging.getLogger(__name__)


class EDSSM(BRCausalModel):

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'multi'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection],
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.save_hyperparameters(args)  # Will be logged to mlflow

    def _init_specific(self, sub_args: DictConfig):
        """
        _init_specific shared for SSM-family multi-input models
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            # No positional encoding — see module docstring for why the SSM
            # family doesn't need one.
            self.output_dropout = nn.Dropout(self.dropout_rate)

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.seq_hidden_units, self.br_size,
                                                                    self.fc_hidden_units, self.dim_treatments, self.dim_outcome,
                                                                    self.alpha, self.update_alpha, self.balancing)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        :param model_args: Sub DictConfig
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.num_heads = new_args['num_heads']

        if 'seq_hidden_units' in new_args:  # Only relevant for encoder: seq_hidden_units should be divisible by num_heads
            # seq_hidden_units should even number - required for fixed positional encoding
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
            comon_multiplier = np.lcm.reduce([sub_args.num_heads, 2]).item()
            if sub_args.seq_hidden_units % comon_multiplier != 0:
                sub_args.seq_hidden_units = sub_args.seq_hidden_units + \
                    (comon_multiplier - sub_args.seq_hidden_units % comon_multiplier)
            print(f'Factual seq_hidden_units of {model_type}: {sub_args.seq_hidden_units}.')

        sub_args.br_size = int(input_size * new_args['br_size'])
        # br-size of encoder = seq_hidden_units of decoder should be divisible by num_heads
        if model_type == 'encoder' and model_args.train_decoder:
            if model_args.decoder.tune_hparams:
                # divisible by all possible num_heads in grid and even (for fixed positional encoding)
                comon_multiplier = np.lcm.reduce(model_args.decoder.hparams_grid.num_heads + [2]).item()
            else:
                comon_multiplier = np.lcm.reduce([model_args.decoder.num_heads, 2]).item()

            if sub_args.br_size % comon_multiplier != 0:
                sub_args.br_size = sub_args.br_size + (comon_multiplier - sub_args.br_size % comon_multiplier)
            print(f'Factual br_size of {model_type}: {sub_args.br_size }.')

        sub_args.fc_hidden_units = int(sub_args.br_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']
