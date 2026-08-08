import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import AlphaRise, FilteringMlFlowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

if torch.backends.mps.is_available() and __import__('os').environ.get('MPS_FLOAT32', '0') == '1':
    torch.set_default_dtype(torch.float32)
    logger.info("MPS_FLOAT32=1 detected — switching to float32 for MPS compatibility.")

def _get_trainer_kwargs(args):
    import os
    if torch.backends.mps.is_available() and os.environ.get('MPS_FLOAT32', '0') == '1':
        return dict(accelerator="mps", devices=1)
    gpus = eval(str(args.exp.gpus)) or None
    return dict(gpus=gpus)


def _cast_dataset_to_float32(dataset_collection):
    """Cast all float64 numpy arrays in dataset splits to float32 for MPS compatibility."""
    import numpy as np
    _splits = ['train_f', 'val_f', 'test_f', 'test_cf_one_step',
               'test_cf_treatment_seq', 'test_f_mc', 'test_cf_treatment_seq_mc',
               'test_f_multi']
    for split_name in _splits:
        split = getattr(dataset_collection, split_name, None)
        if split is not None and hasattr(split, 'data'):
            split.data = {
                k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v
                for k, v in split.data.items()
            }


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for G-Net
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    if torch.backends.mps.is_available() and __import__('os').environ.get('MPS_FLOAT32', '0') == '1':
        _cast_dataset_to_float32(dataset_collection)
        logger.info("MPS_FLOAT32=1: cast all dataset arrays float64 → float32.")
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Conditional networks outputs are of uniform dim between (dim_outcomes + dim_vitals)
    args.model.g_net.comp_sizes = \
        [(args.model.dim_outcomes + args.model.dim_vitals) // args.model.g_net.num_comp] * args.model.g_net.num_comp

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        model_callbacks = [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None
        model_callbacks = []

    # ============================== Initialisation & Training of G-Net ==============================
    model = instantiate(args.model.g_net, args, dataset_collection, _recursive_=False)
    if args.model.g_net.tune_hparams:
        model.finetune(resources_per_trial=args.model.g_net.resources_per_trial)

    trainer = Trainer(**_get_trainer_kwargs(args), logger=mlf_logger, max_epochs=args.exp.max_epochs, callbacks=model_callbacks)
    trainer.fit(model)

    # Validation factual rmse
    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                         one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq_mc'):  # Test n_step_counterfactual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq,
                                                       dataset_collection.test_cf_treatment_seq_mc)
    elif hasattr(dataset_collection, 'test_f_mc'):  # Test n_step_factual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_f, dataset_collection.test_f_mc)

    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)
    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()

