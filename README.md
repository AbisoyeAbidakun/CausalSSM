# CausalStateSpaceModel

Official code release for:

> **"Causal State-Space Model for Causal Inference: Estimating Longitudinal
> Individual Treatment Effects"**
>
> Abisoye Abidakun¹, Mingjun Zhong¹, Georgios Leontidis²
>
> ¹ Department of Computing Science, University of Aberdeen, Aberdeen, UK
> ² UiT The Arctic University of Norway
>
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(placeholder — update once assigned)*

This repository extends the [CausalTransformer](https://github.com/Valentyn1997/CausalTransformer)
codebase ([Melnychuk, Frauen & Feuerriegel, 2022, ICML](https://proceedings.mlr.press/v162/melnychuk22a/melnychuk22a.pdf))
with a selective-state-space encoder, a non-autoregressive parallel multi-step
decoder, and two contrastive regularisers (CPC, LIM) that resolve a
balancing-prediction mutual-information conflict introduced by domain-confusion
adversarial training. See `REPLICATION.md` for exact commands to reproduce every
reported result, including the baselines.

---

## Models

### Proposed (this paper)

| Model | Description |
|---|---|
| **CSSD**  | Selective-SSM encoder (`SST`) + parallel multi-step decoder + domain confusion. Base model. |
| **CSSPD** | CSSD + CPC (Contrastive Predictive Coding) + LIM (Local Information Maximisation). **Main proposed model.** |
| **CHSD**  | CSSD with the treatment stream replaced by causal self-attention (`CausallyConstrainedHybridBlock`) — architectural ablation. |
| **CHSPD** | CHSD + CPC + LIM — combined architectural + objective-level ablation. |

Reported results (Cancer Simulation, MIMIC-III, ablations) are in `REPLICATION.md`,
quoted directly from the paper.

### Baselines evaluated in the paper

| Model | Method family | Reference |
|---|---|---|
| **CT**    | Self-attention + domain confusion (primary baseline) | Melnychuk et al. (2022), ICML |
| **CRN**   | GRU + gradient-reversal adversarial balancing | Bica et al. (2020), ICLR |
| **RMSN**  | LSTM + inverse-probability-of-treatment weighting | Lim et al. (2018), NeurIPS |
| **G-Net** | LSTM g-computation | Li et al. (2021), MLHC |

### Other infrastructure present but not part of the paper's reported comparisons

`EDCT` (encoder-decoder attention variant of CT) and `MSM` (linear IPTW marginal
structural model, Robins et al. 2000) are runnable but are not among the four
baselines the paper's Results section reports against.

---

## Requirements

- Python 3.9+
- CPU is sufficient (all reported results are CPU, float64); see `REPLICATION.md` §1
  for the opt-in MPS/float32 path and its caveats.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
export PYTHONPATH=.
```

Built on [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/),
[Hydra](https://hydra.cc/docs/intro/) for configuration, and
[MLflow](https://mlflow.org/) for experiment tracking.

## MLflow Setup

`config/config.yaml`'s `exp.mlflow_uri` defaults to `file://./mlruns`, which stores
runs locally without a server. To view results in the MLflow UI, run:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Override the URI with `exp.mlflow_uri=<uri>` if you're running a remote tracking server.

---

## Running Experiments

The training script is shared across models and datasets; mandatory arguments are
documented in `config/config.yaml` and the files under `config/`.

```bash
PYTHONPATH=. python3 runnables/train_<training-type>.py \
  +dataset=<dataset> +backbone=<backbone> exp.seed=10 exp.logging=True
```

`<training-type>` is one of `multi` (CSSD, CSSPD, CHSD, CHSPD, CT, MSM),
`enc_dec` (EDCT, CRN — two-phase, add `model.train_decoder=True` for the decoder
phase), `rmsn`, or `gnet`.

### Backbones

```
+backbone=cssd    # CSSD  — runnables/train_multi.py
+backbone=csspd   # CSSPD — runnables/train_multi.py
+backbone=chsd    # CHSD  — runnables/train_multi.py
+backbone=chspd   # CHSPD — runnables/train_multi.py
+backbone=ct      # CT    — runnables/train_multi.py
+backbone=edct    # EDCT  — runnables/train_enc_dec.py
+backbone=crn     # CRN   — runnables/train_enc_dec.py
+backbone=rmsn    # RMSN  — runnables/train_rmsn.py
+backbone=gnet    # G-Net — runnables/train_gnet.py
+backbone=msm     # MSM   — runnables/train_msm.py
```

Each backbone has hyperparameters saved per dataset, accessed via
`+backbone/<backbone>_hparams/cancer_sim_domain_conf=<gamma>` or
`+backbone/<backbone>_hparams/mimic3_real=diastolic_blood_pressure`. Do not
hand-edit these unless you're deliberately deviating from the paper's reported
settings — see `REPLICATION.md` §7 for how they were verified against the actual
logged experiment runs.

For CT/EDCT/CRN, two adversarial balancing objectives are available:
`exp.balancing=domain_confusion` (used throughout this paper) or
`exp.balancing=grad_reverse` (originally CRN's).

### Datasets

```
+dataset=cancer_sim        # Synthetic Tumour Growth Simulator; set gamma via dataset.coeff=<0.0-4.0>
+dataset=mimic3_real       # MIMIC-III Real; requires PhysioNet access, see below
+dataset=mimic3_synthetic  # MIMIC-III semi-synthetic simulator (not used in this paper's reported results)
```

**Getting the MIMIC-III dataset.** `mimic3_real` requires credentialed access to the raw
MIMIC-III tables via [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (free, but
requires completing CITI "Data or Specimens Only Research" training and signing the data
use agreement).

1. Apply for and complete credentialed access: https://physionet.org/content/mimiciii/1.4/
2. Download the raw MIMIC-III tables and run [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract)
   against them to produce `all_hourly_data.h5`.
3. Place `all_hourly_data.h5` in `data/processed/`.
4. Preprocess it into the format this repo expects:
   ```bash
   PYTHONPATH=. python3 src/data/mimic_iii/load_data.py \
     --data_path data/processed/ \
     --output_path data/processed/mimic3_real/
   ```

See `REPLICATION.md` §2 for the full walkthrough, including outcome/treatment
configuration used in the paper's reported results.

### Example — CSSPD on Cancer Simulation, gamma=1, 5 seeds

```bash
PYTHONPATH=. python3 runnables/train_multi.py -m \
  +dataset=cancer_sim +backbone=csspd \
  "+backbone/csspd_hparams/cancer_sim_domain_conf='1'" \
  exp.seed=10,101,1010,10101,101010 exp.logging=True
```

See `REPLICATION.md` for the complete command set covering every proposed model,
every baseline, and both datasets.

---



## Citation

If you use this code, please cite:

```bibtex
@misc{abidakun2026causalssm,
  title  = {Causal State-Space Model for Causal Inference: Estimating Longitudinal Individual Treatment Effects},
  author = {Abidakun, Abisoye and Zhong, Mingjun and Leontidis, Georgios},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

*(arXiv ID is a placeholder — update once assigned.)*

## License

[MIT License](LICENSE) (original copyright retained from the upstream
CausalTransformer repository this project extends).

---
