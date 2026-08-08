# Experiment Replication Guide

This file provides commands to reproduce the experiments reported in the paper. It is
written against the code in this repository, and every number quoted below is taken
directly from the paper text, not estimated or approximated.

---

## 1. Environment Setup

### Requirements

- Python 3.9+
- CPU is sufficient; all reported results were produced on CPU, float64
  (`torch.set_default_dtype(torch.double)` — see `runnables/train_multi.py`)
- ~8 GB RAM per training run

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
export PYTHONPATH=.
```

### Numerical precision

Training defaults to float64 on CPU. `runnables/train_multi.py` supports an opt-in
float32/MPS path (`MPS_FLOAT32=1` on Apple Silicon, or `FORCE_FLOAT32=1` for a
CPU float32 run) — this was **not** used to produce the paper's reported numbers, and
some models with small contrastive-head tensors (CSSPD, CHSPD) are documented in
that file as prone to Metal-driver instability under MPS. Leave both unset for
faithful reproduction.

### MLflow

Runs are logged via MLflow. `config/config.yaml` defaults to
`mlflow_uri: file://./mlruns`, which stores runs locally in `./mlruns/` without
requiring a server. To inspect results in the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Override with `exp.mlflow_uri=<uri>` to log to a remote tracking server.

---

## 2. Data Preparation

### Cancer Simulation (no download required)

A PK-PD simulation of NSCLC tumour dynamics (Geng et al., 2017), generated on the fly.
Confounding parameter gamma in {0,1,2,3,4}. Split: 10,000 train / 1,000 val / 1,000 test.
T_max=60, tau_max=5.

### MIMIC-III Real (requires PhysioNet credentialed access)

De-identified ICU records for 5,000 adult patients, 2 binary treatments (vasopressor,
mechanical ventilation), 25 time-varying covariates, outcome = diastolic blood
pressure. Split: 3,500 train / 750 val / 750 test. T_max=60, tau_max=5.

1. Apply for access: https://physionet.org/content/mimiciii/1.4/
2. Run [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) against the raw
   MIMIC-III tables to produce `all_hourly_data.h5`.
3. Place `all_hourly_data.h5` in `data/processed/`.
4. Preprocess:
   ```bash
   PYTHONPATH=. python3 src/data/mimic_iii/load_data.py \
     --data_path data/processed/ \
     --output_path data/processed/mimic3_real/
   ```

**`data/processed/*.h5` must never be committed to this repository** — MIMIC-III's
Data Use Agreement prohibits redistributing patient-derived data. `.gitignore` already
excludes it; do not override that.

---

## 3. Proposed Models — Cancer Simulation

Protocol: 5 seeds (`10, 101, 1010, 10101, 101010`), gamma in `{0,1,2,3,4}`, mean +- std
reported (matches Melnychuk et al. 2022's protocol, per the paper's Limitations
section).

```bash
SEEDS=(10 101 1010 10101 101010)
GAMMAS=(0 1 2 3 4)

for MODEL in cssd csspd chsd chspd; do
  for GAMMA in "${GAMMAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      PYTHONPATH=. python3 runnables/train_multi.py \
        +dataset=cancer_sim +backbone=${MODEL} \
        "+backbone/${MODEL}_hparams/cancer_sim_domain_conf='${GAMMA}'" \
        exp.seed=${SEED} exp.logging=True
    done
  done
done
```

---

## 4. Proposed Models — MIMIC-III Real

```bash
SEEDS=(10 101 1010 10101 101010)

for MODEL in cssd csspd chsd chspd; do
  PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=mimic3_real +backbone=${MODEL} \
    "+backbone/${MODEL}_hparams/mimic3_real=diastolic_blood_pressure" \
    exp.seed=10,101,1010,10101,101010 exp.logging=True
done
```


## 5. Baselines

### CT

```bash
for GAMMA in 0 1 2 3 4; do
  PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim +backbone=ct \
    "+backbone/ct_hparams/cancer_sim_domain_conf='${GAMMA}'" \
    exp.seed=10,101,1010,10101,101010 exp.logging=True
done

PYTHONPATH=. python3 runnables/train_multi.py \
  +dataset=mimic3_real +backbone=ct \
  "+backbone/ct_hparams/mimic3_real=diastolic_blood_pressure" \
  exp.seed=10,101,1010,10101,101010 exp.logging=True
```

### CRN (two-phase: encoder, then decoder)

```bash
for GAMMA in 0 1 2 3 4; do
  PYTHONPATH=. python3 runnables/train_enc_dec.py \
    +dataset=cancer_sim +backbone=crn \
    "+backbone/crn_hparams/cancer_sim_domain_conf='${GAMMA}'" \
    exp.seed=10,101,1010,10101,101010 exp.logging=True

  PYTHONPATH=. python3 runnables/train_enc_dec.py \
    +dataset=cancer_sim +backbone=crn \
    "+backbone/crn_hparams/cancer_sim_domain_conf='${GAMMA}'" \
    exp.seed=10,101,1010,10101,101010 \
    model.train_decoder=True exp.logging=True
done
```

### RMSN

```bash
for GAMMA in 0 1 2 3 4; do
  PYTHONPATH=. python3 runnables/train_rmsn.py \
    +dataset=cancer_sim +backbone=rmsn \
    "+backbone/rmsn_hparams/cancer_sim_domain_conf='${GAMMA}'" \
    exp.seed=10,101,1010,10101,101010 exp.logging=True
done
```

### G-Net

```bash
for GAMMA in 0 1 2 3 4; do
  PYTHONPATH=. python3 runnables/train_gnet.py \
    +dataset=cancer_sim +backbone=gnet \
    "+backbone/gnet_hparams/cancer_sim_domain_conf='${GAMMA}'" \
    exp.seed=10,101,1010,10101,101010 exp.logging=True
done
```

Check `config/backbone/{crn,rmsn,gnet}_hparams/` for the exact hparam group names
available in this repo before running — verify with:
```bash
PYTHONPATH=. python3 runnables/train_multi.py --cfg job +dataset=cancer_sim +backbone=ct
```

---

## 6. Extracting results from MLflow

```bash
PYTHONPATH=. python3 - <<'EOF'
import mlflow, pandas as pd

client = mlflow.tracking.MlflowClient()
rows = []
for exp in client.search_experiments():
    for run in client.search_runs(experiment_ids=[exp.experiment_id]):
        p, m = run.data.params, run.data.metrics
        rows.append({
            "experiment": exp.name,
            "seed": p.get("exp/seed"),
            "coeff": p.get("dataset/coeff"),
            **{k: v for k, v in m.items() if "rmse" in k.lower()},
        })
df = pd.DataFrame(rows)
print(df.groupby(["experiment", "coeff"]).mean(numeric_only=True).round(3))
EOF
```

---

## 7. Hyperparameters

Core dimensions, shared across all models for fair comparison (paper's
Implementation Details section): `d_model=32` (`seq_hidden_units`), `d_BR=24`
(`br_size`), `d_state=16` (`ssm_d_state`), `L=2` SSM layers (`num_layer`).

Optimiser: Adam, `lr=1e-4` on MIMIC-III (64 batches, 300 epochs) or `lr=1e-3` on
Cancer Simulation (128 batches, 200 epochs); early stopping patience 20 epochs.

CPC/LIM (CSSPD, CHSPD only): `lambda_MS=3.5` (Cancer Sim) / `2.0` (MIMIC-III, per
the actual hparam configs in this repo — see `csspd_hparams/`), `lambda_CPC=0.05`,
`lambda_LIM=0.1`, both selected by grid search on the MIMIC-III validation set (paper
text), with `warmup_epochs=120` before CPC/LIM activate.

These are already encoded in `config/backbone/{cssd,csspd,chsd,chspd}_hparams/` —
you should not need to override them for reproduction. If you do, verify against the
committed hparam files rather than re-deriving from this section.

---

## 8. Estimated Runtime (single CPU)

Actual wall-clock time depends heavily on your machine. As a rough guide: Cancer
Simulation converges faster than MIMIC-III (fewer epochs, smaller sequences);
CSSD-family models are markedly faster than CT at inference because the parallel
decoder replaces CT's per-horizon autoregressive rollout (see cssd.py's docstring).
Budget for a full 4-model x 5-gamma x 5-seed Cancer Simulation sweep taking
substantially longer than a single run — parallelise across seeds/gammas where your
hardware allows.

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run with `PYTHONPATH=.` or `pip install -e .`.

**MLflow not logging / run appears empty**
Start `mlflow server --port=5000` (matching `config/config.yaml`'s `mlflow_uri`)
before training.

**Port 5000 already in use / MLflow UI unreachable with a 403**
On macOS, port 5000 is commonly claimed by the system AirPlay Receiver. Disable it
(System Settings -> General -> AirDrop & Handoff -> AirPlay Receiver) or point
`exp.mlflow_uri` at a different port.

**MIMIC-III preprocessor fails with a missing-column error**
Confirm you ran MIMIC-Extract first and that `all_hourly_data.h5` (its output, not
raw MIMIC-III tables) is in `data/processed/`.

**Hydra config resolution fails**
Run `python3 runnables/train_multi.py --cfg job +dataset=<name> +backbone=<name>` to
print the fully resolved config and locate the broken field.
