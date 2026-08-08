"""
One-shot script to manually log an already-completed RMSN run to MLflow.
Use this when a run was done with exp.logging=False and you want to record
the console-printed metrics without retraining.

Usage:
    PYTHONPATH=. python3 scripts/log_rmsn_to_mlflow.py

Edit the METRICS dict below to match your console output before running.
"""

import mlflow

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI      = "file://./mlruns"
EXPERIMENT_NAME = "RMSN/mimic3_real"
RUN_NAME        = "seed10_diastolic_blood_pressure_manual"

# Hyper-parameters / tags to record alongside the metrics
TAGS = {
    "seed":             "10",
    "dataset":          "mimic3_real",
    "outcome":          "diastolic blood pressure",
    "backbone":         "rmsn",
    "hparams_config":   "diastolic_blood_pressure",
    "logged_manually":  "true",   # flag so you know this wasn't live-logged
}

# ── Metrics from the console output ───────────────────────────────────────────
# Propensity-treatment network
METRICS = {
    # Propensity treatment
    "prop_treatment_val_bce_all":   0.6378817740600357,
    "prop_treatment_val_rmse_orig": 0.6386329450256445,
    "prop_treatment_test_bce_all":  0.6339578197175454,
    "prop_treatment_test_rmse_orig":0.6337986652246106,

    # Propensity history network
    "prop_history_val_bce_all":     0.6355639532362947,
    "prop_history_val_rmse_orig":   0.6364836086597828,
    "prop_history_test_bce_all":    0.6318613133240195,
    "prop_history_test_bce_orig":   0.6315267571051345,

    # Encoder (one-step factual RMSE)
    "encoder_val_rmse_all":         5.072630469764275,
    "encoder_val_rmse_orig":        5.031435298966303,
    "encoder_test_rmse_all":        5.107515346848171,
    "encoder_test_rmse_orig":       5.108753601469073,

    # Decoder (val factual + n-step test)
    "decoder_val_rmse_all":         8.9484245038255,
    "decoder_val_rmse_orig":        8.948424503825503,
    "decoder_test_rmse_2-step":     9.490968104482342,
    "decoder_test_rmse_3-step":     10.12947660870352,
    "decoder_test_rmse_4-step":     10.618240252213736,
    "decoder_test_rmse_5-step":     11.01277103774003,
    "decoder_test_rmse_6-step":     11.352126877722357,
}

# ── Log to MLflow ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tags(TAGS)
    mlflow.log_metrics(METRICS)
    print(f"✓ Logged {len(METRICS)} metrics to experiment '{EXPERIMENT_NAME}' "
          f"run '{RUN_NAME}' at {MLFLOW_URI}")
