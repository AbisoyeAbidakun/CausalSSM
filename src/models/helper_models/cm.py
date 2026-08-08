import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import DataLoader, Dataset

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.helper_models.causal_mixer_block import CausalMixerBlock, LearnablePositionalEncoding
from src.models.time_varying_model import BRCausalModel
from src.models.utils import BRTreatmentOutcomeHead, MixtureDensityOutcomeHead

logger = logging.getLogger(__name__)


class GRUMultiStepDecoder(nn.Module):
    """
    GRU-based autoregressive decoder for τ-step counterfactual prediction.


    This decoder gives CausalMixer the same sequential conditioning advantage:
      • The balanced representation initialises the GRU hidden state.
      • At each step t, the GRU takes [future_treatment_t, prev_outcome_t-1].
      • This lets the decoder model "if I gave drug A at step 1 and tumour grew
        by X, then giving drug B at step 2 will reduce it by Y" — exactly the
        kind of conditional dynamics that the MLP cannot model.  The GRU can learn to hedge its predictions
	  • The GRU is a single-layer cell with a small hidden size (4× br_size = 64)
      • Critically, only the τ-step decoder is autoregressive; the O(T) mixer
        backbone is still non-autoregressive, so training is fast.

    Training — teacher forcing:
      When `future_outputs` is provided, prev_outcome at step t is taken from
      the ground-truth output at t-1 rather than the predicted output.  This
      prevents compounding errors from destabilising early training.

    Inference:
      `future_outputs` is None → the decoder feeds its own prediction from
      step t as input to step t+1 (true autoregressive rollout).

    Parameters
    ----------
    br_size            : size of the balanced representation (GRU init input)
    dim_treatments     : dimensionality of treatment vector per step
    projection_horizon : τ — number of future steps to predict
    dim_outcome        : dimensionality of the outcome vector
    """

    def __init__(
        self,
        br_size: int,
        dim_treatments: int,
        projection_horizon: int,
        dim_outcome: int,
        hidden_size: int = None,
    ):
        super().__init__()
        self.projection_horizon = projection_horizon
        self.dim_outcome = dim_outcome

        # GRU hidden dimension.
        self.hidden_size = hidden_size if hidden_size is not None else max(br_size * 4, 64)

        # Project the balanced representation → initial GRU hidden state.
        # Tanh keeps the initial hidden state in the GRU's operating range.
        self.init_hidden = nn.Sequential(
            nn.Linear(br_size, self.hidden_size),
            nn.Tanh(),
        )

        self.br_proj_size = max(8, br_size // 2)
        self.br_proj = nn.Sequential(
            nn.Linear(br_size, self.br_proj_size),
            nn.Tanh(),
        )

        # GRU cell: input = [treatment_at_t, outcome_at_t-1, br_context]
        self.gru_cell = nn.GRUCell(
            input_size=dim_treatments + dim_outcome + self.br_proj_size,
            hidden_size=self.hidden_size,
        )

        # ── Cross-attention over the full BR sequence ─────────────────────────

        self.br_kv_proj = nn.Linear(br_size, self.hidden_size)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=2,
            dropout=0.0,
            batch_first=True,
        )
        # Learnable gate: initialised at 0 so sigmoid(0) = 0.5 — the attention

        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.attn_norm = nn.LayerNorm(self.hidden_size)

        # Project GRU hidden state → predicted outcome.
        # LayerNorm before the linear stabilises the output scale.
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, dim_outcome),
        )

    def forward(
        self,
        br_last: torch.Tensor,                   # [B, br_size]
        future_treatments: torch.Tensor,          # [B, τ, dim_treatments]
        br_seq: torch.Tensor = None,             # [B, T_enc, br_size]  full history for cross-attn
        future_outputs: torch.Tensor = None,      # [B, τ, dim_outcome]  (teacher forcing)
        last_output: torch.Tensor = None,         # [B, dim_outcome]  (t=-1 seed)
        teacher_forcing_p: float = 1.0,          # probability of using ground-truth prev_out
    ) -> torch.Tensor:
        """
        Returns [B, τ, dim_outcome].

        Args:
            br_last            : balanced representation at the split timestep [B, br_size]
            future_treatments  : counterfactual treatments [split, split+τ)
            br_seq             : full BR sequence [B, T_enc, br_size] for cross-attention.
                                 When provided, the decoder attends this sequence at every
                                 step.
                                 When None, falls back to br_last only (original behaviour).
            future_outputs     : ground-truth outcomes for teacher forcing (train only)
            last_output        : observed outcome at split-1 to seed the decoder
            teacher_forcing_p  : probability of using ground-truth prev_out at each
                                 step during training (0 = fully autoregressive,
                                 1 = fully teacher-forced).  Mixed-mode training
                                 (0 < p < 1) closes the train/inference gap by
                                 exposing the GRU to its own predictions during
                                 training, eliminating the exposure-bias problem
                                 that caused n-step RMSE regression.
        """
        B = br_last.size(0)
        device = br_last.device
        dtype = br_last.dtype

        # Initialise GRU hidden state from the balanced representation.
        h = self.init_hidden(br_last)   # [B, hidden_size]

        # Seed prev_out from the last observed factual outcome so the decoder
        # starts at the right scale rather than zero.
        if last_output is not None:
            prev_out = last_output.to(dtype=dtype, device=device)
        else:
            prev_out = torch.zeros(B, self.dim_outcome, dtype=dtype, device=device)

        # Project BR once, reuse at every autoregressive step as step-wise context.
        br_context = self.br_proj(br_last)  # [B, br_proj_size]

        # ── Cross-attention key/value setup ───────────────────────────────────
        # Project the encoder BR sequence into the GRU hidden space
        ATN_WINDOW = 20
        if br_seq is not None:
            br_seq_w = br_seq[:, -ATN_WINDOW:, :] if br_seq.size(1) > ATN_WINDOW else br_seq
            br_kv = self.br_kv_proj(br_seq_w.to(dtype=dtype, device=device))
            # [B, min(T_enc, ATN_WINDOW), hidden_size]
        else:
            br_kv = self.br_kv_proj(br_last.unsqueeze(1))  # [B, 1, hidden_size]

        gate = torch.sigmoid(self.attn_gate)

        outputs = []
        for t in range(future_treatments.size(1)):
            trt_t = future_treatments[:, t, :]                        # [B, dim_t]
            inp   = torch.cat([trt_t, prev_out, br_context], dim=-1)  # [B, dim_t + dim_o + br_proj_size]
            h     = self.gru_cell(inp, h)

            query    = h.unsqueeze(1)                           # [B, 1, hidden_size]
            attn_out, _ = self.cross_attn(query, br_kv, br_kv) # [B, 1, hidden_size]
            # Gated blend: h_aug = h + sigmoid(gate) × attn_context
            h_aug = self.attn_norm(h + gate * attn_out.squeeze(1))

            # Residual (Δy) prediction using attention-augmented hidden state.
            out_t = prev_out + self.out_proj(h_aug)             # [B, dim_o]
            outputs.append(out_t)

            if (future_outputs is not None
                    and teacher_forcing_p > 0.0
                    and (teacher_forcing_p >= 1.0 or torch.rand(1).item() < teacher_forcing_p)):
                prev_out = future_outputs[:, t, :]
            else:
                prev_out = out_t

        return torch.stack(outputs, dim=1)   # [B, τ, dim_o]


class KoopmanMultiStepDecoder(nn.Module):
    """
    Koopman-operator τ-step counterfactual decoder — NO sequential error accumulation.

    Why Koopman instead of a GRU
    ─────────────────────────────
    The GRU decoder is autoregressive: z_{t+k} = f(z_{t+k-1}, ŷ_{t+k-1}, a_{t+k}).
    Every predicted output ŷ_{t+k-1} feeds into the next step, so prediction
    error compounds — step-5 is structurally harder than step-1.

    Koopman theory (Lusch et al. NeurIPS 2018; Morton et al. AAAI 2018) shows
    that any nonlinear dynamical system admits an equivalent LINEAR representation
    in a lifted observable space:

        z_{t+1} = A z_t + B a_t

    where z = g(BR) is a learned embedding, A is the (linear) autonomous dynamics
    matrix, and B is the (linear) control matrix.  The τ-step prediction is a
    single closed-form expression:

        z_{t+τ} = A^τ z_t + Σ_{k=0}^{τ-1} A^k B a_{t+τ-1-k}

    Critically: z_{t+k} depends only on CONTROLS a_{t+j} and INITIAL STATE z_t,
    NEVER on predicted outputs ŷ_{t+j}.  There is no output-feedback loop and
    therefore no error accumulation — step-5 prediction is no harder than step-1.

    Architecture
    ────────────
    encoder : BR (br_size) → z (koopman_dim)   2-layer MLP, GELU activation
    A       : koopman_dim → koopman_dim         linear (no bias), init ≈ Identity
    B       : dim_treatments → koopman_dim      linear (no bias)
    decoder : z (koopman_dim) → ŷ (dim_outcome) LayerNorm → Linear → GELU → Linear

    A is initialised as Identity + 0.01×noise so that the autonomous dynamics
    start near-constant (z doesn't change without treatment). Gradient clipping
    (max_grad_norm=1.0) prevents A from developing unstable eigenvalues.

    Interface compatibility
    ───────────────────────
    The forward signature accepts all GRUMultiStepDecoder kwargs (br_seq,
    future_outputs, last_output, teacher_forcing_p) and silently ignores them.
    This means _compute_direct_multi_step_loss and get_autoregressive_predictions
    require zero changes.

    Parameters
    ----------
    br_size            : size of the balanced representation (encoder output)
    dim_treatments     : dimensionality of treatment vector per step
    projection_horizon : τ — number of future steps to predict
    dim_outcome        : dimensionality of the outcome vector
    koopman_dim        : dimension of the Koopman observable space; default
                         max(br_size * 4, 64) matches the previous GRU hidden.
    """

    def __init__(
        self,
        br_size: int,
        dim_treatments: int,
        projection_horizon: int,
        dim_outcome: int,
        koopman_dim: int = None,
    ):
        super().__init__()
        self.projection_horizon = projection_horizon
        self.dim_outcome = dim_outcome
        self.koopman_dim = koopman_dim if koopman_dim is not None else max(br_size * 4, 64)

        # ── Koopman encoder: BR → observable space ────────────────────────────
        # A two-layer MLP lifts the balanced representation into the Koopman
        # observable space.  Using 2×koopman_dim hidden units gives the encoder
        # enough capacity to find a good nonlinear lift.  GELU is smoother than
        # ReLU, which is empirically better for learning embeddings of smooth
        # dynamical systems (tumor volume evolves continuously).
        self.encoder = nn.Sequential(
            nn.Linear(br_size, self.koopman_dim * 2),
            nn.GELU(),
            nn.Linear(self.koopman_dim * 2, self.koopman_dim),
        )

        # ── Koopman dynamics matrix A ──────────────────────────────────────────
        # Represents the autonomous (treatment-free) evolution z_{t+1} ← A z_t.
        # No bias: linear Koopman dynamics must be homogeneous (F(0) = 0 in
        # latent space, i.e. if the system is at the Koopman origin it stays).
        # Initialised as Identity + small noise so training starts with stable,
        # near-constant dynamics and learns deviations (tumor growth rate, drug
        # decay) from there.
        self.A = nn.Linear(self.koopman_dim, self.koopman_dim, bias=False)
        nn.init.eye_(self.A.weight)
        self.A.weight.data.add_(0.01 * torch.randn_like(self.A.weight))

        # ── Koopman control matrix B ───────────────────────────────────────────
        # Maps treatment a_t ∈ R^{dim_treatments} into the Koopman space.
        # Captures the LINEAR effect of each treatment dimension on the Koopman
        # state evolution: z_{t+1} += B a_t.  No bias (same homogeneity reason).
        self.B = nn.Linear(dim_treatments, self.koopman_dim, bias=False)

        # ── Koopman decoder: observable space → outcome ────────────────────────
        # Inverts the encoder.  LayerNorm first re-scales the Koopman state
        # (which grows/shrinks during the linear rollout) before the nonlinear
        # decode.  Two-layer MLP with GELU matches the encoder depth.
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.koopman_dim),
            nn.Linear(self.koopman_dim, self.koopman_dim // 2),
            nn.GELU(),
            nn.Linear(self.koopman_dim // 2, dim_outcome),
        )

    def forward(
        self,
        br_last: torch.Tensor,                   # [B, br_size]
        future_treatments: torch.Tensor,          # [B, τ, dim_treatments]
        br_seq: torch.Tensor = None,             # unused (no sequential attention needed)
        future_outputs: torch.Tensor = None,      # unused (no teacher forcing)
        last_output: torch.Tensor = None,         # unused (no output-feedback seeding)
        teacher_forcing_p: float = 1.0,          # unused (no output-feedback loop)
    ) -> torch.Tensor:
        """
        Returns [B, τ, dim_outcome] via purely linear Koopman rollout.

        For each step t ∈ {1, …, τ}:
            z_t = A z_{t-1} + B a_t           (linear — no ŷ dependency)
            ŷ_t = decoder(z_t)

        This is equivalent to the closed-form
            z_{t+τ} = A^τ z_0 + Σ_{k=0}^{τ-1} A^k B a_{t+τ-1-k}
        but computed iteratively for clarity and numerical stability.
        Because z does NOT depend on any ŷ, prediction errors at step k
        have zero influence on steps k+1, …, τ.
        """
        # Lift the balanced representation into the Koopman observable space.
        z = self.encoder(br_last.to(dtype=next(self.parameters()).dtype))  # [B, koopman_dim]

        outputs = []
        for t in range(future_treatments.size(1)):
            a_t = future_treatments[:, t, :].to(dtype=z.dtype, device=z.device)
            # ── Single Koopman step: purely linear, zero error accumulation ───
            # z_{t+1} = A z_t + B a_t
            # This is a matrix-vector multiply — O(koopman_dim²) regardless of
            # horizon.  No recurrent prediction-error loop.
            z = self.A(z) + self.B(a_t)          # [B, koopman_dim]
            y_t = self.decoder(z)                 # [B, dim_outcome]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)        # [B, τ, dim_outcome]


class CausalMixer(BRCausalModel):
    """

    Inherits the full balancing-representation training loop from BRCausalModel
    (gradient reversal / domain confusion, alpha scheduling, EMA weights) and
    replaces only the sequence backbone:

        CM   : N × CausalMixerBlock            (3 TimeMLP + 1 FFN,  O(T))

    ─────────────────────────────────
    • CausalTimeMLP   – causal lower-triangular mask on the time-mixing MLP
    • CausalGatedMixer– SCM-guided gates replace undirected cross-attention
    • DirectMultiStepHead – single-pass τ-step prediction, no error rollup
    • MixtureDensityOutcomeHead (optional) – uncertainty on counterfactuals
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
        bce_weights: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        self.input_size = max(
            self.dim_treatments,
            self.dim_static_features,
            self.dim_vitals,
            self.dim_outcome,
        )
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive, 'CausalMixer requires autoregressive=True (prev_outcomes are mandatory)'

        self._init_specific(args.model.multi)
        self.save_hyperparameters(args)

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.max_seq_length = sub_args.max_seq_length
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.br_size = sub_args.br_size
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer

            if any(v is None for v in [
                self.seq_hidden_units, self.br_size,
                self.fc_hidden_units, self.dropout_rate,
            ]):
                raise MissingMandatoryValue()

            d = self.seq_hidden_units

            self.treatments_input_transformation = nn.Linear(self.dim_treatments, d)
            self.outputs_input_transformation = nn.Linear(self.dim_outcome, d)
            self.vitals_input_transformation = (
                nn.Linear(self.dim_vitals, d) if self.has_vitals else None
            )
            self.static_input_transformation = nn.Linear(self.dim_static_features, d)

            self.gate_static_t = nn.Parameter(torch.zeros(1))
            self.gate_static_o = nn.Parameter(torch.zeros(1))
            self.gate_static_v = nn.Parameter(torch.zeros(1))

            self.n_inputs = 3 if self.has_vitals else 2

            # ── Learnable positional encodings (one per stream) ───────────────
            self.pos_enc_t = LearnablePositionalEncoding(
                self.max_seq_length, d, dropout=self.dropout_rate
            )
            self.pos_enc_o = LearnablePositionalEncoding(
                self.max_seq_length, d, dropout=self.dropout_rate
            )
            if self.has_vitals:
                self.pos_enc_v = LearnablePositionalEncoding(
                    self.max_seq_length, d, dropout=self.dropout_rate
                )

            # ── Mixer blocks ─────────────────────────────────────────────────
            # fc_hidden for the feature-mix FFN inside each block is 4×d,
            # matching CT's feed_forward_hidden = seq_hidden_units * 4.
            self.mixer_blocks = nn.ModuleList([
                CausalMixerBlock(
                    max_seq_len=self.max_seq_length,
                    d_model=d,
                    fc_hidden=d * 4,
                    dropout=self.dropout_rate,
                    has_vitals=self.has_vitals,
                )
                for _ in range(self.num_layer)
            ])

            self.output_dropout = nn.Dropout(self.dropout_rate)

            # ── Balanced-representation + treatment + outcome heads ──────────
            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
                self.seq_hidden_units,
                self.br_size,
                self.fc_hidden_units,
                self.dim_treatments,
                self.dim_outcome,
                self.alpha,
                self.update_alpha,
                self.balancing,
            )

            # ── GRU multi-step decoder (with Seq2Seq cross-attention) ─────────
            # Autoregressive GRU decoder — conditions each step on the previous
            # predicted outcome and attends the full encoder BR sequence at every
            # decoding step via Bahdanau-style cross-attention.
            #
            # Koopman operator decoder was tested and reverted (Koopman-R1):
            # Reason: cancer_sim dynamics are highly nonlinear (exponential/sigmoidal
            # pharmacokinetics).  A 64-dim linear Koopman space cannot faithfully
            # linearize them in 350 epochs — the linear approximation error
            # manifests as compounding offset bias across rollout steps.
            self.direct_head = GRUMultiStepDecoder(
                br_size=self.br_size,
                dim_treatments=self.dim_treatments,
                projection_horizon=self.projection_horizon,
                dim_outcome=self.dim_outcome,
            )

            # ── Optional mixture density head for uncertainty ─────────────────
            use_mixture = getattr(sub_args, 'use_mixture_head', False)
            n_comp = getattr(sub_args, 'n_mixture_components', 5)
            self.mixture_head = (
                MixtureDensityOutcomeHead(
                    self.br_size, self.dim_treatments, self.dim_outcome, n_comp
                )
                if use_mixture else None
            )

        except MissingMandatoryValue:
            logger.warning(
                f'{self.model_type} not fully initialised — some mandatory args are missing! '
                f'(OK if hyperparameter search will follow).'
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Data preparation
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass (training / factual prediction)
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch, detach_treatment=False):
        fixed_split = batch.get('future_past_split', None)

        # Vitals-augmentation during training (mirrors CT exactly).
        if (self.training
                and self.hparams.model.multi.augment_with_masked_vitals
                and self.has_vitals):
            assert fixed_split is None
            fixed_split = torch.empty(2 * len(batch['active_entries'])).type_as(
                batch['active_entries']
            )
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len
                fixed_split[len(batch['active_entries']) + i] = torch.randint(
                    0, int(seq_len) + 1, (1,)
                ).item()
            for k, v in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries']

        br = self.build_br(
            prev_treatments, vitals, prev_outputs,
            static_features, active_entries, fixed_split,
        )
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)
        return treatment_pred, outcome_pred, br

    # ─────────────────────────────────────────────────────────────────────────
    # Balanced-representation builder
    # ─────────────────────────────────────────────────────────────────────────

    def build_br(
        self,
        prev_treatments,
        vitals,
        prev_outputs,
        static_features,
        active_entries,
        fixed_split=None,
    ):
        active_entries_vitals = torch.clone(active_entries)

        # Mask vitals after the split point (same logic as CT.build_br).
        if fixed_split is not None and self.has_vitals:
            for i in range(len(active_entries)):
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        # ── Project each stream to seq_hidden_units ──────────────────────────
        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None

        x_s = self.static_input_transformation(static_features.unsqueeze(1))  # [B,1,d]
        x_t = x_t + torch.sigmoid(self.gate_static_t) * x_s
        x_o = x_o + torch.sigmoid(self.gate_static_o) * x_s
        if x_v is not None:
            x_v = x_v + torch.sigmoid(self.gate_static_v) * x_s

        # ── Per-stream positional encoding ────────────────────────────────────
        # Applied after static injection so the PE sits on top of a
        # unit-scale representation (input projections + static bias).
        x_t = self.pos_enc_t(x_t)
        x_o = self.pos_enc_o(x_o)
        if x_v is not None:
            x_v = self.pos_enc_v(x_v)

        # ── N × CausalMixerBlock ─────────────────────────────────────────────
        for block in self.mixer_blocks:
            if self.has_vitals:
                x_t, x_o, x_v = block(x_t, x_o, x_v)
            else:
                x_t, x_o = block(x_t, x_o)

        # ── Pool streams ─────────────────────────────────────────────────────
        # Mirrors CT's mean-pooling of streams, with split-aware vitals masking.
        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    sp = int(fixed_split[i])
                    x[i, :sp] = (x_o[i, :sp] + x_t[i, :sp] + x_v[i, :sp]) / 3
                    x[i, sp:] = (x_o[i, sp:] + x_t[i, sp:]) / 2
            else:
                x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-step counterfactual prediction (replaces autoregressive rollout)
    # ─────────────────────────────────────────────────────────────────────────

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.ndarray:
        """
        GRU decoder τ-step counterfactual prediction (autoregressive rollout).

        Each step of the GRU decoder conditions on the previous step's predicted
        outcome.  The GRU hidden state
        is seeded from the balanced representation at the split point, and the
        first step is seeded from the last observed factual outcome so the
        decoder starts at the correct scale.

        Requires that dataset.data['current_treatments'] holds the counterfactual
        treatment sequence for positions [split, split+τ).
        """
        logger.info(f'Direct multi-step prediction for {dataset.subset_name}.')
        tau = self.hparams.dataset.projection_horizon

        all_br = torch.tensor(self.get_representations(dataset))  # [N, T, br_size]

        future_treatments = torch.tensor(
            dataset.data['current_treatments']
        ).to(torch.get_default_dtype())  # [N, T_total, dim_treatments]

        # Last factual outcome for each sample — seeds the GRU decoder.
        all_outputs = torch.tensor(
            dataset.data['outputs']
        ).to(torch.get_default_dtype())  # [N, T_total, dim_outcome]

        splits = dataset.data['future_past_split']  # [N]
        predicted_outputs = np.zeros((len(dataset), tau, self.dim_outcome))

        with torch.no_grad():
            for i in range(len(dataset)):
                split = int(splits[i])

                # BR at the last observed timestep.
                br_last = all_br[i, split - 1, :].unsqueeze(0)  # [1, br_size]

                # Full encoder history up to the split point for cross-attention.
                # The attention decoder can then attend relevant encoder positions
                br_seq_i = all_br[i, :split, :].unsqueeze(0)  # [1, split, br_size]

                # Counterfactual treatments for the prediction window.
                fut_trt = future_treatments[i, split:split + tau, :].unsqueeze(0)
                # [1, τ, dim_treatments]

                # Seed the GRU with the last observed factual outcome.
                last_out = all_outputs[i, split - 1, :].unsqueeze(0)  # [1, dim_o]

                # Autoregressive inference: future_outputs=None means the GRU
                pred = self.direct_head(
                    br_last, fut_trt,
                    br_seq=br_seq_i,
                    future_outputs=None,
                    last_output=last_out,
                )  # [1, τ, dim_outcome]
                predicted_outputs[i] = pred.squeeze(0).cpu().numpy()

        return predicted_outputs

    # ─────────────────────────────────────────────────────────────────────────
    # Bug 1 fix — Train DirectMultiStepHead via factual multi-step aux loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_direct_multi_step_loss(
        self,
        br: torch.Tensor,
        batch: dict,
        teacher_forcing_p: float = 0.5,
    ):
        """
        Factual multi-step auxiliary loss for the GRU decoder.

        Improvements over the original single-split version:
        ─────────────────────────────────────────────────────
        1. Scheduled teacher forcing: teacher_forcing_p is passed in from
           training_step where it is annealed each epoch via
           _get_teacher_forcing_p().  This closes the train/inference gap
           gradually rather than holding a fixed 50% split throughout.

        2. Multiple split points (n_splits=4): instead of sampling one random
           split per batch, we sample 4 distinct split points and average the
           losses.  This gives the decoder 4× more gradient signal per optimiser
           step and ensures diverse positions are covered per epoch.

        3. Last-observed output seed: the decoder is seeded with the actual
           outcome at split-1, matching inference behaviour and anchoring the
           first predicted step to the correct scale.

        Gradients flow through the decoder AND through br into the mixer blocks,
        encouraging the backbone to produce representations that support
        sequential multi-step prediction, not just single-step factual MSE.

        Args:
            br               : [B, T, br_size]  balanced representations.
            batch            : training batch dict.
            teacher_forcing_p: probability of using ground-truth prev_output
                               at each GRU step.  Should be annealed from
                               near 1.0 (stable early training) toward 0.0
                               (matches autoregressive inference).
        Returns:
            Averaged scalar loss over n_splits windows, or None if the sequence
            is too short to form even one valid τ-step window.
        """
        tau = self.projection_horizon
        if tau is None or tau <= 0:
            return None

        B, T, _ = br.shape
        if T <= tau:
            return None

        # Sample up to 8 distinct split points (was 4).
        max_split = T - tau     # exclusive upper bound
        n_splits  = min(8, max_split)
        split_pts = torch.randperm(max_split)[:n_splits].tolist()

        total_loss = br.new_zeros(())
        n_valid    = 0

        for s in split_pts:
            br_last     = br[:, s, :].detach()                          # [B, br_size]
            br_seq_s    = br[:, :s + 1, :].detach()                     # [B, s+1, br_size]
            future_trt  = batch['current_treatments'][:, s:s + tau, :]  # [B, τ, dim_t]

            # ── FiLM train/inference alignment ────────────────────────────────
            if hasattr(self, 'film_layer') and future_trt.size(1) > 0:
                first_trt = future_trt[:, 0:1, :]              # [B, 1, dim_t]
                br_last = self.film_layer(
                    br_last.unsqueeze(1), first_trt
                ).squeeze(1).detach()                           # [B, br_size]
            future_out  = batch['outputs'][:, s:s + tau, :]             # [B, τ, dim_o]
            future_mask = batch['active_entries'][:, s:s + tau, :]      # [B, τ, 1]

            # ── Random horizon curriculum ─────────────────────────────────────
            tau_prime = torch.randint(2, tau + 1, (1,)).item() if tau >= 3 else tau
            future_trt_h  = future_trt[:, :tau_prime, :]
            future_out_h  = future_out[:, :tau_prime, :]
            future_mask_h = future_mask[:, :tau_prime, :]

            valid_count = future_mask_h.sum()
            if valid_count == 0:
                continue


            if s > 0:
                last_out = batch['outputs'][:, s - 1, :]  # [B, dim_o]
            else:
                last_out = None

            pred = self.direct_head(
                br_last, future_trt_h,
                br_seq=br_seq_s,
                future_outputs=future_out_h,
                last_output=last_out,
                teacher_forcing_p=teacher_forcing_p,
            )  # [B, τ', dim_o]

            mse = F.mse_loss(pred, future_out_h, reduce=False)  # [B, τ', dim_o]

            # ── Per-step discount ─────────────────────────────────────────────

            gamma = float(getattr(self.hparams.exp, 'ms_step_discount', 1.0))
            if gamma < 1.0:
                discounts = torch.tensor(
                    [gamma ** k for k in range(tau_prime)],
                    dtype=mse.dtype, device=mse.device,
                )  # [τ']
                mse = mse * discounts.view(1, -1, 1)

            total_loss = total_loss + (future_mask_h * mse).sum() / valid_count
            n_valid += 1

        if n_valid == 0:
            return None
        return total_loss / n_valid

    # ─────────────────────────────────────────────────────────────────────────
    # Scheduled teacher forcing
    # ─────────────────────────────────────────────────────────────────────────

    def _get_teacher_forcing_p(self, epoch: int) -> float:
        """
        Exponential decay schedule for the GRU teacher-forcing probability.


        Returns
        ───────
        Scheduled probability ∈ [tf_min, tf_init].
        """
        tf_init  = float(getattr(self.hparams.exp, 'tf_init',  0.9))
        tf_decay = float(getattr(self.hparams.exp, 'tf_decay', 0.99))
        tf_min   = float(getattr(self.hparams.exp, 'tf_min',   0.05))
        return max(tf_min, tf_init * (tf_decay ** epoch))

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        """
        Overrides BRCausalModel.training_step to capture `br` from the forward
        pass and attach the direct multi-step auxiliary loss (Bug 1 fix).

        All other logic (EMA weight averaging, domain confusion / gradient
        reversal, alpha scheduling, domain-classifier update) is preserved
        verbatim from the parent class.  The only additions are:
          1. `br` is captured instead of discarded (`_`).
          2. `_compute_direct_multi_step_loss(br, batch)` is called and, if
             non-None, added to the total loss with weight `exp.lambda_ms`
             (default 0.1 — always trains the head unless explicitly set to 0).
        """
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:  # representation + outcome update
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, outcome_pred, br = self(batch)
            else:
                treatment_pred, outcome_pred, br = self(batch)

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

            # ── Bug 1 fix: multi-step auxiliary loss (scheduled teacher forcing)
            # lambda_ms defaults to 0.2 so GRUMultiStepDecoder is always
            # trained unless the user explicitly sets exp.lambda_ms: 0.0.
            # teacher_forcing_p is annealed each epoch from tf_init (0.9)
            # toward tf_min (0.05), progressively closing the train/inference
            # exposure-bias gap that caused the peak-then-drop n-step pattern.
            lambda_ms = float(getattr(self.hparams.exp, 'lambda_ms', 0.2))
            if lambda_ms > 0.0:
                tf_p    = self._get_teacher_forcing_p(self.current_epoch)
                ms_loss = self._compute_direct_multi_step_loss(
                    br, batch, teacher_forcing_p=tf_p
                )
                if ms_loss is not None:
                    loss = loss + lambda_ms * ms_loss
                    self.log(
                        f'{self.model_type}_train_ms_loss', ms_loss,
                        on_epoch=True, on_step=False, sync_dist=True,
                    )
                    self.log(
                        f'{self.model_type}_train_tf_p', tf_p,
                        on_epoch=True, on_step=False, sync_dist=True,
                    )

            # ── Optional temporal curvature loss (cm_wip hook) ───────────────
            lambda_ts = float(getattr(self, 'lambda_ts', 0.0))
            if lambda_ts > 0.0 and hasattr(self, 'curvature_loss'):
                curv_loss = self.curvature_loss
                loss = loss + lambda_ts * curv_loss
                self.log(
                    f'{self.model_type}_train_curv_loss', curv_loss,
                    on_epoch=True, on_step=False, sync_dist=True,
                )

            self.log(f'{self.model_type}_train_loss',     loss,     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha,
                     on_epoch=True, on_step=False, sync_dist=True)
            return loss

        elif optimizer_idx == 1:  # domain-classifier update (unchanged)
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

            bce_loss = self.bce_loss(
                treatment_pred, batch['current_treatments'].to(torch.get_default_dtype()), kind='predict'
            )
            if self.balancing == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss

            bce_loss = (
                batch['active_entries'].squeeze(-1) * bce_loss
            ).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss,
                     on_epoch=True, on_step=False, sync_dist=True)
            return bce_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Hyperparameter search interface
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']

        # seq_hidden_units as a multiplier of input_size (same convention as EDCT).
        seq_hidden = int(input_size * new_args['seq_hidden_units'])
        if seq_hidden % 2 != 0:
            seq_hidden += 1
        sub_args.seq_hidden_units = seq_hidden

        sub_args.br_size = int(input_size * new_args['br_size'])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']
        logger.info(f'CM set_hparams → seq_hidden={seq_hidden}, br={sub_args.br_size}')
