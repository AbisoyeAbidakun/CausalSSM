"""
CausalStateSpaceModel — model registry.

Paper: "Causal State-Space Model for Causal Inference: Estimating
Longitudinal Individual Treatment Effects", Abisoye Abidakun, Mingjun
Zhong, Georgios Leontidis.

Proposed models
----------------
CSSD  : Selective-SSM encoder + parallel multi-step decoder + domain
        confusion (Section 4, "The Training Objective function of CSSD")
CSSPD : CSSD + CPC + LIM (Section 4.6-4.7 / Section 5) — the paper's main
        proposed model
CHSD  : CSSD with the CCH hybrid encoder (Section 4, architectural ablation)
CHSPD : CHSD + CPC + LIM (Section 4, combined ablation)

Baselines evaluated in the paper (Section "Baseline Methods")
---------------------------------------------------------------
RMSN  : LSTM + propensity weighting (Lim et al., 2018)
CRN   : GRU + gradient reversal (Bica et al., 2020)
G-Net : LSTM g-computation (Li et al., 2021)
CT    : Three-stream self-attention + domain confusion (Melnychuk, Frauen
        & Feuerriegel, 2022) — the primary baseline. Code in ct.py/edct.py
        is adapted from the original authors' public repository (see
        LICENSE).

Other baseline infrastructure present in this repo but not part of the
paper's reported comparisons
-----------------------------------------------------------------------
EDCT  : Encoder-decoder variant of CT (used internally by CRN's two-phase
        training, not reported as a standalone comparison row)
MSM   : Linear IPW marginal structural model (Robins, Hernan & Brumback,
        2000). Runnable via runnables/train_msm.py but not one of the four
        baselines the paper's Results section reports against.
"""

from src.models.time_varying_model import TimeVaryingCausalModel, BRCausalModel

# Proposed models — new names
from src.models.cssd import CSSD  # noqa: F401
from src.models.csspd import CSSPD  # noqa: F401
from src.models.chsd import CHSD  # noqa: F401
from src.models.chspd import CHSPD  # noqa: F401

# Shared SSM base class for the proposed-model family
from src.models.sst import SST  # noqa: F401

# Baselines
from src.models.rmsn import RMSN, RMSNPropensityNetworkTreatment, RMSNPropensityNetworkHistory, RMSNEncoder, RMSNDecoder  # noqa: F401,E501
from src.models.crn import CRN, CRNEncoder, CRNDecoder  # noqa: F401
from src.models.gnet import GNet  # noqa: F401
from src.models.edct import EDCT, EDCTEncoder, EDCTDecoder  # noqa: F401
from src.models.ct import CT  # noqa: F401
from src.models.msm import MSM  # noqa: F401

__all__ = [
    # Proposed
    "CSSD", "CSSPD", "CHSD", "CHSPD", "SST",
    # Baselines reported in the paper
    "RMSN", "RMSNPropensityNetworkTreatment", "RMSNPropensityNetworkHistory", "RMSNEncoder", "RMSNDecoder",
    "CRN", "CRNEncoder", "CRNDecoder",
    "GNet",
    "CT",
    # Other baseline infrastructure (not in the paper's reported comparisons)
    "EDCT", "EDCTEncoder", "EDCTDecoder", "MSM",
    # Base classes
    "TimeVaryingCausalModel", "BRCausalModel",
]
