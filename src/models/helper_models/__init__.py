"""
helper_models — internal building blocks for the SSM model family.

Contents
--------
causal_mamba_layer.py : CausalMambaLayer — selective-state-space scan (Mamba-style)
causal_mixer_block.py : CausalGatedMixer, CausalMixerBlock — SCM-guided cross-stream mixing
cmcpd.py              : ParallelMultiStepDecoder — τ independent decoder heads
cmcp.py               : CPCHead, LocalInfoMaxHead — contrastive regularisation heads
cmc.py                : CausalMixerCAETC — causal mixer with domain confusion
cm.py                 : CausalMixer — base causal mixer (prior architecture)

These are imported by the top-level model files (cssd.py, csspd.py, utils_ssm.py, etc.)
via explicit ``from src.models.helper_models.<module> import <Class>`` calls.
"""
