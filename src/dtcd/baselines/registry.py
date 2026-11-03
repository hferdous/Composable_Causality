from __future__ import annotations
from typing import Dict, Type, Any
from functools import partial

from dtcd.baselines.base import BaseDiscoveryMethod, BaseCATEMethod

# Third-party baselines live behind the optional [discovery], [deep] and [cate]
# extras. Importing this registry must not fail on a core-only install, so every
# wrapper that pulls an optional dependency is guarded. Methods that fail to
# import resolve to None and raise an actionable error only if requested.
try:
    from dtcd.baselines.grangerf_wrapper import GrangerF
except ImportError:
    GrangerF = None

try:
    from dtcd.baselines.varlingam_wrapper import VARLiNGAM
except ImportError:
    VARLiNGAM = None

try:
    from dtcd.baselines.pcmci_wrapper import PCMCI_Base
    from dtcd.baselines.pcmciplus_wrapper import PCMCIplus_ParCorr
except ImportError:
    PCMCI_Base = PCMCIplus_ParCorr = None
try:
    from dtcd.baselines.dynotears_wrapper import DYNOTEARS
except ImportError:
    DYNOTEARS = None

try:
    from dtcd.baselines.rhino_wrapper import Rhino
    from dtcd.baselines.neuralgc_wrapper import NeuralGC
    from dtcd.baselines.deep_wrappers import TCDF_Wrapper, ACD_Wrapper, CausalTransformerWrapper
except ImportError:
    Rhino = NeuralGC = TCDF_Wrapper = ACD_Wrapper = CausalTransformerWrapper = None

try:
    from dtcd.baselines.lpcmci_wrapper import CDNOD_Wrapper, LPCMCI_Wrapper
except ImportError:
    CDNOD_Wrapper = LPCMCI_Wrapper = None

try:
    from dtcd.baselines.headline_methods import UnCLe, VARLiNGAM_imp, CUTSPlus
except ImportError:
    UnCLe = VARLiNGAM_imp = CUTSPlus = None

try:
    from dtcd.baselines.catenets_wrapper import CATENets_Wrapper
except ImportError:
    CATENets_Wrapper = None

try:
    from dtcd.baselines.xlearner_wrapper import XLearnerWrapper
except ImportError:
    XLearnerWrapper = None
from dtcd.baselines.sortnregress import SortnRegress, RandomRegress

# Registry for Discovery Methods
DISCOVERY_REGISTRY: Dict[str, Any] = {
    "GrangerF": GrangerF,
    "VARLiNGAM": VARLiNGAM,
    "PCMCI_ParCorr": partial(PCMCI_Base, ci_test="ParCorr") if PCMCI_Base else None,
    "PCMCI_GPDC": partial(PCMCI_Base, ci_test="GPDC") if PCMCI_Base else None,
    "PCMCI_CMIknn": partial(PCMCI_Base, ci_test="CMIknn") if PCMCI_Base else None,
    "PCMCIplus_ParCorr": PCMCIplus_ParCorr,
    "DYNOTEARS": DYNOTEARS,
    "Rhino": Rhino,
    "NeuralGC_cMLP": partial(NeuralGC, model_type="cMLP") if NeuralGC else None,
    "NeuralGC_cLSTM": partial(NeuralGC, model_type="cLSTM") if NeuralGC else None,
    "CDNOD": CDNOD_Wrapper,
    "LPCMCI": LPCMCI_Wrapper,
    "TCDF": TCDF_Wrapper,
    "ACD": ACD_Wrapper,
    "UnCLe": UnCLe,
    "VARLiNGAM_imp": VARLiNGAM_imp,
    "CUTS+": CUTSPlus,
    # Varsortability controls. Not causal discovery methods; they quantify how
    # much of a benchmark's difficulty is carried by marginal variance alone.
    "SortnRegress": SortnRegress,
    "RandomRegress": RandomRegress,
}

# Registry for CATE Methods
CATE_REGISTRY: Dict[str, Any] = {
    "TARNet": partial(CATENets_Wrapper, model_type="TARNet") if CATENets_Wrapper else None,
    "SNet": partial(CATENets_Wrapper, model_type="SNet") if CATENets_Wrapper else None,
    "FlexTENet": partial(CATENets_Wrapper, model_type="FlexTENet") if CATENets_Wrapper else None,
    "XLearner": XLearnerWrapper,
    "CausalTransformer": CausalTransformerWrapper,
}

def get_discovery_method(name: str, max_lag: int, seed: int = 0, **hp: Any) -> BaseDiscoveryMethod:
    if name not in DISCOVERY_REGISTRY:
        raise ValueError(f"Method {name} not found.")
    m = DISCOVERY_REGISTRY[name]
    if m is None:
        raise ImportError(
            f"Method '{name}' requires an optional dependency that is not "
            f"installed. Try: pip install 'dtcd[discovery,deep]'"
        )
    if isinstance(m, partial):
        return m(max_lag=max_lag, seed=seed, **hp)
    return m(max_lag=max_lag, seed=seed, **hp)

def get_cate_method(name: str, seed: int = 0, **hp: Any) -> BaseCATEMethod:
    if name not in CATE_REGISTRY:
        raise ValueError(f"Method {name} not found.")
    m = CATE_REGISTRY[name]
    if m is None:
        raise ImportError(
            f"CATE method '{name}' requires an optional dependency that is not "
            f"installed. Try: pip install 'dtcd[cate]'"
        )
    if isinstance(m, partial):
        return m(seed=seed, **hp)
    return m(seed=seed, **hp)
