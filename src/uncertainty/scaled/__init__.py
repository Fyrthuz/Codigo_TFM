from .cross_entropy import CalibratedMCDropout as CrossEntropyMCDropout
from .nll_relaxed import SegCalibratedMCDropout as NLLMCDropout
from .hamming import SegCalibratedMCDropout as HammingMCDropout

__all__ = ["CrossEntropyMCDropout", "NLLMCDropout", "HammingMCDropout"]
