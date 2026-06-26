import numpy as np
import pytest

try:
    import pydensecrf
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

from src.utils.crf import refine_with_crf_uncertainty


@pytest.mark.skipif(not HAS_CRF, reason="pydensecrf not installed")
class TestCRF:
    def test_binary_input_shapes(self):
        image = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        prob_map = np.random.rand(32, 32)
        uncertainty_map = np.random.rand(32, 32)
        probs, seg, uncert = refine_with_crf_uncertainty(
            image, prob_map, uncertainty_map, n_iters=2
        )
        assert probs.shape == (2, 32, 32)
        assert seg.shape == (32, 32)
        assert uncert.shape == (32, 32)

    def test_multiclass_input_shapes(self):
        image = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        prob_map = np.random.rand(3, 32, 32)
        prob_map = prob_map / prob_map.sum(axis=0, keepdims=True)
        uncertainty_map = np.random.rand(32, 32)
        probs, seg, uncert = refine_with_crf_uncertainty(
            image, prob_map, uncertainty_map, n_iters=2
        )
        assert probs.shape == (3, 32, 32)
        assert seg.shape == (32, 32)
        assert uncert.shape == (32, 32)
