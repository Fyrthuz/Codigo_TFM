import numpy as np

from src.utils.crf import refine_with_crf_uncertainty


def _dice(a, b):
    a = a > 0.5
    b = b > 0.5
    return 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-8)


class TestRefineCRF:
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
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-6)

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

    def test_prob_map_on_different_grid_than_image(self):
        # Foundation models resize internally: image at 128x128, probability
        # map at the model's native output size. The edge-stopping gate must
        # be resized to the probability grid (used to be a shape mismatch).
        image = np.random.rand(3, 128, 128).astype(np.float32)
        prob_map = np.random.rand(64, 64).astype(np.float32)
        uncertainty_map = np.random.rand(64, 64).astype(np.float32)
        probs, seg, uncert = refine_with_crf_uncertainty(
            image, prob_map, uncertainty_map, n_iters=1
        )
        assert probs.shape == (2, 64, 64)
        assert seg.shape == (64, 64)

    def test_clean_disk_is_preserved(self):
        """Regression: the previous implementation eroded the mask by ~40%.

        A soft-edged disk must keep both its overlap and its area after
        refinement. With the old (buggy) refinement this yielded
        dice ~0.73 with an area ratio of ~0.57, and some real samples
        collapsed to an empty mask.
        """
        h = w = 256
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
        gt = (r < 40).astype(np.float32)
        prob = 1.0 / (1.0 + np.exp((r - 40) / 2.0))
        unc = -(prob * np.log(prob + 1e-8) + (1 - prob) * np.log(1 - prob + 1e-8))
        image = np.full((3, h, w), 100, dtype=np.uint8) / 255.0

        _, seg, _ = refine_with_crf_uncertainty(image, prob, unc)

        ratio = (seg > 0).sum() / max((prob > 0.5).sum(), 1)
        assert _dice(seg, gt) > 0.93
        assert 0.85 <= ratio <= 1.10
