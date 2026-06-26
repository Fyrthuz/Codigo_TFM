import numpy as np
import pytest

from src.utils.fusion import weighted_average_with_uncertainty, dynamic_threshold_multiclass


class TestWeightedAverage:
    def test_identical_inputs(self):
        prob = np.ones((32, 32)) * 0.8
        uncert = np.ones((32, 32)) * 0.2
        consensus_prob, consensus_uncert, consensus_mask = weighted_average_with_uncertainty(
            prob, uncert, prob, uncert, prob, uncert,
            weighting_method="inverse",
        )
        assert np.allclose(consensus_prob, prob)
        assert np.allclose(consensus_uncert, uncert)
        assert consensus_mask.dtype == np.uint8

    def test_one_certain_source(self):
        prob_certain = np.ones((16, 16))
        uncert_certain = np.zeros((16, 16))
        prob_uncertain = np.zeros((16, 16))
        uncert_uncertain = np.ones((16, 16))

        consensus_prob, _, _ = weighted_average_with_uncertainty(
            prob_certain, uncert_certain,
            prob_uncertain, uncert_uncertain,
            prob_uncertain, uncert_uncertain,
            weighting_method="inverse",
        )
        assert np.allclose(consensus_prob, 1.0)

    def test_methods(self):
        prob = np.ones((16, 16)) * 0.7
        uncert = np.ones((16, 16)) * 0.3
        for method in ["inverse", "exponential", "powerlaw"]:
            cp, cu, cm = weighted_average_with_uncertainty(
                prob, uncert, prob, uncert, prob, uncert,
                weighting_method=method,
            )
            assert cp.shape == (16, 16)
            assert cu.shape == (16, 16)

    def test_threshold_methods(self):
        prob = np.ones((16, 16)) * 0.7
        uncert = np.ones((16, 16)) * 0.3
        for thresh in ["naive", "otsu", "percentile"]:
            cp, cu, cm = weighted_average_with_uncertainty(
                prob, uncert, prob, uncert, prob, uncert,
                threshold_method=thresh,
            )
            assert cm.dtype == np.uint8


class TestDynamicThreshold:
    def test_otsu_binary(self):
        probs = np.random.rand(64, 64)
        thresholds = dynamic_threshold_multiclass(probs, method="otsu")
        assert len(thresholds) == 1
        assert 0 <= thresholds[0] <= 1

    def test_percentile(self):
        probs = np.random.rand(64, 64)
        thresholds = dynamic_threshold_multiclass(probs, method="percentile", percentile=75)
        assert len(thresholds) == 1
        assert thresholds[0] >= np.percentile(probs, 50)

    def test_multiclass(self):
        probs = np.random.rand(3, 64, 64)
        thresholds = dynamic_threshold_multiclass(probs, method="otsu")
        assert len(thresholds) == 3
