import numpy as np
import pytest

from src.utils.metrics import compute_iou, compute_dice, compute_metrics, certainty_score


class TestComputeIoU:
    def test_perfect_overlap(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))
        assert compute_iou(pred, target) == 1.0

    def test_no_overlap(self):
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))
        assert compute_iou(pred, target) == 0.0

    def test_half_overlap(self):
        pred = np.zeros((10, 10))
        pred[:5, :] = 1
        target = np.zeros((10, 10))
        target[:, :5] = 1
        iou = compute_iou(pred, target)
        assert 0.0 < iou < 1.0

    def test_both_empty(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))
        assert compute_iou(pred, target) == 1.0

    def test_threshold(self):
        pred = np.ones((10, 10)) * 0.3
        target = np.ones((10, 10))
        assert compute_iou(pred, target, threshold=0.5) == 0.0


class TestComputeDice:
    def test_perfect_overlap(self):
        pred = np.ones((10, 10))
        target = np.ones((10, 10))
        assert compute_dice(pred, target) == 1.0

    def test_no_overlap(self):
        pred = np.ones((10, 10))
        target = np.zeros((10, 10))
        assert compute_dice(pred, target) == 0.0

    def test_both_empty(self):
        pred = np.zeros((10, 10))
        target = np.zeros((10, 10))
        assert compute_dice(pred, target) == 1.0


class TestComputeMetrics:
    def test_perfect_prediction(self):
        prob = np.ones((10, 10))
        gt = np.ones((10, 10))
        metrics = compute_metrics(prob, gt)
        assert metrics["nll"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["brier"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["precision"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["recall"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["ece"] == pytest.approx(0.0, abs=1e-2)

    def test_worst_prediction(self):
        prob = np.zeros((10, 10))
        gt = np.ones((10, 10))
        metrics = compute_metrics(prob, gt)
        assert metrics["accuracy"] == 0.0
        assert metrics["recall"] == 0.0

    def test_half_correct(self):
        prob = np.zeros((10, 10))
        prob[:5, :] = 0.9
        gt = np.zeros((10, 10))
        gt[:5, :] = 1
        metrics = compute_metrics(prob, gt)
        assert 0.9 <= metrics["accuracy"] <= 1.0  # most pixels are correctly 0, rest are close to 1

    def test_probability_clipping(self):
        prob = np.ones((10, 10)) * 1e-10
        prob[0, 0] = 0.0
        prob[1, 1] = 1.0
        gt = np.ones((10, 10))
        metrics = compute_metrics(prob, gt)
        assert np.isfinite(metrics["nll"])


class TestCertaintyScore:
    def test_max_certainty(self):
        uncertainty = np.zeros((10, 10))
        gt = np.ones((10, 10))
        score = certainty_score(uncertainty, gt)
        assert score == pytest.approx(1.0, abs=1e-2)

    def test_max_uncertainty(self):
        uncertainty = np.ones((10, 10)) * np.log(2)
        gt = np.ones((10, 10))
        score = certainty_score(uncertainty, gt)
        assert score == pytest.approx(0.0, abs=1e-2)

    def test_empty_gt_returns_nan(self):
        uncertainty = np.zeros((10, 10))
        gt = np.zeros((10, 10))
        assert np.isnan(certainty_score(uncertainty, gt))
