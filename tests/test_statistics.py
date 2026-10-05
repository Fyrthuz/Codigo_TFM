import numpy as np
import pandas as pd
import pytest

from src.utils.statistics import compare_methods


def _synthetic(n_patients=12, slices=5, delta=0.0, noise=0.01, seed=0):
    """Paired synthetic data: slices clustered in patients.

    Each slice has a patient-level base value plus per-slice noise; the
    'fusion' method adds `delta` on top of the shared base.
    """
    rng = np.random.default_rng(seed)
    rows = []
    patient_map = {}
    for p in range(n_patients):
        base = rng.uniform(0.7, 0.95)
        for s in range(slices):
            name = f"sample_{p * slices + s}"
            patient_map[name] = f"P{p:02d}"
            rows.append({"sample": name, "method": "fusion", "metric": "dice",
                         "value": base + delta + rng.normal(0, noise)})
            rows.append({"sample": name, "method": "normal", "metric": "dice",
                         "value": base + rng.normal(0, noise)})
    return pd.DataFrame(rows), patient_map


class TestCompareMethods:
    def test_detects_positive_effect(self):
        df, pm = _synthetic(delta=0.05, noise=0.005)
        res = compare_methods(df, pm, "fusion", "normal", "dice", n_boot=500, seed=42)
        assert res["delta"] == pytest.approx(0.05, abs=0.01)
        assert res["ci_low"] > 0.0
        assert res["p_wilcoxon"] < 0.01
        assert res["n_patients"] == 12
        assert res["n_slices"] == 60
        assert res["pct_patients_improved"] == 100.0

    def test_null_effect_not_significant(self):
        df, pm = _synthetic(delta=0.0, noise=0.01)
        res = compare_methods(df, pm, "fusion", "normal", "dice", n_boot=500, seed=42)
        assert abs(res["delta"]) < 0.01
        assert res["p_wilcoxon"] > 0.05

    def test_ignores_samples_without_patient(self):
        df, pm = _synthetic()
        df = df[df["sample"] != "sample_0"]
        res = compare_methods(df, pm, "fusion", "normal", "dice", n_boot=200, seed=1)
        # sample_0 was dropped from the frame, patient P00 keeps its other slices
        assert res["n_slices"] == 59
        assert res["n_patients"] == 12

    def test_patient_clustering_reflected_in_n(self):
        df, pm = _synthetic(n_patients=6, slices=8)
        res = compare_methods(df, pm, "fusion", "normal", "dice", n_boot=200, seed=2)
        assert res["n_patients"] == 6
        assert res["n_slices"] == 48
