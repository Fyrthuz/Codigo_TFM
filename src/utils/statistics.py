"""Patient-level statistical analysis for the segmentation pipelines.

The test sets contain several slices from the same patient, so slices are
not independent samples (pseudo-replication). This module therefore:

  1. Rebuilds the sample -> source-file -> patient mapping and verifies it
     against the saved images (fail loudly instead of silently reporting
     patient statistics computed on a wrong mapping).
  2. Computes paired per-slice differences between two methods
     (e.g. Fusion vs Normal) on the detailed per-sample metrics.
  3. Builds confidence intervals with a cluster (patient-level) bootstrap:
     patients are resampled with replacement and all their slices are
     carried along, mirroring the grouped structure of the data.
  4. Runs Wilcoxon signed-rank tests on per-patient mean differences.

Usage:
    python -m src.utils.statistics --pipeline unet
    python -m src.utils.statistics --pipeline foundation
    python -m src.utils.statistics --pipeline all

Outputs:
    results/visualizations/statistical_tests.csv
    results_foundation_universeg/visualizations/statistical_tests.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sps

from src.utils.dataset import load_test_indices, recover_image_mask_pairs

DEFAULT_COMPARISONS = [
    ("fusion", "normal"),
    ("crf", "fusion"),
    ("tta", "normal"),
    ("noisy", "normal"),
]
DEFAULT_METRICS = ["dice", "iou", "ece", "nll_fg"]


def build_patient_map(root_dir: str, test_indices_path: str, kind: str):
    """Map output sample names (sample_k / test_k / support_k) to patients."""
    pairs, patient_ids = recover_image_mask_pairs(root_dir=root_dir)
    test_idx = load_test_indices(test_indices_path)
    test_set = set(test_idx)

    if kind == "unet":
        mapping = {f"sample_{k}": patient_ids[i] for k, i in enumerate(test_idx)}
    elif kind == "foundation":
        mapping = {f"test_{k}": patient_ids[i] for k, i in enumerate(test_idx)}
        train_idx = [i for i in range(len(pairs)) if i not in test_set][:64]
        mapping.update({f"support_{k}": patient_ids[i] for k, i in enumerate(train_idx)})
    else:
        raise ValueError("kind must be 'unet' or 'foundation'")
    return mapping, pairs, test_idx


def verify_sample_mapping(pairs, test_idx, results_dir: str, kind: str,
                          n_checks: int = 3, max_mae: float = 3.0 / 255):
    """Check that saved original_image.png files match the source images.

    Guards against silently wrong statistics: if the on-disk iteration
    order of the dataset changed since the pipeline run, sample_k would
    map to the wrong patient. Raises RuntimeError on mismatch.
    """
    n = len(test_idx)
    checks = sorted({0, n // 2, n - 1})[:n_checks]
    for k in checks:
        source = np.asarray(Image.open(pairs[test_idx[k]][0]).convert("RGB"), dtype=np.float32)
        name = f"sample_{k}" if kind == "unet" else f"test_{k}"
        saved_path = os.path.join(results_dir, name, "original_image.png")
        saved = np.asarray(Image.open(saved_path), dtype=np.float32)
        if saved.ndim == 3:
            saved = saved[..., 0]
        # Both pipelines save the G (T1c) channel: it is the middle channel
        # of the RGB reading for UNet and the replicated channel for the
        # foundation pipeline, so compare against source[..., 1].
        mae = float(np.mean(np.abs(saved - source[..., 1]))) / 255.0
        if mae > max_mae:
            raise RuntimeError(
                f"Sample mapping mismatch for {name}: MAE={mae:.4f} > {max_mae:.4f}. "
                "The dataset iteration order changed since the pipeline run; "
                "statistics would be computed on the wrong patients."
            )


def compare_methods(df: pd.DataFrame, patient_map: dict, method_a: str, method_b: str,
                    metric: str, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Paired comparison A vs B for one metric with patient-level statistics.

    Returns a dict with the pooled mean difference (slice level), the
    cluster-bootstrap CI, the Wilcoxon signed-rank p-value computed on
    per-patient means, n patients/slices and improvement rates.
    """
    sub = df[df.metric == metric]
    a = sub[sub.method == method_a].set_index("sample")["value"]
    b = sub[sub.method == method_b].set_index("sample")["value"]
    paired = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    paired = paired[paired.index.isin(patient_map)]

    diffs = paired["a"] - paired["b"]
    pid = pd.Series({s: patient_map[s] for s in diffs.index})

    diff_values = diffs.to_numpy()
    patient_labels = pid.to_numpy()
    diffs_by_patient = {p: diff_values[patient_labels == p] for p in pid.unique()}
    keys = list(diffs_by_patient)
    n_patients = len(keys)

    # Cluster bootstrap: resample patients (with replacement), pool their slices
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n_patients, n_patients)
        boot[i] = np.concatenate([diffs_by_patient[keys[j]] for j in sel]).mean()
    ci_low, ci_high = (float(v) for v in np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

    patient_means = np.array([diffs_by_patient[k].mean() for k in keys])
    if np.allclose(patient_means, 0.0):
        p_value = 1.0
    else:
        p_value = float(sps.wilcoxon(patient_means, alternative="two-sided").pvalue)

    sd = float(np.std(patient_means, ddof=1)) if n_patients > 1 else 0.0
    cohen_dz = float(np.mean(patient_means) / sd) if sd > 0 else float("nan")

    return {
        "comparison": f"{method_a} vs {method_b}",
        "metric": metric,
        "delta": float(diffs.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_wilcoxon": p_value,
        "n_patients": n_patients,
        "n_slices": int(len(diffs)),
        "pct_patients_improved": float(100 * np.mean(patient_means > 0)),
        "pct_slices_improved": float(100 * (diffs > 0).mean()),
        "cohen_dz": cohen_dz,
    }


def run_for_pipeline(kind: str, results_dir: str, root_dir: str, test_indices_path: str,
                     comparisons=None, metrics=None, n_boot: int = 10000, seed: int = 42) -> pd.DataFrame:
    comparisons = comparisons or DEFAULT_COMPARISONS
    metrics = metrics or DEFAULT_METRICS

    patient_map, pairs, test_idx = build_patient_map(root_dir, test_indices_path, kind)
    verify_sample_mapping(pairs, test_idx, results_dir, kind)

    csv_path = os.path.join(results_dir, "visualizations", "detailed_metrics.csv")
    df = pd.read_csv(csv_path)
    if kind == "foundation":
        df = df[df["sample"].str.startswith("test_")]

    rows = [
        compare_methods(df, patient_map, a, b, metric, n_boot=n_boot, seed=seed)
        for a, b in comparisons
        for metric in metrics
    ]
    out = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "visualizations", "statistical_tests.csv")
    out.to_csv(out_path, index=False)
    return out


def _format_table(result: pd.DataFrame) -> str:
    lines = [
        "| Comparación | Métrica | Δ medio | IC 95% (bootstrap paciente) | p (Wilcoxon) | n pac. | "
        "Mejora pac. | Mejora slices | d_z |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in result.itertuples():
        lines.append(
            f"| {r.comparison} | {r.metric} | {r.delta:+.4f} | "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_wilcoxon:.3f} | {r.n_patients} | "
            f"{r.pct_patients_improved:.0f}% | {r.pct_slices_improved:.0f}% | {r.cohen_dz:+.2f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Patient-level statistical tests (bootstrap + Wilcoxon)")
    parser.add_argument("--pipeline", choices=["unet", "foundation", "all"], default="all")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--foundation-results-dir", default="./results_foundation_universeg")
    parser.add_argument("--data-root", default="./MRI/filtered_data")
    parser.add_argument("--test-indices", default="test_indices.json")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipelines = []
    if args.pipeline in ("unet", "all"):
        pipelines.append(("unet", args.results_dir))
    if args.pipeline in ("foundation", "all"):
        pipelines.append(("foundation", args.foundation_results_dir))

    for kind, results_dir in pipelines:
        print(f"\n{'=' * 80}\n{kind.upper()} — paired tests at patient level "
              f"({args.n_boot} bootstrap samples)\n{'=' * 80}")
        result = run_for_pipeline(
            kind, results_dir, args.data_root, args.test_indices,
            n_boot=args.n_boot, seed=args.seed,
        )
        print(_format_table(result))
        print(f"\nSaved: {os.path.join(results_dir, 'visualizations', 'statistical_tests.csv')}")


if __name__ == "__main__":
    main()
