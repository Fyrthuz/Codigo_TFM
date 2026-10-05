"""Generate the figures embedded in the README from saved pipeline results.

Figures per experiment (main 1% protocol and 4% thesis protocol, for both
models):

  1. Qualitative panels: rows = selected test samples (easy/typical/hard,
     chosen deterministically from per-sample Dice percentiles), columns =
     the produced masks per method overlaid on the input image (red =
     predicted mask, yellow = ground-truth contour) plus the fusion
     uncertainty map.
  2. Distribution panels: box + jittered points per method for Dice, IoU,
     NLL_fg and ECE.
  3. Forest plots of the patient-level paired tests (bootstrap CI +
     Wilcoxon p-value) from statistical_tests.csv.

Usage:
    python -m src.utils.make_figures            # → docs/*.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

METHOD_COLUMNS = [
    ("normal", "Normal", "original/mask.png"),
    ("mc_dropout", "MC Dropout", "mc_dropout/mean_mask_prediction.png"),
    ("tta", "TTA", "tta/mean_mask_prediction.png"),
    ("noisy", "Noisy", "noisy/mean_mask_prediction.png"),
    ("fusion", "Fusión", "fusion/mask.png"),
    ("crf", "CRF", "refined/probability.png"),  # binarized at 0.5 on load
]
COMPARISON_ORDER = ["fusion vs normal", "tta vs normal", "noisy vs normal", "crf vs fusion"]
DIST_METRICS = [("dice", "Dice"), ("iou", "IoU"), ("nll_fg", "NLL_fg"), ("ece", "ECE")]
METHOD_LABELS = {key: title for key, title, _ in METHOD_COLUMNS}
METHOD_COLORS = {
    "normal": "#1f77b4", "mc_dropout": "#ff7f0e", "tta": "#2ca02c",
    "noisy": "#d62728", "fusion": "#9467bd", "crf": "#8c564b",
}
PICKS = {"fácil": 85, "típico": 50, "difícil": 12}


def _load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_mask(path, target_shape):
    """Load a saved mask/probability map as a binary map on `target_shape`."""
    mask = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    if mask.shape != target_shape:
        img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (target_shape[1], target_shape[0]), Image.NEAREST
        )
        mask = np.asarray(img, dtype=np.float32) / 255.0
    return mask


def pick_samples(detailed_csv, sample_prefix=None, method="fusion", metric="dice", picks=None, hard_min=0.4):
    """Pick example samples deterministically from Dice percentiles."""
    picks = picks or PICKS
    df = pd.read_csv(detailed_csv)
    if sample_prefix:
        df = df[df["sample"].str.startswith(sample_prefix)]
    sub = df[(df.method == method) & (df.metric == metric)][["sample", "value"]].set_index("sample")["value"]
    out, used = {}, set()
    for kind, pct in picks.items():
        target = float(np.percentile(sub.values, pct))
        candidates = (sub - target).abs().sort_values().index
        if kind == "difícil":
            picked = next(s for s in candidates if s not in used and sub[s] >= hard_min)
        else:
            picked = next(s for s in candidates if s not in used)
        out[kind] = picked
        used.add(picked)
    return out


def plot_qualitative_panel(results_dir, detailed_csv, samples, title, out_path):
    dice = pd.read_csv(detailed_csv)
    dice = dice[dice.metric == "dice"]

    n_cols = 2 + len(METHOD_COLUMNS) + 1  # image + gt + methods + uncertainty
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, (kind, sample) in enumerate(samples.items()):
        sample_dir = os.path.join(results_dir, sample)
        gray = _load_gray(os.path.join(sample_dir, "original_image.png"))
        gt = _load_mask(os.path.join(sample_dir, "ground_truth.png"), gray.shape)

        for c, (col_key, col_title, rel_path) in enumerate(
            [("image", "Imagen", None), ("gt", "Ground truth", None)] + METHOD_COLUMNS
        ):
            ax = axes[r, c]
            if col_key == "image":
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            elif col_key == "gt":
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
                ax.contour(gt, levels=[0.5], colors="yellow", linewidths=1.2)
            else:
                mask = _load_mask(os.path.join(sample_dir, rel_path), gray.shape)
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
                rgba = np.zeros((*mask.shape, 4))
                rgba[mask > 0.5] = [0.95, 0.15, 0.15, 0.42]
                ax.imshow(rgba)
                ax.contour(gt, levels=[0.5], colors="yellow", linewidths=1.0)
                val = dice[(dice["sample"] == sample) & (dice.method == col_key)].value
                if len(val):
                    ax.text(4, 12, f"{val.iloc[0]:.2f}", color="white", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_title, fontsize=10)
            if c == 0:
                fd = dice[(dice["sample"] == sample) & (dice.method == "fusion")].value
                fd_txt = f", Dice fusión = {fd.iloc[0]:.2f}" if len(fd) else ""
                ax.set_ylabel(f"Caso {kind} — {sample}\n({fd_txt.lstrip(', ')})", fontsize=9)

        # last column: fusion uncertainty
        ax = axes[r, n_cols - 1]
        unc = np.asarray(Image.open(os.path.join(sample_dir, "fusion/uncertainty.png")), dtype=np.float32) / 255.0
        if unc.shape != gray.shape:
            unc = np.asarray(
                Image.fromarray((unc * 255).astype(np.uint8)).resize((gray.shape[1], gray.shape[0]), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        im = ax.imshow(unc, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("Incertidumbre (fusión)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(0.5, 0.945,
             "Rojo = máscara predicha · amarillo = contorno ground truth · número = Dice del método",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({samples})")


def plot_metric_distributions(detailed_csv, sample_prefix, title, out_path):
    """Box + jittered points per method for the key metrics."""
    df = pd.read_csv(detailed_csv)
    if sample_prefix:
        df = df[df["sample"].str.startswith(sample_prefix)]
    methods = [key for key, _, _ in METHOD_COLUMNS]
    fig, axes = plt.subplots(1, len(DIST_METRICS), figsize=(4.2 * len(DIST_METRICS), 4.0))
    rng = np.random.default_rng(0)
    for ax, (metric, metric_label) in zip(axes, DIST_METRICS):
        data = [df[(df.method == m) & (df.metric == metric)].value.dropna().to_numpy() for m in methods]
        box = ax.boxplot(data, patch_artist=True, tick_labels=list(METHOD_LABELS.values()), showfliers=False)
        for patch, m in zip(box["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.6)
        for i, vals in enumerate(data):
            x = i + 1 + rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(x, vals, s=3.5, color="black", alpha=0.22, zorder=3)
        pooled = np.concatenate(data)
        if metric == "nll_fg":
            # complete misses (dice ~ 0) push NLL_fg up to ~18 and squash the
            # boxes; cap the axis at the 99th percentile for readability
            cap = float(np.percentile(pooled, 99)) * 1.05
            n_out = int((pooled > cap).sum())
            ax.set_ylim(0, cap)
            ax.set_title(f"{metric_label} (recortado al P99; {n_out} puntos fuera)")
        elif metric in ("dice", "iou"):
            ax.set_ylim(0, 1.02)
            ax.set_title(metric_label)
        else:
            ax.set_ylim(0, float(pooled.max()) * 1.10)
            ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_statistical_summary(panels, out_path):
    """Forest plot of paired ΔDice with patient-level bootstrap CI."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, csv_path) in zip(axes, panels):
        df = pd.read_csv(csv_path)
        sub = df[df.metric == "dice"].set_index("comparison")
        ys = np.arange(len(COMPARISON_ORDER))[::-1]
        for y, comp in zip(ys, COMPARISON_ORDER):
            row = sub.loc[comp]
            sig = row["p_wilcoxon"] < 0.05
            color = ("#1a7a1a" if row.delta > 0 else "#b30000") if sig else "#8a8a8a"
            ax.errorbar(
                row.delta, y,
                xerr=[[row.delta - row.ci_low], [row.ci_high - row.delta]],
                fmt="o", color=color, ecolor=color, capsize=4, markersize=7,
            )
            p_txt = "p < 0.001" if row.p_wilcoxon < 0.001 else f"p = {row.p_wilcoxon:.3f}"
            ax.text(0.995, y, p_txt, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9, color=color)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(COMPARISON_ORDER, fontsize=10)
        lo, hi = ax.get_xlim()
        span = hi - lo if hi > lo else 1.0
        ax.set_xlim(lo - 0.08 * span, hi + 0.42 * span)  # room for p-value labels
        n_pat = int(sub.loc[COMPARISON_ORDER[0], "n_patients"])
        n_sli = int(sub.loc[COMPARISON_ORDER[0], "n_slices"])
        ax.set_title(f"{label} ({n_pat} pacientes, {n_sli} slices)", fontsize=11)
        ax.set_xlabel("Δ Dice (A − B), IC 95% bootstrap por paciente", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Tests pareados a nivel de paciente (Wilcoxon) — significativo en color, no significativo en gris",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_pipeline_figures(results_dir, sample_prefix, tag, model_title, scope_label, out_dir):
    """Qualitative panel + metric distributions for one experiment."""
    detailed = os.path.join(results_dir, "visualizations", "detailed_metrics.csv")
    samples = pick_samples(detailed, sample_prefix=sample_prefix)
    plot_qualitative_panel(
        results_dir, detailed, samples,
        f"{model_title} — máscaras por método ({scope_label})",
        os.path.join(out_dir, f"qualitative_{tag}.png"),
    )
    plot_metric_distributions(
        detailed, sample_prefix,
        f"{model_title} — distribución de métricas por método ({scope_label})",
        os.path.join(out_dir, f"distribution_{tag}.png"),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate README figures from saved results")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--foundation-results-dir", default="./results_foundation_universeg")
    parser.add_argument("--results-dir-4pct", default="./results_4pct")
    parser.add_argument("--foundation-results-dir-4pct", default="./results_foundation_universeg_4pct")
    parser.add_argument("--out-dir", default="./docs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    make_pipeline_figures(args.results_dir, None, "unet", "UNet 2D", "test, pacientes no vistos", args.out_dir)
    make_pipeline_figures(args.foundation_results_dir, "test_", "universeg",
                          "UniVerSeg (sin entrenamiento)", "test, pacientes no vistos", args.out_dir)
    make_pipeline_figures(args.results_dir_4pct, None, "4pct_unet", "UNet 2D",
                          "protocolo 4% — 372 slices", args.out_dir)
    make_pipeline_figures(args.foundation_results_dir_4pct, "test_", "4pct_universeg",
                          "UniVerSeg (sin entrenamiento)", "protocolo 4% — val+test, 87 slices", args.out_dir)

    unet_stats = os.path.join(args.results_dir, "visualizations", "statistical_tests.csv")
    found_stats = os.path.join(args.foundation_results_dir, "visualizations", "statistical_tests.csv")
    unet_stats_4pct = os.path.join(args.results_dir_4pct, "visualizations", "statistical_tests.csv")
    found_stats_4pct = os.path.join(args.foundation_results_dir_4pct, "visualizations", "statistical_tests.csv")

    plot_statistical_summary(
        [("UNet 2D (1%)", unet_stats), ("UniVerSeg (1%, test)", found_stats)],
        os.path.join(args.out_dir, "statistical_summary.png"),
    )
    plot_statistical_summary(
        [("UNet 2D (4%)", unet_stats_4pct), ("UniVerSeg (4%, val+test)", found_stats_4pct)],
        os.path.join(args.out_dir, "statistical_summary_4pct.png"),
    )


if __name__ == "__main__":
    main()
