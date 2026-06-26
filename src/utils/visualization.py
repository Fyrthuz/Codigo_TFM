import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_image(array, path):
    import PIL.Image
    if hasattr(array, "cpu"):
        array = array.cpu().numpy()
    # Reduce multi-dimensional arrays to 2D (take middle slice)
    if isinstance(array, np.ndarray):
        if array.ndim == 5:
            array = array[0]  # remove batch
        if array.ndim == 4:
            mid = array.shape[1] // 2
            array = array[:, mid]  # middle depth slice, channels stay
        if array.ndim == 3:
            # 3D volume or 2D+channel
            if array.shape[0] in (1, 3, 4):  # (C, H, W)
                array = np.transpose(array, (1, 2, 0))
            else:  # (D, H, W)
                mid = array.shape[0] // 2
                array = array[mid]
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
    if isinstance(array, np.ndarray) and array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif isinstance(array, np.ndarray) and array.dtype in (np.float32, np.float64):
        array = np.clip(array, 0, 1)
        array = (array * 255).astype(np.uint8)
    elif isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.integer):
        # Normalize integer masks to 0-255
        max_val = array.max()
        if max_val > 0 and max_val < 255:
            array = (array.astype(np.float32) / max_val * 255).astype(np.uint8)
        elif max_val > 255:
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = np.array([[array.item()]], dtype=np.uint8)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    img = PIL.Image.fromarray(array)
    img.save(path)


def plot_metrics_comparison(
    mean_results, methods, metrics, save_path,
    filename="metrics_summary.png",
):
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        plt.title(metric)
        values = [mean_results[method].get(metric, np.nan) for method in methods]
        plt.bar(methods, values)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()


def plot_enhanced_comparison(
    mean_results, methods, metrics, save_path,
    filename="enhanced_metrics_comparison.png",
):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    plt.figure(figsize=(18, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        values = [mean_results[method].get(metric, np.nan) for method in methods]
        clean_vals = [v for v in values if not (np.isnan(v) if isinstance(v, float) else False)]
        bars = plt.bar(methods, values, color=colors)
        for bar in bars:
            height = bar.get_height()
            if not (isinstance(height, float) and np.isnan(height)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=9, rotation=45,
                )
        if clean_vals:
            plt.ylim(0, 1.2 * max(clean_vals))
        plt.title(metric)
        plt.ylabel("Score")
        plt.xticks(rotation=45)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=methods[i])
        for i in range(len(methods))
    ]
    plt.figlegend(handles=handles, labels=methods, loc="upper right", bbox_to_anchor=(1.1, 0.9), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches="tight")
    plt.close()


def plot_box_comparison(
    overall_metrics, methods, metrics, save_path,
    filename="box_plot_comparison.png",
):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    plt.figure(figsize=(18, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i + 1)
        data = []
        for method in methods:
            values = [
                sample_data[method].get(metric, np.nan)
                for sample_data in overall_metrics.values()
                if method in sample_data
            ]
            data.append(values)
        box = plt.boxplot(data, patch_artist=True, tick_labels=methods)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        plt.title(metric)
        plt.ylabel("Score")
        plt.xticks(rotation=45)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i], label=methods[i])
        for i in range(len(methods))
    ]
    plt.figlegend(handles=handles, labels=methods, loc="upper right", bbox_to_anchor=(1.1, 0.9), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), bbox_inches="tight")
    plt.close()


def save_metrics_csv(mean_results, methods, metrics, save_path, filename="metrics_summary.csv"):
    df = pd.DataFrame.from_dict(mean_results, orient="index")
    df.to_csv(os.path.join(save_path, filename))
