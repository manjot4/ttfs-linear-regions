from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from .configs import DEPTH_VALUES, WIDTH_VALUES


def plot_width_depth(
    width_summary: pd.DataFrame,
    depth_summary: pd.DataFrame,
    schedules,
    output: str | Path | None = None,
    title: str | None = None,
):
    schedule_order = list(schedules.keys())
    ws = width_summary.sort_values("width")
    hs = depth_summary
    mat = (
        hs.pivot(index="depth", columns="schedule", values="mean_regions")
        .reindex(index=DEPTH_VALUES, columns=schedule_order)
    )
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    ax[0].errorbar(ws.width, ws.mean_regions, yerr=ws.std_regions, marker="o", linewidth=1.8, capsize=3)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([w for w in WIDTH_VALUES if w in set(ws.width.astype(int))])
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].set_xlabel("Width")
    ax[0].set_ylabel("Number of regions")
    ax[0].set_title("Depth 1")
    ax[0].grid(alpha=0.22)

    data = mat.to_numpy(dtype=float)
    im = ax[1].imshow(data, origin="lower", aspect="auto", cmap="Spectral_r", vmin=0, vmax=float(np.nanmax(data)), interpolation="nearest")
    ax[1].set_xticks(np.arange(len(schedule_order)))
    ax[1].set_xticklabels(schedule_order, rotation=24, ha="right")
    ax[1].set_yticks(np.arange(len(DEPTH_VALUES)))
    ax[1].set_yticklabels(DEPTH_VALUES)
    ax[1].set_xlabel("Width schedule")
    ax[1].set_ylabel("Depth")
    ax[1].set_title("Depth / width schedule")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.035)
    cb.set_label("Number of regions")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig


def plot_training_regions(region_df: pd.DataFrame, output: str | Path | None = None):
    agg = (
        region_df.groupby(["dataset", "model_family", "width", "epoch"], as_index=False)
        .agg(mean_regions=("regions", "mean"), std_regions=("regions", "std"))
    )
    datasets = list(agg.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = agg[agg.dataset == ds]
        for (model, width), g in sub.groupby(["model_family", "width"]):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g.mean_regions, marker="o", label=f"{model}, w={width}")
        ax.set_title(ds.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of regions")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig


def plot_training_accuracy(accuracy_df: pd.DataFrame, output: str | Path | None = None):
    agg = (
        accuracy_df.groupby(["dataset", "model_family", "init_name", "width", "epoch"], as_index=False)
        .agg(mean_accuracy=("test_accuracy", "mean"), std_accuracy=("test_accuracy", "std"))
    )
    datasets = list(agg.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = agg[agg.dataset == ds]
        for (model, init_name, width), g in sub.groupby(["model_family", "init_name", "width"]):
            g = g.sort_values("epoch")
            ax.plot(g.epoch, g.mean_accuracy, marker="o", label=f"{model}/{init_name}, w={width}")
        ax.set_title(ds.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig
