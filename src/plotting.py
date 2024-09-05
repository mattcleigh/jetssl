"""Collection of plotting functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import torch as T
import wandb
from matplotlib import gridspec
from matplotlib import pyplot as plt

from mltools.mltools.torch_utils import to_np

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8

_ORDER = [
    "nobackbone",
    "untrained",
    "reg",
    "kmeans",
    "vae",
    "flow",
    "diff",
    "mdm",
]

_COLORS = {
    "nobackbone": "C0",
    "untrained": "C0",
    "reg": "C1",
    "diff": "C5",
    "flow": "C2",
    "vae": "C4",
    "kmeans": "C3",
    "mdm": "C6",
}

_LABELS = {
    "nobackbone": "No Backbone",
    "untrained": "From Scratch",
    "reg": "Regression",
    "diff": "Diffusion",
    "flow": "Flow",
    "vae": "VQVAE",
    "kmeans": "K-Means",
    "mdm": "SDM",
}


def plot_labels(data: dict, pred: T.Tensor, n_samples: int = 5) -> None:
    # Unpack the data
    csts_id = data["csts_id"]
    mask = data["mask"]
    null_mask = data["null_mask"]

    # Create a copy of the csts_id tensor with the predicted values
    pred_csts_id = csts_id.clone()
    pred_csts_id[null_mask] = pred

    # Convert all the tensors to numpy
    csts_id = to_np(csts_id)
    mask = to_np(mask)
    null_mask = to_np(null_mask)
    pred_csts_id = to_np(pred_csts_id)

    # Cycle through the batch
    for b in range(min(csts_id.shape[0], n_samples)):
        # Select the current jet
        c = csts_id[b]
        m = mask[b]
        nm = null_mask[b]
        rc = pred_csts_id[b]

        # Split the features into the original, survived and sampled
        original = c[m]
        survived = c[m & ~nm]
        sampled = rc[m]

        # Create the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        bins = np.arange(CSTS_ID + 1)

        # Plot the histogram of the original jets
        o_hist, bins = np.histogram(original, bins=bins)
        ax.stairs(o_hist, bins, color="k", label="Original")

        # Plot the histogram of the survived jets
        s_hist, _ = np.histogram(survived, bins=bins)
        ax.stairs(s_hist, bins, fill=True, alpha=0.3, color="g", label="Survived")

        # Stack ontop of that a histogram of the sampled jets
        p_hist, _ = np.histogram(sampled, bins=bins)
        ax.stairs(
            p_hist,
            bins,
            baseline=s_hist,
            fill=True,
            alpha=0.3,
            color="b",
            label="Sampled",
            zorder=-1,
        )

        # Get the highest value to set the yscale
        max_val = max([o_hist.max(), s_hist.max(), p_hist.max()])
        ax.set_ylim(0, max_val * 1.6)
        ax.set_xlim(0, 8)

        ax.legend()
        ax.set_xlabel("Constituent Type")
        fig.tight_layout()
        fig.savefig(f"plots/jet_class_{b}")
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        if wandb.run is not None:
            wandb.log({f"jet_class_{b}": wandb.Image(img)}, commit=False)
        plt.close()


def plot_continuous(
    data: dict,
    pred: T.Tensor,
    n_samples: int = 5,
) -> None:
    """Plot the original, survived and sampled continuous features of the jets."""
    # Unpack the sample
    csts = data["csts"]
    mask = data["mask"]
    null_mask = data["null_mask"]

    # Create a copy of the csts_id tensor with the predicted values
    pred_csts = csts.clone()
    pred_csts[null_mask] = pred.type(pred_csts.dtype)

    # Convert all the tensors to numpy
    csts = to_np(csts)
    mask = to_np(mask)
    null_mask = to_np(null_mask)
    pred_csts = to_np(pred_csts)

    # Cycle through the batch
    for b in range(min(csts.shape[0], n_samples)):
        # Select the current jet
        c = csts[b]
        m = mask[b]
        nm = null_mask[b]
        rc = pred_csts[b]  # reconstructed

        # Split the features into the original, survived and sampled
        original = c[m]
        survived = c[m & ~nm]
        sampled = rc[m]

        # Create the figure and axes
        fig, axes = plt.subplots(1, csts.shape[-1], figsize=(2 * csts.shape[-1], 3))
        labels = [
            r"$p_T$",
            r"$\eta$",
            r"$\phi$",
            r"$d0$",
            r"$z0$",
            r"Err$(d0)$",
            r"Err$(z0)$",
        ]

        # Cycle through the features
        for i, ax in enumerate(axes):
            # Create the bins and clip to include overflow/underflow
            bins = np.linspace(-3, 3, 21)
            original[:, i] = np.clip(original[:, i], bins[0], bins[-1])
            survived[:, i] = np.clip(survived[:, i], bins[0], bins[-1])
            sampled[:, i] = np.clip(sampled[:, i], bins[0], bins[-1])

            # Plot the histogram of the original jets
            o_hist, _ = np.histogram(original[:, i], bins=bins)
            ax.stairs(o_hist, bins, color="k", label="Original")

            # Plot the histogram of the survived jets
            s_hist, _ = np.histogram(survived[:, i], bins=bins)
            ax.stairs(s_hist, bins, fill=True, alpha=0.3, color="g", label="Survived")

            # Stack ontop of that a histogram of the sampled jets
            p_hist, _ = np.histogram(sampled[:, i], bins=bins)
            ax.stairs(
                p_hist,
                bins,
                baseline=s_hist,
                fill=True,
                alpha=0.3,
                color="b",
                label="Sampled",
                zorder=-1,
            )
            ax.set_xlabel(labels[i])
            ax.set_xlim(-3, 3)

            # Get the highest value to set the yscale
            max_val = max([o_hist.max(), s_hist.max(), p_hist.max()])
            ax.set_ylim(0, max_val * 1.6)

        ax.legend()
        fig.tight_layout()
        fig.savefig(f"plots/jet_{b}")
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        if wandb.run is not None:
            wandb.log({f"jet_{b}": wandb.Image(img)}, commit=False)
        plt.close()


def plot_metric(
    df: pd.DataFrame,
    model_list: list,
    x_name: str,
    y_name: str,
    x_label: str,
    y_label: str,
    path: str,
    x_lim: list | None = None,
    y_lim: list | None = None,
    y2_lim: tuple | None = None,
    log_x: bool = False,
    log_y: bool = False,
    show_grid: bool = False,
    figsize: tuple = (5, 5),
    subplot="none",
    inset_bounds: list | None = None,
    inset_loc: list | None = None,
    legend_kwargs: dict | None = None,
):
    assert subplot in {"none", "ratio", "zoom"}
    fig = plt.figure(figsize=figsize)

    if subplot == "none":
        ax0 = fig.add_subplot()
        fig2 = plt.figure()
        ax1 = fig2.add_subplot()
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)

    if inset_bounds is not None:
        axins = ax0.inset_axes(inset_bounds, xlim=inset_loc[:2], ylim=inset_loc[2:])
        _, lines = ax0.indicate_inset_zoom(axins, edgecolor="black")
        lines[0].set_visible(True)
        lines[1].set_visible(False)
        lines[2].set_visible(False)
        lines[3].set_visible(True)
        if log_x:
            axins.set_xscale("log")
        if log_y:
            axins.set_yscale("log")
        axins.grid(True, which="major", ls="--", alpha=0.5)
        axins.set_xticklabels([])
        # axins.get_yaxis().set_visible(False)

    # Cycle through the models
    m_names = sorted(model_list, key=_ORDER.index)

    denom = None
    for m in m_names:
        # Get the data for this model
        data = df[df["model"] == m]
        data = data.drop(columns=["model"])

        # Combine the seeds into mean and std
        data = data.groupby([x_name]).agg(["mean", "std"]).reset_index()
        data = data.astype("f")

        # Plot values with error
        x = data[x_name]
        y = data[y_name]["mean"]
        y_dev = np.nan_to_num(data[y_name]["std"])  # Can be nan with one seed
        down = y - y_dev
        up = y + y_dev

        # Plot the data
        col = _COLORS[m]
        ax0.plot(x, y, "-o", color=col, label=_LABELS[m])
        ax0.fill_between(x, down, up, alpha=0.2, color=col)

        # Add the inset plot
        if inset_bounds is not None:
            axins.plot(x, y, "-o", color=col)
            axins.fill_between(x, down, up, alpha=0.2, color=col)

        # Plot subplots
        if subplot == "none":
            continue

        if subplot == "ratio":
            if denom is None:
                denom = y
        else:
            denom = 1
        ax1.plot(x, y / denom, "-o", color=col)
        ax1.fill_between(x, down / denom, up / denom, alpha=0.2, color=col)

    # Make sure the sig figs for the ax1 are the same as ax0
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Format the plot
    ax0.legend(frameon=False, **(legend_kwargs or {}))
    ax0.set_ylabel(y_label)
    if subplot == "none":
        ax0.set_xlabel(x_label)
    else:
        ax1.set_xlabel(x_label)
    if subplot == "ratio":
        ax1.set_ylabel(f"Ratio to\n{_LABELS[m_names[0]]}")
    if subplot == "zoom":
        ax1.set_ylabel(f"{y_label} (zoomed)")
    if log_x:
        ax0.set_xscale("log")
        ax1.set_xscale("log")
    if log_y:
        ax0.set_yscale("log")
    if x_lim is not None:
        ax0.set_xlim(x_lim)
        ax1.set_xlim(x_lim)
    if y_lim is not None:
        ax0.set_ylim(y_lim)
    if y2_lim is not None:
        ax1.set_ylim(y2_lim)
    if show_grid:
        ax0.grid(True, which="major", ls="--", alpha=0.5)
        ax1.grid(True, which="major", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, top=0.75)

    # Make the directory and save the plot
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close("all")
