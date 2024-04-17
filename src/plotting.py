"""Collection of plotting functions."""

import numpy as np
import PIL
import torch as T
import wandb
from matplotlib import pyplot as plt

from mltools.mltools.torch_utils import to_np

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


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
