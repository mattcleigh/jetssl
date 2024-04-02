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


def plot_labels(
    csts_id: T.Tensor,
    mask: T.Tensor,
    null_mask: T.Tensor,
    rec_csts_id: T.Tensor,
    n_samples: int = 5,
) -> None:
    # Convert all the tensors to numpy
    csts_id = to_np(csts_id)
    mask = to_np(mask)
    null_mask = to_np(null_mask)
    rec_csts_id = to_np(rec_csts_id)

    # Cycle through the batch
    for b in range(min(csts_id.shape[0], n_samples)):
        # Select the current jet
        c = csts_id[b]
        m = mask[b]
        nm = null_mask[b]
        rc = rec_csts_id[b]

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
        ax.legend()
        ax.set_xlabel("Constituent Type")
        fig.tight_layout()
        fig.savefig(f"plots/jet_class_{b}")
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb.log({f"jet_class_{b}": wandb.Image(img)}, commit=False)
        plt.close()


def plot_continuous(
    csts: T.Tensor,
    mask: T.Tensor,
    null_mask: T.Tensor,
    rec_csts: T.Tensor,
    n_samples: int = 5,
) -> None:
    """Plot the original, survived and sampled continuous features of the jets."""
    # Convert all the tensors to numpy
    csts = to_np(csts)
    mask = to_np(mask)
    null_mask = to_np(null_mask)
    rec_csts = to_np(rec_csts)

    # Cycle through the batch
    for b in range(min(csts.shape[0], n_samples)):
        # Select the current jet
        c = csts[b]
        m = mask[b]
        nm = null_mask[b]
        rc = rec_csts[b]

        # Split the features into the original, survived and sampled
        original = c[m]
        survived = c[m & ~nm]
        sampled = rc[m]

        # Create the figure and axes
        fig, axes = plt.subplots(1, csts.shape[-1], figsize=(2 * csts.shape[-1], 3))

        # Cycle through the features
        for i, ax in enumerate(axes):
            # Plot the histogram of the original jets
            o_hist, bins = np.histogram(original[:, i], bins=11)
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
        ax.legend()
        axes[0].set_yscale("log")
        fig.tight_layout()
        fig.savefig(f"plots/jet_{b}")
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb.log({f"jet_{b}": wandb.Image(img)}, commit=False)
        plt.close()
