"""A collection of functions to apply preprocessing to jet tuples.

The jet tuple is a tuple of the following format:

csts_cont: np.ndarray
    The constituents of the jet in the format [pt, (del)eta, (del)phi]
csts_cat: np.ndarray
    The categorical constituents of the jet, like the charge or pdg id
high: np.ndarray
    The high level features of the jet, pt, eta, phi, mass
labels:
    The type of jet
mask: np.ndarray
    A boolean mask for the constituents
"""

from copy import deepcopy

import numpy as np

from mltools.mltools.numpy_utils import log_squash


def apply_augmentation(jet_dict: dict, augmentation: callable) -> tuple:
    """Returns a new augmented copy of the jet dict."""
    assert all(k in jet_dict for k in ["csts", "mask"])
    new_csts, new_mask = augmentation(jet_dict["csts"], jet_dict["mask"])
    jet_dict["csts"] = new_csts
    jet_dict["mask"] = new_mask
    return jet_dict


def get_augmented_twins(jet_dict: dict, augmentation: callable) -> tuple:
    """Creates an augmented copy of the jet dict returns both."""
    return jet_dict, apply_augmentation(jet_dict, augmentation)


def apply_masking(jet_dict: tuple, masking_fn: callable) -> tuple:
    """Applies a masking function to the jet tuple."""
    assert all(k in jet_dict for k in ["csts", "mask"])
    jet_dict["null_mask"] = masking_fn(jet_dict["csts"], jet_dict["mask"])
    return jet_dict


def jitter_neutral_impact(jet_dict: tuple) -> np.ndarray:
    """Add noise to the impact parameters of neutral particles."""
    assert all(k in jet_dict for k in ["csts", "csts_id", "mask"])
    assert jet_dict["csts"].shape[-1] >= 4, "Expected at least 4 features in the csts"

    # Work out which are the neutral particles
    neutral_mask = np.copy(jet_dict["mask"])
    neutral_mask &= (jet_dict["csts_id"] == 0) | (jet_dict["csts_id"] == 2)
    n_neut = np.sum(neutral_mask)

    # Add noise to the impact parameters (last 4 parameters of the csts)
    rng = np.random.default_rng()
    csts = jet_dict["csts"]
    csts[neutral_mask, -4:] = rng.normal(0, 0.3, (n_neut, 4))

    # Make sure error terms are still positive
    csts[neutral_mask, -1] = np.abs(csts[neutral_mask, -1])
    csts[neutral_mask, -3] = np.abs(csts[neutral_mask, -3])

    jet_dict["csts"] = csts
    return jet_dict


def log_squash_csts_pt(jet_dict: tuple) -> np.ndarray:
    """Squashes the pt of the constituents using a log function."""
    assert "csts" in jet_dict
    jet_dict["csts"][..., 0] = log_squash(jet_dict["csts"][..., 0])
    return jet_dict


def drop_d0(jet_dict: tuple, max_val: float = 10) -> np.ndarray:
    """Drop if the d0 is greater than a certain value."""
    assert all(k in jet_dict for k in ["csts", "mask"])
    assert jet_dict["csts"].shape[-1] >= 4, "Expected at least 4 features in the csts"
    mask = np.abs(jet_dict["csts"][..., 3]) < max_val
    jet_dict["mask"] *= mask
    jet_dict["csts"] *= mask[..., None]
    return jet_dict


def tanh_d0_dz(jet_dict: tuple) -> np.ndarray:
    """Squashes the d0 and dz of the constituents using a tanh function."""
    assert "csts" in jet_dict
    assert jet_dict["csts"].shape[-1] >= 6, "Expected at least 6 features in the csts"
    jet_dict["csts"][..., 3] = np.tanh(jet_dict["csts"][..., 3])
    jet_dict["csts"][..., 5] = np.tanh(jet_dict["csts"][..., 5])
    return jet_dict


def compose(jet_dict, transforms: list) -> tuple:
    """Composes a series of preprocessing functions into a single function."""
    # Cycle through all the functions to compose
    for fn in transforms:
        # Stupid mutable dictionary...
        jet_dict = deepcopy(jet_dict)

        # Check if the function is callable
        if not callable(fn):
            raise TypeError(f"Expected a callable function, got {fn}")

        # If the input is actually a tuple, then apply the function to each element
        if isinstance(jet_dict, tuple):
            jet_dict = tuple(map(fn, jet_dict))

        # Otherwise, just apply the function
        else:
            jet_dict = fn(jet_dict)

    return jet_dict
