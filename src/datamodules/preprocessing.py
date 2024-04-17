from copy import deepcopy

import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch.nn.functional import pad
from torch.utils.data.dataloader import default_collate


def batch_preprocess(batch: list, fn: BaseEstimator) -> dict:
    """Preprocess the entire batch of jets.

    Preprocesing over the entire batch is much quicker than doing it per jet
    """
    # Colate the jets into a single dictionary
    jet_dict = default_collate(batch)

    # Load the constituents
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    # Check if the number of features is the same, else add nans
    if (feat_diff := fn.n_features_in_ - csts.shape[-1]) > 0:
        csts = pad(csts, (0, feat_diff))

    # Replace the impact parameters with the new values
    csts[mask] = T.from_numpy(fn.transform(csts[mask])).float()

    # Trim the padded features
    if feat_diff > 0:
        csts = csts[..., :-feat_diff]

    # Replace the neutral impact parameters with gaussian noise
    # They are zero padded anyway so contain no information!!!
    if jet_dict["csts"].shape[-1] > 3:
        neutral_mask = mask & (jet_dict["csts_id"] == 0) | (jet_dict["csts_id"] == 2)
        csts[..., -4:][neutral_mask] = T.randn_like(csts[..., -4:][neutral_mask])

    # Replace with the new constituents
    jet_dict["csts"] = csts
    return jet_dict


def jet_preprocess(jet_dict: list, fn: BaseEstimator) -> dict:
    """Preprocess a single jet."""
    # Deep copy the jet dictionary
    jet_dict = deepcopy(jet_dict)

    # Load the constituents
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    # Check if the number of features is the same, else add nans
    if (feat_diff := fn.n_features_in_ - csts.shape[-1]) > 0:
        csts = np.pad(csts, ((0, 0), (0, feat_diff)))

    # Replace the impact parameters with the new values
    csts[mask] = fn.transform(csts[mask]).astype(csts.dtype)

    # Trim the padded features
    if feat_diff > 0:
        csts = csts[..., :-feat_diff]

    # Replace the neutral impact parameters with gaussian noise
    # They are zero padded anyway so contain no information!!!
    if jet_dict["csts"].shape[-1] > 3:
        neutral_mask = mask & (jet_dict["csts_id"] == 0) | (jet_dict["csts_id"] == 2)
        rng = np.random.default_rng()
        csts[..., -4:][neutral_mask] = rng.standard_normal(
            csts[..., -4:][neutral_mask].shape, dtype=csts.dtype
        )

    # Replace with the new constituents
    jet_dict["csts"] = csts
    return jet_dict