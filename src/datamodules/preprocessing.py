import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch.nn.functional import pad
from torch.utils.data.dataloader import default_collate


def preprocess(jet_dict: dict, fn: BaseEstimator) -> dict:
    """Preprocess a jet dict using a sklearn transformer.

    Works on both single and batched jets.
    Preprocesing over the entire batch is much quicker than doing it per jet
    """
    # Load the constituents
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    # Check if the number of features is the same, else pad
    if (feat_diff := fn.n_features_in_ - csts.shape[-1]) > 0:
        zeros = np.zeros((csts.shape[:-1] + (feat_diff,)), dtype=csts.dtype)
        csts = np.concatenate((csts, zeros), axis=-1)

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


def batch_masking(jet_dict: dict, fn: callable, key: str = "null_mask") -> dict:
    """Applies a masking function of a batch of jets."""
    assert all(k in jet_dict for k in ["csts", "mask"])

    # If the data is batched
    if jet_dict["mask"].ndim == 2:
        msk = []
        for csts, mask in zip(jet_dict["csts"], jet_dict["mask"], strict=False):
            msk.append(fn(csts, mask))
        msk = np.array(msk)

    # If the data is a single jet
    else:
        msk = fn(jet_dict["csts"], jet_dict["mask"])

    jet_dict[key] = msk
    return jet_dict


def batch_preprocess(batch: list, fn: BaseEstimator) -> dict:
    """Preprocess the entire batch of jets.

    This runs on pytorch tensors and should slot in as a collate function
    for the DataLoader.
    """
    # Colate the jets into a single dictionary
    jet_dict = default_collate(batch)

    # Load the constituents
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    # Check if the number of features is the same, else pad
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
