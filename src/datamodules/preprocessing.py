from copy import deepcopy

import numpy as np
import torch as T
from sklearn.base import BaseEstimator
from torch.nn.functional import pad
from torch.utils.data.dataloader import default_collate


def preprocess(
    jet_dict: dict[np.ndarray],
    fn: BaseEstimator,
    hlv_fn: BaseEstimator | None = None,
    get_ratios: bool = False,
) -> dict:
    """Preprocess a jet dict using a sklearn transformer.

    Should be used before collating, ideally as the dataclass transforms argument.

    Works on both single and batched jets.
    Preprocesing over the entire batch is much quicker than doing it per jet.
    """
    # Load the constituents
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]
    is_batched = mask.ndim == 2

    # Get the mask ratios BEFORE any preprocessing
    if get_ratios:
        # If no mask is present, then the ratios are simply zeros
        if "null_mask" not in jet_dict:
            if is_batched:
                ratio = np.zeros((csts.shape[0], 2), dtype=np.float32)
            else:
                ratio = np.zeros(2, dtype=np.float32)
        else:
            jets = jet_dict["jets"]
            null_mask = jet_dict["null_mask"]

            mask_ratio = null_mask.sum(-1) / mask.sum(-1)
            csts_pt = csts[..., 0]
            pt_masked = (csts_pt * null_mask).sum(-1)
            pt_total = (csts_pt * mask).sum(-1)
            pt_ratio = pt_masked / pt_total
            ratio = np.stack((mask_ratio, pt_ratio), dtype=np.float32, axis=-1)
        jet_dict["jets"] = ratio

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

    # If there is a hlvs function, apply it
    if hlv_fn is not None and not get_ratios:
        jets = jet_dict["jets"]
        if exp_jets := (jets.ndim == 1):  # Must work on batched and single jets
            jets = jets[None, ...]
        jets = hlv_fn.transform(jets).astype(jets.dtype)
        if exp_jets:
            jets = jets[0]
        jet_dict["jets"] = jets

    return jet_dict


def batch_preprocess(
    batch: list[T.Tensor],
    fn: BaseEstimator,
    hlv_fn: BaseEstimator | None = None,
    get_ratios: bool = False,
) -> dict:
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

    # If there is a hlv function, apply it
    if hlv_fn is not None:
        jets = jet_dict["jets"]
        jet_dict["jets"] = T.from_numpy(hlv_fn.transform(jets)).float()
    elif get_ratios:
        print("I havent implemented this yet!")
    return jet_dict


def batch_masking(
    jet_dict: dict[np.ndarray], fn: callable, key: str = "null_mask"
) -> dict:
    """Applies a masking function of a batch of jets."""
    # If the data is a list of dicts
    if isinstance(jet_dict, list):
        return [batch_masking(jet, fn, key) for jet in jet_dict]

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


def compose(jet_dict, transforms: list) -> tuple:
    """Composes a series of preprocessing functions into a single function."""
    # Cycle through all the functions to compose
    for fn in transforms:
        # Check if the function is callable
        if not callable(fn):
            raise TypeError(f"Expected a callable function, got {fn}")

        # Apply the function to a copy of the dict (mutable)
        jet_dict = fn(deepcopy(jet_dict))

    return jet_dict
