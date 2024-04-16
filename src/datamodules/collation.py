import torch as T
from sklearn.base import BaseEstimator
from torch.utils.data.dataloader import default_collate


def minimize_padding(batch: list) -> dict:
    """Given a batch of jets, compress the relevant tensors to minimise the padding."""
    # Always run the default collate function first
    jet_dict = default_collate(batch)

    if "mask" not in jet_dict:
        return jet_dict

    # Calculate the minimum mask required per jet
    mask = jet_dict["mask"]
    max_csts = mask.sum(axis=-1).max()

    # Check if the mask is already minimal
    if max_csts == mask.shape[-1]:
        return jet_dict

    # Get the array that sorts the mask
    # Eps step preserves order which is important for pos encoding
    eps = T.linspace(0, 0.1, mask.shape[-1], device=mask.device)
    sort_mask = T.argsort(mask.float() - eps, descending=True, dim=-1)[:, :max_csts]

    # Use gather to replace each element of the jet dict
    jet_dict["mask"] = T.gather(mask, 1, sort_mask)

    if "csts" in jet_dict:
        csts = jet_dict["csts"]
        new_sort = sort_mask.unsqueeze(-1).expand(-1, -1, csts.shape[-1])
        jet_dict["csts"] = T.gather(csts, 1, new_sort)

    if "csts_id" in jet_dict:
        jet_dict["csts_id"] = T.gather(jet_dict["csts_id"], 1, sort_mask)

    if "vtx_id" in jet_dict:
        jet_dict["vtx_id"] = T.gather(jet_dict["vtx_id"], 1, sort_mask)

    if "null_mask" in jet_dict:
        jet_dict["null_mask"] = T.gather(jet_dict["null_mask"], 1, sort_mask)

    return jet_dict


def batch_preprocess_impact(batch: list, fn: BaseEstimator) -> dict:
    """Preprocess the badly bahaved impact parameters of the jets."""
    # Always run the default collate function first
    jet_dict = default_collate(batch)

    # Check that the required keys are present
    assert all(k in jet_dict for k in ["csts", "csts_id", "mask"])

    # If there are no impact parameters to reshape, skip
    if jet_dict["csts"].shape[-1] < 7:
        return jet_dict

    # Pass the constituent impact parameters through the preprocessor
    csts = jet_dict["csts"]
    mask = jet_dict["mask"]

    # Replace the impact parameters with the new values
    csts[..., -4:][mask] = T.from_numpy(fn.transform(csts[mask][:, -4:])).float()

    # Replace the neutral particles with gaussian noise
    # They are zero padded anyway so contain no information!!!
    neutral_mask = mask & (jet_dict["csts_id"] == 0) | (jet_dict["csts_id"] == 2)
    csts[..., -4:][neutral_mask] = T.randn_like(csts[..., -4:][neutral_mask])

    # Replace with the new constituents
    jet_dict["csts"] = csts
    return jet_dict
