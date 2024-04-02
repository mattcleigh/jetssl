import torch as T
from torch.utils.data.dataloader import default_collate


def minimize_padding(jet_dict: dict) -> dict:
    """Given a batch of jets, compress the relevant tensors to minimise the padding."""
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


def collate_with_fn(batch: list, fn: callable) -> dict:
    """Run an extra function on the batch after collation."""
    return fn(default_collate(batch))
