import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

from src.datamodules.hdf import JetMappable

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [32]),
    ("csts_id", "f", [32]),
    ("mask", "bool", [32]),
    ("labels", "l"),
    ("vtx_id", "l"),
]


# Create the datasets
sh_data = JetMappable(
    path="/srv/fast/share/rodem/shlomi",
    features=features,
    n_classes=3,
    processes="training",
    n_files=1,
)
sh_labels = ["light", "charm", "bottom"]

import torch as T

csts = T.from_numpy(sh_data.data_dict["csts"])
mask = T.from_numpy(sh_data.data_dict["mask"])
labels = T.from_numpy(sh_data.data_dict["labels"])
vtx_id = T.from_numpy(sh_data.data_dict["vtx_id"])

vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
eye = T.eye(vtx_id.shape[1], dtype=T.bool).unsqueeze(0)
vtx_mask = vtx_mask & ~eye  # No self-edges

# Calculate the number of same-vertex edges
vtx_same = vtx_id.unsqueeze(-1) == vtx_id.unsqueeze(-2)
targets = vtx_same[vtx_mask]

pos_weight = (targets == 0).sum() / targets.sum()
