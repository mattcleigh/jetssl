import rootutils
import torch as T

root = rootutils.setup_root(search_from=".", pythonpath=True)

from src.datamodules.hdf import JetMappable

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [32]),
    ("csts_id", "f", [32]),
    ("mask", "bool", [32]),
    ("vtx_id", "l"),
    ("labels", "l"),
    ("track_type", "l"),
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

csts = T.from_numpy(sh_data.data_dict["csts"])
mask = T.from_numpy(sh_data.data_dict["mask"])
labels = T.from_numpy(sh_data.data_dict["labels"])
vtx_id = T.from_numpy(sh_data.data_dict["vtx_id"])
track_type = T.from_numpy(sh_data.data_dict["track_type"])

vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
vtx_mask = T.triu(vtx_mask, diagonal=1)

# Calculate the number of same-vertex edges
targets = vtx_id.unsqueeze(-1) == vtx_id.unsqueeze(-2)
targets = targets[vtx_mask]
pos_weight = (targets == 0).sum() / targets.sum()
print(pos_weight)

# Check the sum of all weights
print(targets.sum() * pos_weight)
print((targets == 0).sum())

# Calculate the different track types
track_types = track_type[mask]
counts = T.unique(track_types, return_counts=True)[1]
weights = len(track_types) / (4 * counts.float())

# Check the sum of all weights
weight_per_track = weights[track_types]
print(counts)
print(weights)
print(weight_per_track.sum())
print(len(track_types))

# Check the types of tracks coming from each label
for lab in [0, 1, 2]:
    mask_l = labels == lab
    track_types_l = track_type[mask_l][mask[mask_l]]
    for t in [0, 1, 2, 3]:
        print(f"Label {lab}, Track type {t}: {T.sum(track_types_l == t)}")
