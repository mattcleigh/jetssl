from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import numpy as np
from torch.utils.data import DataLoader

from mltools.mltools.plotting import plot_multi_hists
from src.datamodules.collation import collate_with_fn, minimize_padding
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.masking import random_masking
from src.datamodules.transforms import (
    apply_masking,
    compose,
    jitter_neutral_impact,
    log_squash_csts_pt,
    tanh_d0_dz,
)

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [32]),
    ("csts_id", "f", [32]),
    ("mask", "bool", [32]),
    ("labels", "l"),
]

# Define the preprocessing pipeline
pipeline = partial(
    compose,
    transforms=[
        jitter_neutral_impact,
        log_squash_csts_pt,
        tanh_d0_dz,
        partial(apply_masking, masking_fn=partial(random_masking, mask_fraction=0.5)),
    ],
)

# Create the datasets
sh_data = JetMappable(
    path="/srv/fast/share/rodem/shlomi",
    features=features,
    n_classes=3,
    processes="training",
    n_files=1,
    transforms=pipeline,
)
sh_labels = ["light", "charm", "bottom"]

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=features,
    processes="all",
    n_classes=10,
    n_files=1,
    transforms=pipeline,
)
jc_labels = list(JC_CLASS_TO_LABEL.keys())


# Split the datasets into seperate lists based on the labels
def csts_per_class(dataset, labels) -> dict:
    csts = dataset.data_dict["csts"]
    label_idx = dataset.data_dict["labels"]
    mask = dataset.data_dict["mask"]
    csts = [csts[label_idx == i] for i in np.unique(label_idx)]
    mask = [mask[label_idx == i] for i in np.unique(label_idx)]
    csts = [c[m] for c, m in zip(csts, mask, strict=False)]
    return dict(zip(labels, csts, strict=False))


sh_csts = csts_per_class(sh_data, sh_labels)
jc_csts = csts_per_class(jc_data, jc_labels)

# Plot distributions of the constituents for each class
cst_features = ["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"]
plot_multi_hists(
    data_list=list(sh_csts.values()),
    data_labels=list(sh_csts.keys()),
    bins=51,
    logy=True,
    col_labels=cst_features,
    path=root / "plots/shlomi.png",
    do_norm=True,
)

plot_multi_hists(
    data_list=list(jc_csts.values()),
    data_labels=list(jc_csts.keys()),
    bins=51,
    logy=True,
    col_labels=cst_features,
    path=root / "plots/jetclass.png",
    do_norm=True,
)

# Plot the distribution of the constituent id values
sh_csts_id = sh_data.data_dict["csts_id"][sh_data.data_dict["mask"]][..., None]
jc_csts_id = jc_data.data_dict["csts_id"][jc_data.data_dict["mask"]][..., None]
plot_multi_hists(
    data_list=[jc_csts_id, sh_csts_id],
    logy=True,
    data_labels=["JetClass", "Shlomi"],
    col_labels=["particle id"],
    path=root / "plots/csts_id.png",
    do_norm=True,
)

# Define dataloaders which will apply the preprocessing pipeline
sh_data.transforms = pipeline
jc_data.transforms = pipeline
sh_loader = DataLoader(
    sh_data,
    batch_size=10_000,
    num_workers=0,
    shuffle=True,
    collate_fn=partial(collate_with_fn, fn=minimize_padding),
)

jc_loader = DataLoader(
    jc_data,
    batch_size=10_000,
    num_workers=0,
    shuffle=True,
    collate_fn=partial(collate_with_fn, fn=minimize_padding),
)

# Plot the first batch
jc_dict = next(iter(jc_loader))
sh_dict = next(iter(sh_loader))
jc_csts = jc_dict["csts"][jc_dict["mask"]]
sh_csts = sh_dict["csts"][sh_dict["mask"]]
plot_multi_hists(
    data_list=[jc_csts, sh_csts],
    bins=51,
    logy=True,
    data_labels=["JetClass", "Shlomi"],
    col_labels=cst_features,
    path=root / "plots/batch.png",
    do_norm=True,
)

# Get the id of the constituents with d0 = 0
jc_csts_id = jc_dict["csts_id"][jc_dict["mask"]]
d0_is_zero = jc_csts[..., -4] == 0
print(jc_csts_id[d0_is_zero])
