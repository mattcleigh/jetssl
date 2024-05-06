from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from mltools.mltools.plotting import plot_multi_hists
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.masking import random_masking
from src.datamodules.preprocessing import batch_preprocess
from src.datamodules.transforms import apply_masking

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [32]),
    ("csts_id", "f", [32]),
    ("mask", "bool", [32]),
    ("labels", "l"),
]

# Define the preprocessing pipeline
pipeline = partial(apply_masking, masking_fn=partial(random_masking, mask_fraction=0.5))

# Create the datasets
sh_data = JetMappable(
    path="/srv/fast/share/rodem/shlomi",
    features=features,
    n_classes=3,
    processes="training",
    n_files=1,
    transforms=pipeline,
)
print(len(sh_data))
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
cst_features = ["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"]
cst_labels = [
    r"$p_\text{T}$",
    r"$\Delta\eta$",
    r"$\Delta\phi$",
    r"$d0$",
    r"$\sigma(d0)$",
    r"$z0$",
    r"$\sigma(z0)$",
]


# Plot distributions of the constituents
jc_csts = jc_data.data_dict["csts"][jc_data.data_dict["mask"]]
sh_csts = sh_data.data_dict["csts"][sh_data.data_dict["mask"]]

# Make the neutral jc impact parameters nans
is_neut = jc_data.data_dict["csts_id"][jc_data.data_dict["mask"]]
is_neut = (is_neut == 0) | (is_neut == 2)
jc_csts[is_neut, 3:] = np.nan

for i in range(len(cst_features)):
    plot_multi_hists(
        data_list=[jc_csts[:, i : i + 1], sh_csts[:, i : i + 1]],
        fig_height=4,
        data_labels=["JetClass", "SVFD"],
        bins=25,
        logy=True,
        ignore_nans=True,
        incl_overflow=False,
        incl_underflow=False,
        col_labels=[cst_labels[i]],
        legend_kwargs={"loc": "upper right"},
        path=root / f"plots/data/{cst_features[i]}.pdf",
        do_norm=True,
    )


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
plot_multi_hists(
    data_list=list(sh_csts.values()),
    data_labels=list(sh_csts.keys()),
    bins=51,
    logy=True,
    col_labels=cst_features,
    path=root / "plots/data/shlomi.png",
    do_norm=True,
)

plot_multi_hists(
    data_list=list(jc_csts.values()),
    data_labels=list(jc_csts.keys()),
    bins=51,
    logy=True,
    col_labels=cst_features,
    path=root / "plots/data/jetclass.png",
    do_norm=True,
)

# Plot the distribution of the constituent id values
id_types = [
    "$\\gamma$\n0",
    "hadron\n-1",
    "hadron\n0",
    "hadron\n+1",
    "$e$\n-1",
    "$e$\n+1",
    "$\\mu$\n-1",
    "$\\mu$\n+1",
]
sh_csts_id = sh_data.data_dict["csts_id"][sh_data.data_dict["mask"]][..., None]
jc_csts_id = jc_data.data_dict["csts_id"][jc_data.data_dict["mask"]][..., None]
sh_counts = np.unique(sh_csts_id, return_counts=True)[1].astype("f")
jc_counts = np.unique(jc_csts_id, return_counts=True)[1].astype("f")
sh_counts /= sh_counts.sum()
jc_counts /= jc_counts.sum()
sh_counts = np.insert(sh_counts, [0, 1], [0, 0])
counts = {"JetClass": jc_counts, "SVFD": sh_counts}
x = np.arange(8)
width = 0.3
multiplier = -0.5
fig, ax = plt.subplots(figsize=(6, 4))
for attribute, measurement in counts.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
ax.set_ylabel("a.u")
ax.set_xticks(x, id_types)
ax.legend(loc="upper right")
ax.set_yscale("log")
fig.tight_layout()
plt.savefig(root / "plots/data/csts_id.pdf")
plt.close()


# Define dataloaders which will apply the preprocessing pipeline
sh_data.transforms = pipeline
jc_data.transforms = pipeline
collate_fn = partial(
    batch_preprocess,
    fn=joblib.load(root / "resources/preprocessor_all.joblib"),
)
sh_loader = DataLoader(
    sh_data,
    batch_size=10_000,
    num_workers=0,
    shuffle=True,
    collate_fn=collate_fn,
)

jc_loader = DataLoader(
    jc_data,
    batch_size=10_000,
    num_workers=0,
    shuffle=True,
    collate_fn=collate_fn,
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
    path=root / "plots/data/batch.png",
    do_norm=True,
)
