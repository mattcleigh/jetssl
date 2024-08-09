from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from mltools.mltools.plotting import plot_multi_hists
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.preprocessing import batch_preprocess

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [64]),
    ("csts_id", "f", [64]),
    ("mask", "bool", [64]),
    ("labels", "l"),
    ("jets", "f"),
]
cst_features = [
    "pt",
    "deta",
    "dphi",
    "d0val",
    "d0err",
    "dzval",
    "dzerr",
]
cst_labels = [
    r"$p_\text{T} [GeV]$",
    r"$\Delta\eta$",
    r"$\Delta\phi$",
    r"$d0$",
    r"$\sigma(d0)$",
    r"$z0$",
    r"$\sigma(z0)$",
]
cst_bins = [
    np.linspace(0, 800, 25),
    np.linspace(-1, 1, 25),
    np.linspace(-1, 1, 25),
    np.linspace(-1000, 1000, 25),
    np.linspace(0, 0.8, 25),
    np.linspace(-1000, 1000, 25),
    np.linspace(0, 2, 25),
]


jet_features = ["pt", "eta", "phi", "mass", "ncst"]
jet_labels = [
    "$p_\text{T} [GeV]$",
    r"$\eta$",
    r"$\phi$",
    "Mass [GeV]",
    r"$N_\text{cst}$",
]

# Create the datasets
bt_data = JetMappable(
    path="/srv/fast/share/rodem/btag",
    features=features,
    n_classes=3,
    processes="train",
    n_files=1,
)
print(len(bt_data))
bt_labels = ["light", "charm", "bottom"]

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=features,
    processes="all",
    n_classes=10,
    n_files=1,
)
jc_labels = list(JC_CLASS_TO_LABEL.keys())

# Plot distributions of the constituents
bt_csts = bt_data.data_dict["csts"][bt_data.data_dict["mask"]]
jc_csts = jc_data.data_dict["csts"][jc_data.data_dict["mask"]]

# Make values of eta > 0.8 nans (ignored by plotting)
jc_csts[np.abs(jc_csts[:, 1]) > 0.795, 1] = np.nan

# Make the neutral jc impact parameters nans (ignored by plotting)
bt_csts_id = bt_data.data_dict["csts_id"][bt_data.data_dict["mask"]]
# is_neut = (jc_csts_id == 0) | (jc_csts_id == 2)
# jc_csts[is_neut, 3:] = np.nan

for i in range(len(cst_features)):
    plot_multi_hists(
        data_list=[jc_csts[:, i : i + 1], bt_csts[:, i : i + 1]],
        fig_height=4,
        data_labels=["JetClass", "BTag"],
        bins=cst_bins[i],
        logy=True,
        ignore_nans=True,
        incl_overflow=True,
        incl_underflow=True,
        col_labels=[cst_labels[i]],
        legend_kwargs={"loc": "upper right"},
        hist_kwargs=[{"fill": True, "alpha": 0.5}, {"fill": True, "alpha": 0.5}],
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


# sh_csts = csts_per_class(sh_data, sh_labels)
jc_csts = csts_per_class(jc_data, jc_labels)

# Plot distributions of the constituents for each class
# plot_multi_hists(
#     data_list=list(sh_csts.values()),
#     data_labels=list(sh_csts.keys()),
#     bins=51,
#     logy=True,
#     col_labels=cst_features,
#     path=root / "plots/data/shlomi.png",
#     do_norm=True,
# )

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
bt_csts_id = bt_data.data_dict["csts_id"][bt_data.data_dict["mask"]][..., None]
jc_csts_id = jc_data.data_dict["csts_id"][jc_data.data_dict["mask"]][..., None]
bt_counts = np.unique(bt_csts_id, return_counts=True)[1].astype("f")
jc_counts = np.unique(jc_csts_id, return_counts=True)[1].astype("f")
bt_counts /= bt_counts.sum()
jc_counts /= jc_counts.sum()
bt_counts = np.insert(bt_counts, [0, 1], [0, 0])
counts = {"JetClass": jc_counts, "BTag": bt_counts}
x = np.arange(8)
width = 0.3
multiplier = -0.5
fig, ax = plt.subplots(figsize=(6, 3))
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
# sh_data.transforms = pipeline
# jc_data.transforms = pipeline
collate_fn = partial(
    batch_preprocess,
    fn=joblib.load(root / "resources/cst_quant.joblib"),
)
jc_loader = DataLoader(
    jc_data,
    batch_size=10_000,
    num_workers=0,
    shuffle=True,
    collate_fn=collate_fn,
)

# Plot the first batch
# jc_dict = next(iter(jc_loader))
# sh_dict = next(iter(sh_loader))
# jc_csts = jc_dict["csts"][jc_dict["mask"]]
# sh_csts = sh_dict["csts"][sh_dict["mask"]]
# plot_multi_hists(
#     data_list=[jc_csts, sh_csts],
#     bins=51,
#     logy=True,
#     data_labels=["JetClass", "Shlomi"],
#     col_labels=cst_features,
#     path=root / "plots/data/batch.png",
#     do_norm=True,
# )
