from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch as T
from torch.utils.data import DataLoader
from torchpq.clustering import KMeans

from mltools.mltools.plotting import plot_multi_hists
from mltools.mltools.torch_utils import to_np
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.preprocessing import batch_preprocess

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ["csts", "f", [128]],
    ["csts_id", "f", [128]],
    ["mask", "bool", [128]],
    ["labels", "l"],
]

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=features,
    processes="all",
    n_classes=10,
    n_files=1,
)
jc_labels = list(JC_CLASS_TO_LABEL.keys())
cst_features = ["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"]

# Create the dataloader
preprocessor = joblib.load(root / "resources/preprocessor_all.joblib")
jc_loader = DataLoader(
    jc_data,
    batch_size=1_000,
    num_workers=0,
    shuffle=True,
    collate_fn=partial(batch_preprocess, fn=preprocessor),
)

# Cycle through the first 40 batches to get the preprocessed data
csts = []
csts_id = []
for i, batch in enumerate(jc_loader):
    if i > 10:
        break
    csts.append(batch["csts"][batch["mask"]])
    csts_id.append(batch["csts_id"][batch["mask"]])
csts = T.vstack(csts).to("cuda")
csts_id = T.hstack(csts_id).to("cuda")

# Create and fit the kmeans
kmeans = KMeans(8192, max_iter=500, verbose=10)
labels = kmeans.fit(csts.T.contiguous())

# Convert to numpy for plotting
csts_np = to_np(csts)
labels_np = to_np(labels)
csts_id_np = to_np(csts_id)

# Get the reconstructed data
recon = to_np(kmeans.centroids[:, labels].T)

# Invert the pre-processing
csts_np = preprocessor.inverse_transform(csts_np)
recon = preprocessor.inverse_transform(recon)
neut_mask = (csts_id_np == 0) | (csts_id_np == 2)
csts_np[:, -4:][neut_mask] = 0
recon[:, -4:][neut_mask] = 0

# Plot histograms of the original and reconstructed data
plot_multi_hists(
    data_list=[csts_np, recon],
    data_labels=["Original", "Reconstructed"],
    col_labels=cst_features,
    bins=31,
    path=root / "plots/kmeans_recon.png",
    logy=True,
)

# We want to order the constituents by their labels
order = np.argsort(labels_np[:100_000])  # only plot 100k

# Create a scatter plot of the constituents in pt, eta, and eta, phi
colors = labels_np / labels_np.max()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for a in range(len(axes)):
    axes[a].scatter(
        csts_np[order][:, a],
        csts_np[order][:, a + 1],
        marker="o",
        c=colors[order],
        cmap="tab20",
        alpha=0.1,
    )
axes[0].set_xscale("log")
axes[0].set_xlabel("log-pt")
axes[0].set_ylabel("eta")
axes[1].set_xlabel("eta")
axes[1].set_ylabel("phi")

plt.savefig(root / "plots/kmeans_labels.png")
plt.close("all")
