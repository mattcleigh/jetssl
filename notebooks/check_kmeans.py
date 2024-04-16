from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch as T
from torch.utils.data import DataLoader
from torchpq.clustering import KMeans

from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.plotting import plot_multi_hists
from mltools.mltools.torch_utils import to_np
from src.datamodules.collation import batch_preprocess_impact
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.transforms import (
    log_squash_csts_pt,
)

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [128]),
    ("csts_id", "f", [128]),
    ("mask", "bool", [128]),
    ("labels", "l"),
]

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=features,
    processes="all",
    n_classes=10,
    n_files=1,
    transforms=partial(log_squash_csts_pt),
)
jc_labels = list(JC_CLASS_TO_LABEL.keys())
cst_features = ["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"]

# Create the dataloader
jc_loader = DataLoader(
    jc_data,
    batch_size=1_000,
    num_workers=0,
    shuffle=True,
    collate_fn=partial(
        batch_preprocess_impact,
        fn=joblib.load(root / "src/datamodules/impact_processor.joblib"),
    ),
)

# Cycle through the first 40 batches to get the preprocessed data
csts = []
mask = []
for i, batch in enumerate(jc_loader):
    if i > 50:
        break
    csts.append(batch["csts"])
    mask.append(batch["mask"])
csts = T.vstack(csts).to("cuda")
mask = T.vstack(mask).to("cuda")

# Create and fit the normaliser
normaliser = IterativeNormLayer(csts.shape[-1]).to("cuda")
normaliser.fit(csts, mask)
normed = normaliser(csts, mask)

# Create and fit the kmeans
kmeans = KMeans(8192, max_iter=500, tol=1e-4, verbose=10)
inputs = normed[mask].T.contiguous()
labels = kmeans.fit(inputs)

# Get the reconstructed data
recon = T.zeros_like(csts)
recon[mask] = kmeans.centroids[:, labels].T
recon = normaliser.reverse(recon)

# Convert to numpy for plotting
csts_np = to_np(csts[mask])
recon_np = to_np(recon[mask])
labels_np = to_np(labels)

# Plot histograms of the original and reconstructed data
plot_multi_hists(
    data_list=[csts_np, recon_np],
    data_labels=["Original", "Reconstructed"],
    col_labels=cst_features,
    bins=31,
    path="kmeans_recon.png",
)


# We want to order the constituents by their labels
order = np.argsort(labels_np[:100_000])  # only plot 10k
csts_np = csts_np[order]
labels_np = labels_np[order]

# Create a scatter plot of the constituents in pt, eta, and eta, phi
colors = labels_np / labels_np.max()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(csts_np[:, 0], csts_np[:, 1], marker=".", c=colors, cmap="viridis")
axes[0].set_xlabel("pt")
axes[0].set_ylabel("eta")

axes[1].scatter(csts_np[:, 1], csts_np[:, 2], marker=".", c=colors, cmap="viridis")
axes[1].set_xlabel("eta")
axes[1].set_ylabel("phi")

plt.savefig("kmeans_labels.png")
plt.close("all")
