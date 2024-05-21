from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import matplotlib.pyplot as plt
import torch as T
from torch.utils.data import DataLoader
from torchpq.clustering import KMeans

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

# Load the VQ-VAE model
model_path = "/srv/beegfs/scratch/groups/rodem/jetssl/jetssl2/vqvae/quantizer.pkl"
quantizer = T.load(model_path, map_location="cuda")

# Cycle through the first 40 batches to get the preprocessed data
all_csts = []
all_csts_id = []
all_vae_idx = []
all_vae_z = []
for i, batch in enumerate(jc_loader):
    if i > 50:
        break
    csts = batch["csts"]
    mask = batch["mask"]
    csts_id = batch["csts_id"]
    with T.autocast(device_type="cuda"):
        vae_idx, vae_z, _, _ = quantizer(csts.to("cuda"), mask.to("cuda"))
    all_csts.append(csts[mask])
    all_csts_id.append(csts_id[mask])
    all_vae_idx.append(vae_idx)
    all_vae_z.append(vae_z[mask])
csts = T.vstack(all_csts).to("cuda")
csts_id = T.hstack(all_csts_id).to("cuda")
all_vae_idx = T.hstack(all_vae_idx).to("cuda")
all_vae_z = T.vstack(all_vae_z).to("cuda")

# Create and fit the kmeans
kmeans = KMeans(1024, max_iter=500, verbose=10)
labels = kmeans.fit(csts.T.contiguous())

# Convert to numpy for plotting
csts_np = to_np(csts)
csts_id_np = to_np(csts_id)
labels_np = to_np(labels)
vae_z_np = to_np(all_vae_z)
vae_idx_np = to_np(all_vae_idx)

# Invert the pre-processing
csts_np = preprocessor.inverse_transform(csts_np)

# We want to order the constituents by their labels
order = vae_idx_np < 20  # np.argsort(vae_idx_np)[:50_000]  # only plot 100k
colors = vae_idx_np / vae_idx_np.max()

# Create a scatter plot of the constituents in pt, eta, and eta, phi
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for a in range(len(axes)):
    axes[a].scatter(
        csts_np[order][:, a],
        csts_np[order][:, a + 1],
        marker=".",
        c=colors[order],
        cmap="tab20",
    )
axes[0].set_xscale("log")
axes[0].set_xlabel("pt")
axes[0].set_ylabel("eta")
axes[1].set_xlabel("eta")
axes[1].set_ylabel("phi")

plt.savefig(root / "plots/kmeans_labels2.png")
plt.close("all")
