import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.datamodules.hdf import JetClassMappable
from src.models2.labellers import KMeansLabeller, MultiKMeansLabeller

data = JetClassMappable(
    dset="val",
    n_files=1,
    path="/srv/beegfs/scratch/groups/rodem/datasets/JetClassH5",
    n_csts=32,
    processes="TTBar",
)

N = 100_000
num_labels = 256
csts = T.from_numpy(data.csts[:1000, :, :3])
mask = T.from_numpy(data.mask[:1000])

kmeans = KMeansLabeller(inpt_dim=csts.shape[-1], num_labels=num_labels, n_samples=10)
kmeans.fit(csts, mask)
kmeans.forward(csts, mask)

# Mask the constituents
m_csts = csts[mask]

# Plot the scatter plot of the cluster_centers
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(m_csts[:, 1], m_csts[:, 2], alpha=0.005)
ax[0].scatter(kmeans.cluster_centers[:, 1], kmeans.cluster_centers[:, 2], c="k")
pt_hist, bins = np.histogram(m_csts[:, 0], bins=100)
ax[1].stairs(pt_hist, bins)
ax[1].scatter(kmeans.cluster_centers[:, 0], np.ones(num_labels), c="k")
ax[1].set_yscale("log")
fig.tight_layout()
fig.savefig("kmeans.png")
plt.close()

kmeans = MultiKMeansLabeller(
    inpt_dim=3,
    num_labels=768,
    slices=[1, 2],
    labels_per_slice=[12, 64],
)
kmeans.fit(csts, mask)
idxes = kmeans.forward(csts, mask)
s0 = kmeans.sub_labellers[0]
s1 = kmeans.sub_labellers[1]

# Plot the scatter plot of the cluster_centers
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(m_csts[:, 1], m_csts[:, 2], alpha=0.005)
ax[0].scatter(s1.cluster_centers[:, 0], s1.cluster_centers[:, 1], c="k")
pt_hist, bins = np.histogram(m_csts[:, 0], bins=100)
ax[1].stairs(pt_hist, bins)
ax[1].scatter(s0.cluster_centers[:, 0], np.ones(s0.cluster_centers.shape[0]), c="k")
# ax[1].set_yscale("log")
fig.tight_layout()
fig.savefig("multi_kmeans.png")
plt.close()
