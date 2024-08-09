from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

output_dir = "/srv/beegfs/scratch/groups/rodem/jetssl/"
project_name = "token"
network_names = ["vae", "kmeans", "none"]
processes = ["QCD", "TTbar"]

data = {n: [] for n in network_names}
rocs = {n: [] for n in network_names}
for net in network_names:
    path = Path(output_dir, project_name, net, "outputs", "test_set.h5")
    with h5py.File(path, "r") as f:
        labels = f["label"][:].squeeze()
        outputs = f["output"][:].squeeze()
    data[net] += [(net, labels, outputs)]
    for i in [0, 1]:
        pred = outputs[..., i]
        truth = labels == i
        fpr, tpr, _ = roc_curve(truth, pred)
        rocs[net] += [(fpr, tpr)]

fig, axis = plt.subplots(1, 2, figsize=(6, 3))

for i, ax in enumerate(axis):
    for net in network_names:
        fpr, tpr = rocs[net][i]
        a = auc(fpr, tpr)
        ax.plot(tpr, 1 / (fpr + 1e-12), label=net)
    ax.set_xlabel(r"$\epsilon_S$")
    ax.set_ylabel(r"1/$\epsilon_B$")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 1e7)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    ax.legend(title=processes[i], loc="upper right")

fig.tight_layout()
fig.savefig("roc_curve.pdf")
plt.close("all")
