from functools import partial

import rootutils
from tqdm import tqdm

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
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
preprocessor = joblib.load(root / "resources/cst_quant.joblib")
jc_loader = DataLoader(
    jc_data,
    batch_size=1_000,
    num_workers=4,
    shuffle=True,
    collate_fn=partial(batch_preprocess, fn=preprocessor),
)

# Cycle through the first 40 batches to get the preprocessed data
all_csts = []
for i, batch in enumerate(tqdm(jc_loader)):
    csts = batch["csts"]
    mask = batch["mask"]
    all_csts.append(csts[mask])
    if i == 40:
        break
csts = T.vstack(all_csts).to("cuda")

# Create and fit the kmeans
kmeans = KMeans(16384, max_iter=1000, verbose=10)
labels = kmeans.fit(csts.T.contiguous())
values = kmeans.centroids.index_select(1, labels).T
T.save(kmeans, root / "resources/kmeans.pkl")

# Convert to numpy for plotting
csts_np = to_np(csts[:1000_000])
values_np = to_np(values[:1000_000])

# Invert the pre-processing
csts_np = preprocessor.inverse_transform(csts_np)
values_np = preprocessor.inverse_transform(values_np)

# Plot
plot_multi_hists(
    data_list=[csts_np, values_np],
    data_labels=["Original", "Reconstructed"],
    col_labels=cst_features,
    bins=30,
    logy=True,
    do_norm=True,
    path=root / "plots/kmeans_reconstruction.png",
)
