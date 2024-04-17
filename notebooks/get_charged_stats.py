from functools import partial

import rootutils
from tqdm import tqdm

root = rootutils.setup_root(search_from=".", pythonpath=True)

import torch as T
from joblib import dump
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader

from src.datamodules.hdf import JetMappable
from src.datamodules.transforms import (
    compose,
    log_squash_csts_pt,
    tanh_d0_dz,
)

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [128]),
    ("csts_id", "f", [128]),
    ("mask", "bool", [128]),
]

# Define the preprocessing pipeline
pipeline = partial(
    compose,
    transforms=[
        log_squash_csts_pt,
        tanh_d0_dz,
    ],
)

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=features,
    processes="all",
    n_classes=10,
    n_files=1,
    transforms=pipeline,
)

jc_loader = DataLoader(
    jc_data,
    batch_size=10_000,
    num_workers=0,
)

# Get the arrays
csts = []
csts_id = []
for data in tqdm(jc_loader):
    csts.append(data["csts"][data["mask"]])
    csts_id.append(data["csts_id"][data["mask"]])
csts = T.vstack(csts)
csts_id = T.hstack(csts_id)

# Replace the neutral impact parameters with NaNs (so they are disregarded in the fit)
neut_mask = (csts_id == 0) | (csts_id == 2)
csts[:, -4:][neut_mask] = T.nan

# Make a quantile transformer
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000)
qt.fit(csts)
dump(qt, "processor.joblib")

# Check how the transformation worked (just plot the neutrals particles)
transformed = qt.transform(csts[~neut_mask][:1000_000])
from mltools.mltools.plotting import plot_multi_hists

plot_multi_hists(
    transformed,
    data_labels=["transformed"],
    col_labels=["pt", "eta", "phi", "d0", "dz", "d0_err", "dz_err"],
    bins=100,
    path="transformed.png",
)
