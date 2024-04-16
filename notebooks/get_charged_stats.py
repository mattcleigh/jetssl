from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

from joblib import dump
from sklearn.preprocessing import QuantileTransformer

from src.datamodules.hdf import JetMappable
from src.datamodules.masking import random_masking
from src.datamodules.transforms import (
    apply_masking,
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
        partial(apply_masking, masking_fn=partial(random_masking, mask_fraction=0.5)),
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

# Get the arrays
jc_data.data_dict = jc_data.data_dict
csts = jc_data.data_dict["csts"]
mask = jc_data.data_dict["mask"]
csts_id = jc_data.data_dict["csts_id"]

# Get the charged particles
charged_mask = mask & ((csts_id != 0) & (csts_id != 2))
charged = csts[..., -4:][charged_mask]  # We only want the impact parameters

import numpy as np

len(np.unique(charged[:, 1]))

# Make a quantile transformer for the charged particles
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000)
qt.fit(charged)
dump(qt, "impact_processor.joblib")

# Check how fast the transformation is
import time

ts = time.time()
transformed = qt.transform(charged[:1000_000])
from mltools.mltools.plotting import plot_multi_hists

plot_multi_hists(
    transformed,
    data_labels=["transformed"],
    col_labels=["d0", "dz", "d0_err", "dz_err"],
    bins=100,
    path="transformed.png",
)

te = time.time()
print(f"Time to fit the quantile transformer: {te - ts} s")
