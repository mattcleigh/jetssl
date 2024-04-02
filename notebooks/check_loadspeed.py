from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodules.collation import collate_with_fn, minimize_padding
from src.datamodules.hdf import JetIterable
from src.datamodules.masking import random_masking
from src.datamodules.transforms import (
    apply_masking,
    compose,
    log_squash_csts_pt,
    tanh_d0_dz,
)

torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 600


# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ("csts", "f", [128]),
    ("csts_id", "f", [128]),
    ("mask", "bool", [128]),
    ("labels", "l"),
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

data = JetIterable(
    path="/srv/fast/share/rodem/JetClassH5/train_100M/",
    features=features,
    processes="all",
    n_classes=10,
    transforms=pipeline,
    shuffle=True,
)

loader = DataLoader(
    data,
    batch_size=1024,
    num_workers=2,
    collate_fn=partial(collate_with_fn, fn=minimize_padding),
)

# Cycle through the batches
avg_cardinality = 0
pbar = tqdm(loader)
for n_batches, batch in enumerate(pbar):
    # Unpack the batch
    csts = batch["csts"]
    csts_id = batch["csts_id"]
    labels = batch["labels"]
    mask = batch["mask"]
    null_mask = batch["null_mask"]

    # Update the average batch size
    avg_cardinality = (avg_cardinality * (n_batches - 1) + csts.shape[1]) / n_batches

    # Check the contents
    assert csts.shape[0] == csts_id.shape[0] == mask.shape[0], "Batch size mismatch"
    assert mask.sum(-1).min() > 0, "Events with 0 real nodes"
    assert (
        mask.sum(-1) - null_mask.sum(-1)
    ).min() > 0, "Events where everything was dropped"

    pbar.set_description(f"Average cardinality: {avg_cardinality:.2f}")
