from functools import partial

import joblib
import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodules.collation import batch_preprocess_impact
from src.datamodules.hdf import JetMappable
from src.datamodules.masking import random_masking
from src.datamodules.transforms import (
    apply_masking,
    compose,
    log_squash_csts_pt,
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
        partial(apply_masking, masking_fn=partial(random_masking, mask_fraction=0.5)),
        log_squash_csts_pt,
    ],
)

data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/train_100M/",
    features=features,
    n_files=1,
    processes="TTBar",
    n_classes=1,
    transforms=pipeline,
)

loader = DataLoader(
    data,
    batch_size=1024,
    num_workers=4,
    shuffle=True,
    collate_fn=partial(
        batch_preprocess_impact,
        fn=joblib.load(root / "src/datamodules/impact_processor.joblib"),
    ),
)

# Cycle through the batches
for batch in tqdm(loader):
    # Unpack the batch
    csts = batch["csts"]
    csts_id = batch["csts_id"]
    labels = batch["labels"]
    mask = batch["mask"]
