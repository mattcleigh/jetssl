from functools import partial

import joblib
import numpy as np
import rootutils
from torch.utils.data import DataLoader
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.plotting import plot_multi_hists
from src.datamodules.hdf_stream import BatchSampler, JetHDFStream
from src.datamodules.masking import random_masking
from src.datamodules.preprocessing import batch_masking, preprocess

features = [
    ("csts", "f", [32]),
    ("csts_id", "f", [32]),
    ("mask", "bool", [32]),
    ("labels", "l"),
]

jc_data = JetHDFStream(
    path="/srv/fast/share/rodem/JetClassH5/train_100M_combined.h5",
    features=features,
    n_classes=10,
    # n_jets_total=100_000,
    transforms=[
        partial(preprocess, fn=joblib.load(root / "resources/preprocessor_all.joblib")),
        partial(batch_masking, fn=random_masking),
    ],
)

loader = DataLoader(
    dataset=jc_data,
    batch_size=None,  # batch size is handled by the sampler
    collate_fn=None,
    shuffle=False,
    sampler=BatchSampler(jc_data, batch_size=1000, shuffle=True),
    num_workers=6,
)

# Plot the first batch
csts = [batch["csts"][batch["mask"]] for batch in tqdm(loader)]
csts = np.concatenate(csts)

plot_multi_hists(
    data_list=[csts],
    bins=51,
    logy=True,
    data_labels=["JetClass"],
    col_labels=["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"],
    path=root / "plots/batch2.png",
    do_norm=True,
)
