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
    ("jets", "f"),
]

jc_data = JetHDFStream(
    path="/srv/fast/share/rodem/JetClassH5/train_100M_combined.h5",
    features=features,
    n_classes=10,
    n_jets=10_000,
    transforms=[
        partial(batch_masking, fn=random_masking),
        partial(
            preprocess,
            fn=joblib.load(root / "resources/cst_quant.joblib"),
            hlv_fn=joblib.load(root / "resources/jet_quant.joblib"),
        ),
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

csts = []
jets = []
for batch in tqdm(loader):
    csts.append(batch["csts"][batch["mask"]])
    jets.append(batch["jets"])
csts = np.vstack(csts)
jets = np.vstack(jets)


plot_multi_hists(
    data_list=[csts],
    bins=51,
    data_labels=["JetClass"],
    col_labels=["pt", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr"],
    path=root / "cst.png",
    do_norm=True,
)

plot_multi_hists(
    data_list=[jets],
    bins=51,
    data_labels=["JetClass"],
    col_labels=["pt", "eta", "phi", "mass", "ncst"],
    path=root / "jets.png",
    do_norm=True,
)
