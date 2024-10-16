from functools import partial
from pathlib import Path

import hydra
import joblib
import numpy as np
import rootutils
import torch as T
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from mltools.mltools import plotting as _
from mltools.mltools.torch_utils import to_np

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import reload_original_config
from src.datamodules.hdf_stream import BatchSampler, JetHDFStream
from src.datamodules.masking import random_masking
from src.datamodules.preprocessing import batch_masking, preprocess
from src.plotting import _COLORS, _LABELS  # noqa: PLC2701


def get_data():
    features = [
        ("csts", "f", [128]),
        ("csts_id", "f", [128]),
        ("mask", "bool", [128]),
        ("labels", "l"),
        ("jets", "f"),
    ]

    jc_data = JetHDFStream(
        path="/srv/fast/share/rodem/JetClassH5/train_100M_combined.h5",
        features=features,
        n_classes=10,
        n_jets=1_000,
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
        sampler=BatchSampler(jc_data, batch_size=1000),
        num_workers=0,
    )

    batch = next(iter(loader))
    n_csts = batch["null_mask"].sum(dim=1)
    top3 = n_csts.argsort(descending=True)[:3]

    data = {k: v[top3].to("cuda") for k, v in batch.items()}
    data["csts_id"] = data["csts_id"].long()
    return data


def get_prediction(data, m_name):
    model_dir = Path(f"/srv/beegfs/scratch/groups/rodem/jetssl/jetssl_fixed/{m_name}")

    orig_cfg = reload_original_config(file_name="full_config.yaml", path=model_dir)
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cuda")

    with T.cuda.amp.autocast():
        data["outputs"] = model.apply_pass(data)
        return model.tasks[m_name]._visualise(model, data)


data = get_data()

# Now load the models and get the predictions
preds = {}
for m_name in ["reg", "diff", "flow", "kmeans"]:
    preds[m_name] = get_prediction(data, m_name)

csts = data["csts"]
mask = data["mask"]
null_mask = data["null_mask"]

# Create a copy of the csts_id tensor with the predicted values
pred_csts = {k: csts.clone() for k, v in preds.items()}
for k, v in preds.items():
    pred_csts[k][null_mask] = v.type(pred_csts[k].dtype)

# Convert all the tensors to numpy
csts = to_np(csts)
mask = to_np(mask)
null_mask = to_np(null_mask)
pred_csts = {k: to_np(v) for k, v in pred_csts.items()}

# Undo the preprocessing
preproc = joblib.load(root / "resources/cst_quant.joblib")
csts[mask] = preproc.inverse_transform(csts[mask])
for k in pred_csts:
    pred_csts[k][mask] = preproc.inverse_transform(pred_csts[k][mask])

# Cycle through the batch
for b in range(3):
    # Select the current jet
    c = csts[b]
    m = mask[b]
    nm = null_mask[b]

    # Split the features into the original, survived and sampled
    original = c[m]
    survived = c[m & ~nm]

    # Create the figure and axes
    labels = [r"$p_T$ [GeV]", r"$\Delta \eta$", r"$\Delta \phi$"]
    lims = [[0, 60], [-1, 1], [-1, 1]]

    # Cycle through the features
    for i in range(3):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))

        # Create the bins and clip to include overflow/underflow
        bins = np.linspace(*lims[i], 21)
        original[:, i] = np.clip(original[:, i], bins[0], bins[-1])
        survived[:, i] = np.clip(survived[:, i], bins[0], bins[-1])

        # Plot the histogram of the original jets
        o_hist, _ = np.histogram(original[:, i], bins=bins)
        ax.stairs(o_hist, bins, fill=True, alpha=0.4, color="k", label="Original")

        # Plot the histogram of the survived jets
        s_hist, _ = np.histogram(survived[:, i], bins=bins)
        ax.stairs(s_hist, bins, fill=True, alpha=0.4, color="b", label="Survived")

        # Stack ontop of that a histogram of the sampled jets
        for k, sampled in pred_csts.items():
            p_hist, _ = np.histogram(sampled[b][m][:, i], bins=bins)
            ax.stairs(p_hist, bins, color=_COLORS[k], label=_LABELS[k], linewidth=2)

        ax.set_ylabel("Counts")
        ax.set_xlabel(labels[i])
        ax.set_xlim(*lims[i])
        # ax.set_yscale("log")

        # Get the highest value to set the yscale
        max_val = max([o_hist.max(), s_hist.max(), p_hist.max()])
        # ax.set_ylim(1, max_val * 1.6)

        if i == 0:
            ax.legend(title=f"Jet {b}", loc="upper right")
        else:
            ax.legend(title=f"Jet {b}", loc="upper left")
        fig.tight_layout()
        fig.savefig(f"plots/jet_{b}_{i}.pdf")
        plt.close()
