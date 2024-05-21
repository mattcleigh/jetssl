from functools import partial

import rootutils

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import torch as T
from torch.utils.data import DataLoader

from mltools.mltools.plotting import plot_multi_hists
from mltools.mltools.torch_utils import to_np
from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.preprocessing import batch_preprocess
from src.models.vae import JetVQVAE

# Define the type of information to load into the dict from the HDF files
# List containing: key, type, slice
features = [
    ["csts", "f", [128]],
    ["csts_id", "f", [128]],
    ["mask", "bool", [128]],
    ["labels", "l"],
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
preprocessor = joblib.load(root / "resources/preprocessor_all.joblib")
jc_loader = DataLoader(
    jc_data,
    batch_size=100,
    num_workers=0,
    shuffle=True,
    collate_fn=partial(batch_preprocess, fn=preprocessor),
)

# Load the VQ-VAE model
model_path = (
    "/srv/beegfs/scratch/groups/rodem/jetssl/jetssl2/vqvae/checkpoints/last.ckpt"
)
vae = JetVQVAE.load_from_checkpoint(model_path).to("cuda")
vae.eval()

# Cycle through the first 40 batches to get the preprocessed data
all_csts = []
all_recon = []
for i, batch in enumerate(jc_loader):
    if i > 10:
        break
    csts = batch["csts"]
    mask = batch["mask"]
    csts_id = batch["csts_id"]
    with T.autocast(device_type="cuda"):
        enc_outs, dec_outs, indices, vq_loss = vae(csts.to("cuda"), mask.to("cuda"))
    all_csts.append(csts[mask])
    all_recon.append(dec_outs[mask])
all_csts = to_np(T.vstack(all_csts))
all_recon = to_np(T.vstack(all_recon))

# Invert the pre-processing
all_csts = preprocessor.inverse_transform(all_csts)
all_recon = preprocessor.inverse_transform(all_recon)

# Plot the original and reconstructed data
plot_multi_hists(
    data_list=[all_csts, all_recon],
    data_labels=["Original", "Reconstructed"],
    col_labels=cst_features,
    bins=21,
    logy=True,
    path=root / "plots/original_reconstructed.png",
)
