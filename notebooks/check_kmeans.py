from functools import partial

import rootutils
from tqdm import tqdm

root = rootutils.setup_root(search_from=".", pythonpath=True)

import joblib
import torch as T
from torch.utils.data import DataLoader
from torchpq.clustering import KMeans

from src.datamodules.hdf import JC_CLASS_TO_LABEL, JetMappable
from src.datamodules.preprocessing import batch_preprocess

jc_data = JetMappable(
    path="/srv/fast/share/rodem/JetClassH5/val_5M/",
    features=None,
    csts_dim=7,
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
all_ids = []
for i, batch in enumerate(tqdm(jc_loader)):
    csts = batch["csts"]
    mask = batch["mask"]
    ids = batch["csts_id"]
    all_csts.append(csts[mask])
    all_ids.append(ids[mask])
    if i == 1000:
        break
all_csts = T.vstack(all_csts).to("cuda")
all_ids = T.hstack(all_ids).to("cuda")

# Calculate the weights for classification with the ids
vals, counts = T.unique(all_ids, return_counts=True)
weights = 1 / counts.float()
weights /= weights.mean()
print("Weights for ID classification:")
print(weights.cpu().numpy())

# Create and fit the kmeans
kmeans = KMeans(16384, max_iter=100, verbose=10)
labels = kmeans.fit(csts.T.contiguous())
values = kmeans.centroids.index_select(1, labels).T
out = kmeans.predict(all_csts.T.contiguous()).long()
vals, counts = T.unique(out, return_counts=True)
assert (vals == T.arange(16384, device="cuda").long()).all()
weights = 1 / counts.float()
weights /= weights.mean()
kmeans.register_buffer("weights", weights)
T.save(kmeans, root / "resources/kmeans_7.pkl")

# Convert to numpy for plotting
# csts_np = to_np(csts[:1000_000])
# values_np = to_np(values[:1000_000])

# Invert the pre-processing
# csts_np = preprocessor.inverse_transform(csts_np)
# values_np = preprocessor.inverse_transform(values_np)

# Plot
# plot_multi_hists(
#     data_list=[csts_np, values_np],
#     data_labels=["Original", "Reconstructed"],
#     col_labels=cst_features,
#     bins=30,
#     logy=True,
#     do_norm=True,
#     path=root / "plots/kmeans_reconstruction.png",
# )
