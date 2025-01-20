from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import rootutils
import yaml
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.plotting import plot_metric, print_latex_table

np.seterr(divide="ignore", invalid="ignore")


def softmax(x: np.ndarray, axis: int = -1, temp: float = 1.0):
    """Compute the softmax of an array of values."""
    e_x = np.exp(x / temp)
    return e_x / e_x.sum(axis=axis, keepdims=True)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs/plotting"),
    config_name="default.yaml",
)
def main(cfg: DictConfig):
    model_path = Path(f"{cfg.path}_{cfg.suffix}")
    model_list = list(cfg.models.values())
    if cfg.suffix == "frozen":
        model_list.append("nobackbone")
    elif cfg.suffix == "finetune":
        model_list.append("untrained")

    # Create a dataframe to hold the results per n_vertices in the jet
    cols = ["model", "n_samples", "seed", "n_trk", "acc"]
    rows = []

    # For each model find all variants and seeds
    for run in tqdm(model_list):
        # Seach in the directory for everything matching the model name
        models = list(model_path.glob(f"{cfg.prefix}_{run}*"))

        # Cycle through the models
        for m in tqdm(models, leave=False):
            # <task_name>_<model_name>_<n_samples>_<seed>
            _task_name, model, n_samples, seed = m.name.split("_")
            n_samples = int(n_samples)
            seed = int(seed)

            # Check if the stats has already been saved
            stat_path = m / "outputs" / "stats.h5"
            if stat_path.exists() and not cfg.force_recompute:
                stats = pd.read_csv(stat_path)

            else:
                print(" - recomputing stats")

                file_path = m / "outputs" / "test_set.h5"
                try:
                    with h5py.File(file_path, "r") as f:
                        output = f["output"][:]
                        track_type = f["track_type"][:]
                        mask = f["mask"][:]
                        # labels = f["labels"][:]
                except Exception as e:
                    print(e)
                    continue

                # Turn the prediction into heavy or not (class 1 or 2)
                pred = np.argmax(output, axis=-1)

                # Get the accuracy per track multiplicity
                accs = []
                for n_trk in range(5, 16):
                    reqs = mask.sum(-1) == n_trk
                    mask_ = mask[reqs]
                    pred_ = pred[reqs][mask_]
                    true_ = track_type[reqs][mask_]
                    acc = balanced_accuracy_score(true_, pred_) * 100
                    accs.append([n_trk, acc])

                # Save the stats
                stats = pd.DataFrame(accs, columns=["n_trk", "acc"])
                stats.to_csv(stat_path, index=False)

            # Add the information to the dataframe
            stats = stats.to_numpy()
            for n_trk, roc in stats:
                rows.append([model, n_samples, seed, n_trk, roc])

    # Combine the data into a pandas dataframe
    df = pd.DataFrame(rows, columns=cols)
    print_latex_table(df, "acc", "n_trk")

    flag = f"{cfg.prefix}_{cfg.suffix}"
    with open(root / "configs/plotting/plot_configs.yaml") as f:
        plot_kwargs = yaml.safe_load(f)[flag]

    plot_metric(
        df,
        model_list,
        path=Path(cfg.plot_dir, flag + ".pdf"),
        **plot_kwargs,
    )


if __name__ == "__main__":
    main()

# Removed from the loop above
# Calculate the purity and efficiency of the dataset
# num = (pred & target & mask).sum(-1)  # true positive
# pure = num / (pred & mask).sum(-1)  # Precision
# eff = num / (target & mask).sum(-1)  # recall
# f1 = 2 / (1 / eff + 1 / pure)

# # We will plot based on the number of tracks in the jet
# n_tracks = mask.sum(-1)
# for jet_type in [1, 2]:  # light, charm, bottom
#     for n_trk in range(2, 16):
#         # Mask to select the jets
#         sel_mask = (n_tracks == n_trk) & (labels.squeeze() == jet_type)

#         # Add the information to the dataframe
#         row = {
#             "model": model,
#             "seed": seed,
#             "n_trk": n_trk,
#             "label": jet_type,
#             "eff": np.nanmean(pure[sel_mask]),
#             "pure": np.nanmean(eff[sel_mask]),
#             "f1": np.nanmean(f1[sel_mask]),
#         }
#         row = pd.DataFrame.from_dict(row, orient="index").T
#         df = pd.concat([df, row])
