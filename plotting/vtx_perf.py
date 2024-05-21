from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import rootutils
import torch as T
from omegaconf import DictConfig
from sklearn.metrics import f1_score

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import matplotlib.pyplot as plt

from src.models.vertexer import get_ari


@hydra.main(
    version_base=None,
    config_path=str(root / "configs/plotting"),
    config_name="jets.yaml",
)
def main(cfg: DictConfig):
    # Load the list of models
    model_list = list(cfg.models.values())

    # Create a dataframe to hold the results per n_vertices in the jet
    cols = ["model", "n_samples", "seed", "n_vtx", "acc", "f1", "perf", "ari"]
    df = pd.DataFrame(columns=cols)

    # For each model find all variants and seeds
    for run in model_list:
        # Seach in the directory for everything matching the model name
        models = list(Path(cfg.path).glob(run + "*"))

        # Cycle through the models
        for m in models:
            # <dataset_name>_<model_name>_<n_samples>_<seed>
            _dset, model, n_samples, seed = m.name.split("_")
            n_samples = int(n_samples)
            seed = int(seed)
            print(model, n_samples, seed)

            # Clip at the shlomi dataset size
            n_samples = min(n_samples, 543544)

            file_path = m / "outputs" / "test_set.h5"
            try:
                with h5py.File(file_path, "r") as f:
                    output = T.from_numpy(f["output"][:])
                    mask = T.from_numpy(f["mask"][:])
                    vtx_id = T.from_numpy(f["vtx_id"][:])
            except FileNotFoundError:
                continue

            # We look at the upper triangle of edges
            vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            vtx_mask = T.triu(vtx_mask, diagonal=1)

            # Calculate the target based on if the vtx id matches
            target = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)
            target = target & vtx_mask

            # Get the predictions of the model (logits)
            preds = (output > 0) & vtx_mask

            # Get the number of secondary vertices in the jet
            metrics = {}
            n_vtx = vtx_id.max(-1)[0]
            for n in range(1, 7):
                n_vtx_mask = n_vtx == n

                # Apply the mask to pull out the class samples
                sel_preds = preds[n_vtx_mask]
                sel_target = target[n_vtx_mask]
                sel_vtx_mask = vtx_mask[n_vtx_mask]
                sel_vtx_id = vtx_id[n_vtx_mask]
                sel_output = output[n_vtx_mask]
                sel_mask = mask[n_vtx_mask]

                # Get the reductions for acc and f1
                corr = sel_preds == sel_target
                red_target = sel_target[sel_vtx_mask]
                red_preds = sel_preds[sel_vtx_mask]

                # Calculate the metrics
                metrics["acc"] = corr[sel_vtx_mask].float().mean().item()
                metrics["f1"] = f1_score(red_target, red_preds).item()
                metrics["perf"] = corr.all((-1, -2)).float().mean().item()
                metrics["ari"] = get_ari(sel_mask, sel_vtx_id, sel_output).mean().item()

                # Add the information to the dataframe
                row = {"model": model, "n_samples": n_samples, "seed": seed, "n_vtx": n}
                row = pd.DataFrame.from_dict({**row, **metrics}, orient="index").T
                df = pd.concat([df, row])

    # Cycle through the classes and plot the results
    met_labels = {
        "acc": "Accuracy",
        "f1": "F1 Score",
        "perf": "Fraction of Perfect Jets",
        "ari": "ARI",
    }
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(5, 5))

        for m in np.unique(df["model"]):
            # Get the data for this model
            data = df[df["model"] == m]
            data = data.drop(columns=["model"])

            # Combine the seeds into mean and std
            data = (
                data.groupby(["n_vtx"]).agg(["mean", "std", "min", "max"]).reset_index()
            )
            x = data["n_vtx"].to_numpy().astype("f")
            y = data[metric]["mean"].to_numpy().astype("f")

            # Plot the data
            line = ax.plot(x, y, "-o", label=m)

            # Add shaded region for the std if defined
            if data[metric]["std"].notna().all():
                down = data[metric]["min"].to_numpy().astype("f")
                up = data[metric]["max"].to_numpy().astype("f")
                # err = data[metric]["std"].to_numpy().astype("f")
                ax.fill_between(
                    x.astype("f"),
                    down,
                    up,
                    alpha=0.2,
                    color=line[0].get_color(),
                )

            ax.set_xlabel("Secondary vertices per jet")
            ax.set_ylabel(met_labels[metric])
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout()

            # Make the directory and save the plot
            path = Path(cfg.plot_dir)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path / f"{cfg.outfile}_{metric}.pdf")
            plt.close("all")

    # Snakemake expects a file to be created
    Path(path / (cfg.outfile + ".pdf")).touch()


if __name__ == "__main__":
    main()
