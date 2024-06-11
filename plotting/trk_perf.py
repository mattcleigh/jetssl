from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import matplotlib.pyplot as plt


@hydra.main(
    version_base=None,
    config_path=str(root / "configs/plotting"),
    config_name="jets.yaml",
)
def main(cfg: DictConfig):
    # Load the list of models
    model_list = list(cfg.models.values())

    # Create a dataframe to hold the results per n_vertices in the jet
    cols = ["model", "seed", "n_trk", "label", "eff", "pure", "f1"]
    df = pd.DataFrame(columns=cols)

    # For each model find all variants and seeds
    for run in model_list:
        # Seach in the directory for everything matching the model name
        models = list(Path(cfg.path).glob(run + "*"))

        # Cycle through the models
        for m in models:
            # <task_name>_<model_name>_<n_samples>_<seed>
            _task, model, n_samples, seed = m.name.split("_")
            n_samples = int(n_samples)
            seed = int(seed)
            print(model, n_samples, seed)

            file_path = m / "outputs" / "test_set.h5"
            try:
                with h5py.File(file_path, "r") as f:
                    output = f["output"][:]
                    track_type = f["track_type"][:]
                    mask = f["mask"][:]
                    labels = f["labels"][:]
            except Exception as e:
                print(e)
                continue

            # Turn the prediction into heavy or not (class 1 or 2)
            pred = np.isin(output.argmax(-1), [1, 2])
            target = np.isin(track_type, [1, 2])
            n_tracks = mask.sum(-1)

            # We will plot based on the number of tracks in the jet
            for jet_type in [1, 2]:  # light, charm, bottom
                for n_trk in range(2, 16):
                    # Mask to select the jets
                    sel_mask = (n_tracks == n_trk) & (labels.squeeze() == jet_type)
                    k = mask[sel_mask]
                    p = pred[sel_mask]
                    t = target[sel_mask]

                    # Purity = TP / (TP + FP)
                    p_num = (p & t & k).sum(-1)
                    p_div = (p & k).sum(-1)
                    pure = p_num / (p_div + 1e-12)

                    # Efficiency = TP / (TP + FN)
                    e_num = (p & t & k).sum(-1)
                    e_div = (t & k).sum(-1)
                    eff = e_num / (e_div + 1e-12)

                    # F1 = 2 * (P * R) / (P + R)
                    f1 = 2 * (pure * eff) / (pure + eff + 1e-12)

                    # Average only where the values are defined
                    pure = pure[p_div > 0].mean()
                    eff = eff[e_div > 0].mean()
                    f1 = f1[(p_div > 0) & (e_div > 0)].mean()

                    # Add the information to the dataframe
                    row = {
                        "model": model,
                        "seed": seed,
                        "n_trk": n_trk,
                        "label": jet_type,
                        "eff": eff,
                        "pure": pure,
                        "f1": f1,
                    }
                    row = pd.DataFrame.from_dict(row, orient="index").T
                    df = pd.concat([df, row])

    met_labels = {
        "eff": "Efficiency",
        "pure": "Purity",
        "f1": "F1-Score",
    }
    jet_labels = {
        1: "c-jets",
        2: "b-jets",
    }

    # We make a seperate plot for c and b jets
    for jet_type in [1, 2]:
        # Seperate plot for purity and efficiency
        for metric in ["eff", "pure", "f1"]:
            # Create the figure
            fig, ax = plt.subplots(figsize=(4, 4))

            # Cycle through each of the models to include
            for m in np.unique(df["model"]):
                # Get the data for this model, jet type and metric
                reqs = (df["model"] == m) & (df["label"] == jet_type)
                data = df[reqs]
                data = data.drop(columns=["model"])
                data = data.drop(columns=["label"])

                # Combine the seeds into mean and std
                data = data.groupby(["n_trk"]).agg(["mean", "min", "max"]).reset_index()

                # X values
                x = data["n_trk"].to_numpy().astype("f")

                # Y values with error
                y = data[metric]["mean"].to_numpy().astype("f")
                down = data[metric]["min"].to_numpy().astype("f")
                up = data[metric]["max"].to_numpy().astype("f")

                # Plot the data
                line = ax.plot(x, y, "-o", label=m)
                ax.fill_between(
                    x.astype("f"),
                    down,
                    up,
                    alpha=0.2,
                    color=line[0].get_color(),
                )

            ax.set_xlabel("Track Multiplicity")
            ax.set_ylabel(met_labels[metric])
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout()

            # Add text to the top left
            ax.text(0.05, 0.95, jet_labels[jet_type], transform=ax.transAxes)
            ax.set_ylim(0, 1.1)

            # Make the directory and save the plot
            path = Path(cfg.plot_dir)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path / f"{cfg.outfile}_{jet_type}_{metric}.pdf")
            plt.close("all")

            # Snakemake expects a file to be created
            Path(path / (cfg.outfile + ".pdf")).touch()


if __name__ == "__main__":
    main()
