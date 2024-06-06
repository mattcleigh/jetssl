from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import rootutils
from omegaconf import DictConfig
from sklearn.metrics import roc_curve

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

    # Create the pandas dataframe to hold all the run information
    df = pd.DataFrame(columns=["model", "n_samples", "seed", "sic"])

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

            # Load the SIC from the exported test set
            file_path = m / "outputs" / "test_set.h5"
            try:
                with h5py.File(file_path, "r") as f:
                    labels = f["label"][:]
                    outputs = f["output"][:]
                    if np.isnan(outputs).any():
                        print(f"nan found in {file_path}")
                        continue
            except Exception as e:
                print(e)
                continue

            # Calculat the SIC at 1% fpr
            fpr, tpr, _ = roc_curve(labels, outputs)
            idx = np.argmin(np.abs(fpr - 0.01))
            sic = tpr[idx] * np.sqrt(1 / (fpr[idx] + 1e-12))

            # Add the information to the dataframe\
            row = pd.DataFrame([[model, n_samples, seed, sic]], columns=df.columns)
            df = pd.concat([df, row])

    # Sort the dataframe by the number of samples
    df = df.sort_values(by="n_samples")
    df = df.reset_index(drop=True)

    # Drop index 10
    # df = df.drop(index=10)

    # Make the plots
    fig, ax = plt.subplots(figsize=(5, 5))

    # Cycle through the models
    for m in np.unique(df["model"]):
        # Get the data for this model
        data = df[df["model"] == m]
        data = data.drop(columns=["model"])

        # Combine the seeds into mean and std
        data = data.groupby(["n_samples"]).agg(["mean", "min", "max"]).reset_index()
        data = data.astype("f")

        # Plot the data
        line = ax.plot(
            data["n_samples"] // 2,
            data["sic"]["mean"],
            "-o",
            label=m,
        )

        # Add shaded region for the std if defined
        ax.fill_between(
            data["n_samples"] // 2,
            data["sic"]["min"],
            data["sic"]["max"],
            alpha=0.2,
            color=line[0].get_color(),
        )

    ax.set_xlabel("Number of Signal Samples")
    ax.set_ylabel("Significance Improvement")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()

    # Make the directory and save the plot
    path = Path(cfg.plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / (cfg.outfile + ".pdf"))
    plt.close()


if __name__ == "__main__":
    main()
