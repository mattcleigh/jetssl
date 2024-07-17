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

    # Create the pandas dataframe to hold all the run information
    columns = ["model", "n_samples", "seed", "accuracy"]
    rows = []

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

            # Load the exported test set
            file_path = m / "outputs" / "test_set.h5"
            try:
                with h5py.File(file_path, "r") as f:
                    labels = f["label"][:]
                    outputs = f["output"][:]
            except FileNotFoundError:
                continue
            pred = np.argmax(outputs, axis=1, keepdims=True)
            acc = (labels == pred).mean() * 100  # Percentage

            # Add the information to the dataframe
            rows.append([model, n_samples, seed, acc])

            if n_samples == 100_000_000:
                print(f"{model} {n_samples} {seed} {acc}")

    df = pd.DataFrame(rows, columns=columns)

    # Sort the dataframe by the number of samples
    df = df.sort_values(by="n_samples")

    # Make the plots
    fig, ax = plt.subplots(figsize=(5, 5))

    # Cycle through the models
    for m in np.unique(df["model"]):
        # Get the data for this model
        data = df[df["model"] == m]
        data = data.drop(columns=["model"])

        # Combine the seeds into mean and std
        data = data.groupby(["n_samples"]).agg(["mean", "std"]).reset_index()

        # Plot the data
        line = ax.plot(
            data["n_samples"],
            data["accuracy"]["mean"],
            "-o",
            label=m,
        )

        # Add shaded region for the std if defined
        if data["accuracy"]["std"].notna().all():
            ax.fill_between(
                data["n_samples"],
                data["accuracy"]["mean"] - data["accuracy"]["std"],
                data["accuracy"]["mean"] + data["accuracy"]["std"],
                alpha=0.2,
                color=line[0].get_color(),
            )

        # Print the final accuracies
        n_samples = data["n_samples"]
        means = data["accuracy"]["mean"]
        stds = data["accuracy"]["std"]
        print(f"Model: {m}")
        for i in range(len(n_samples)):
            print(f"  {n_samples[i]}: {means[i]:.2f} +/- {stds[i]:.2f}")

    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
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
