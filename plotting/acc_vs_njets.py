from pathlib import Path

import h5py
import hydra
import numpy as np
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

    # Create the figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # For each model find all variants
    for model in model_list:
        # Seach in the directory for everything matching the model name
        model_variants = list(Path(cfg.path).glob(model + "*"))
        n_samples = [int(m.name.split("_")[-1]) for m in model_variants]

        # Sort by the number of samples
        model_variants = [
            m for _, m in sorted(zip(n_samples, model_variants, strict=False))
        ]
        n_samples = np.array(sorted(n_samples))

        # Clip at the shlomi dataset size
        if "shlomi" in cfg.outfile:
            n_samples = np.clip(n_samples, 1, 543544)

        # For each varaint load the accuracies
        acc = []
        for variant in model_variants:
            file_path = variant / "outputs" / "test_set.h5"
            with h5py.File(file_path, "r") as f:
                labels = f["label"][:]
                outputs = f["output"][:]
            pred = np.argmax(outputs, axis=1, keepdims=True)
            acc.append((labels == pred).mean())

        # Plot the accuracies
        ax.plot(n_samples, acc, "-o", label=model.split("_")[1])

    # Tidy up
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_xscale("log")
    fig.tight_layout()

    # Make the directory and save the plot
    path = Path(cfg.plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / (cfg.outfile + ".pdf"))
    plt.close()


if __name__ == "__main__":
    main()
