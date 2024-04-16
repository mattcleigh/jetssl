from pathlib import Path

import h5py
import hydra
import numpy as np
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

        # For each varaint load the accuracies
        sics = []
        for variant in model_variants:
            file_path = variant / "outputs" / "test_set.h5"
            with h5py.File(file_path, "r") as f:
                labels = f["label"][:]
                outputs = f["output"][:]

            # Calculat the SIC at 1% fpr
            fpr, tpr, _ = roc_curve(labels, outputs)
            sic = tpr * np.sqrt(1 / (fpr + 1e-12))
            idx = np.argmin(np.abs(fpr - 0.01))
            sics.append(sic[idx])

        # Plot the accuracies
        ax.plot(n_samples, sics, "-o", label=model.split("_")[-1])

    # Tidy up
    ax.set_xlabel("Number of signal samples")
    ax.set_ylabel("Significance Improvement")
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
