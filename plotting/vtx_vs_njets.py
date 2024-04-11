from pathlib import Path

import h5py
import hydra
import numpy as np
import rootutils
import torch as T
from omegaconf import DictConfig
from sklearn.metrics import f1_score

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

    # There are 3 classes in the shlomi dataset
    classes = ["light", "charm", "bottom", "all"]

    # Create the figures, one for each class and one for the combined
    figs = {c: plt.subplots(1, 2, figsize=(8, 4)) for c in classes}

    # For each model find all variants
    for model in model_list:
        # Seach in the directory for everything matching the model name
        model_variants = list(Path(cfg.path).glob(model + "*"))
        n_samples = [int(m.name.split("_")[-1]) for m in model_variants]

        # Sort by the number of samples
        model_variants = [
            m for _, m in sorted(zip(n_samples, model_variants, strict=False))
        ]
        n_samples = np.clip(np.array(sorted(n_samples)), 1, 543544)

        # We will plot the F1 and accuracy scores for each sub-class
        accs = {c: [] for c in classes}
        f1s = {c: [] for c in classes}

        # For each varaint load the data from the output file
        for variant in model_variants:
            file_path = variant / "outputs" / "test_set.h5"
            with h5py.File(file_path, "r") as f:
                output = T.from_numpy(f["output"][:])
                mask = T.from_numpy(f["mask"][:])
                vtx_id = T.from_numpy(f["vtx_id"][:])
                labels = T.from_numpy(f["labels"][:])

            # Get the mask for the edges we are only looking at upper triangle
            vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            tri = T.triu(T.ones_like(vtx_mask[0]), diagonal=1)
            vtx_mask = vtx_mask & tri

            # Calculate the target based on if the vtx id matches
            target = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)
            target = target[vtx_mask]

            # Symmeterise the output matrix
            output_sym = 0.5 * (output + output.transpose(1, 2))
            output_sym = T.sigmoid(output_sym[vtx_mask])

            # Exand to get the labels per edge
            exp_labels = labels[..., None].expand_as(vtx_mask)[vtx_mask]

            # Measure the accuracy and F1 score for each class
            for i, c in enumerate(classes):
                if c == "all":
                    class_mask = T.ones_like(exp_labels).bool()
                else:
                    class_mask = exp_labels == i
                sel_out = output_sym[class_mask]
                sel_tar = target[class_mask]
                acc = T.mean(((sel_out > 0.5) == sel_tar).float()).item()
                f1 = f1_score(sel_tar, (sel_out > 0.5))
                accs[c].append(acc)
                f1s[c].append(f1)

        # Cycle through the classes and plot the results
        for c in classes:
            fig, axes = figs[c]
            axes[0].plot(n_samples, f1s[c], "-o", label=model.split("_")[1])
            axes[1].plot(n_samples, accs[c], "-o", label=model.split("_")[1])

    # Tidy up and save
    for c in classes:
        fig, axes = figs[c]
        axes[0].set_xlabel("Number of training samples")
        axes[1].set_xlabel("Number of training samples")
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("F1 score")
        axes[0].legend()
        axes[1].legend()
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        fig.tight_layout()

        # Make the directory and save the plot
        path = Path(cfg.plot_dir)
        path.mkdir(parents=True, exist_ok=True)
        nm = (
            "" if c == "all" else c
        )  # Snakemake is looking for a file without any flags
        fig.savefig(path / (cfg.outfile + f"{nm}.pdf"))
        plt.close()


if __name__ == "__main__":
    main()
