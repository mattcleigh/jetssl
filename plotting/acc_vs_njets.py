from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
import rootutils
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.plotting import plot_metric, print_latex_table


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

    # Create the pandas dataframe to hold all the run information
    columns = ["model", "n_samples", "seed", "accuracy"]
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
                acc = stats["accuracy"][0]

            else:
                print(" - recomputing stats")

                # Load the exported test set
                file_path = m / "outputs" / "test_set.h5"
                try:
                    with h5py.File(file_path, "r") as f:
                        labels = f["label"][:]
                        outputs = f["output"][:]
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue

                # Calculate the accuracy
                pred = np.argmax(outputs, axis=1, keepdims=True)
                acc = (labels == pred).mean() * 100  # Percentage

                # Save the stats
                stats = pd.DataFrame({"accuracy": [acc]})
                stats.to_csv(stat_path, index=False)

            # Add the information to the dataframe
            rows.append([model, n_samples, seed, acc])

    # Combine the data into a pandas dataframe
    df = pd.DataFrame(rows, columns=columns)
    print_latex_table(df, "accuracy")

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
