from pathlib import Path

import h5py
import hydra
import pandas as pd
import rootutils
import torch as T
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.models.vertexer import get_ari
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

    # Create a dataframe to hold the results per n_vertices in the jet
    cols = ["model", "n_samples", "seed", "n_vtx"]
    metrics = ["ari"]  # "acc", "f1", "perf"]
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

                # Load the output file
                file_path = m / "outputs" / "test_set.h5"
                try:
                    with h5py.File(file_path, "r") as f:
                        output = T.from_numpy(f["output"][:])
                        mask = T.from_numpy(f["mask"][:])
                        vtx_id = T.from_numpy(f["vtx_id"][:])
                except Exception as e:
                    print(e)
                    continue

                # We look at the upper triangle of edges
                vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                vtx_mask = T.triu(vtx_mask, diagonal=1)

                # Calculate the target based on if the vtx id matches
                target = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)
                target = target & vtx_mask

                # Get the predictions of the model (logits)
                # preds = (output > 0) & vtx_mask

                # Get the number of secondary vertices in the jet
                n_vtx = vtx_id.max(-1)[0]
                met_row = []
                for n in range(1, 7):
                    n_vtx_mask = n_vtx == n

                    # Apply the mask to pull out the class samples
                    # sel_preds = preds[n_vtx_mask]
                    # sel_target = target[n_vtx_mask]
                    # sel_vtx_mask = vtx_mask[n_vtx_mask]
                    sel_vtx_id = vtx_id[n_vtx_mask]
                    sel_output = output[n_vtx_mask]
                    sel_mask = mask[n_vtx_mask]

                    # Get the reductions for acc and f1
                    # corr = sel_preds == sel_target
                    # red_target = sel_target[sel_vtx_mask]
                    # red_preds = sel_preds[sel_vtx_mask]

                    # Calculate the metrics
                    # acc = corr[sel_vtx_mask].float().mean().item()
                    # f1 = f1_score(red_target, red_preds).item()
                    # perf = corr.all((-1, -2)).float().mean().item()
                    ari = get_ari(sel_mask, sel_vtx_id, sel_output).mean().item()
                    met_row.append([n, ari])

                # Save the stats
                stats = pd.DataFrame(met_row, columns=["n_vtx", "ari"])
                stats.to_csv(stat_path, index=False)

            # Add the information to the dataframe
            stats = stats.to_numpy()
            for n, ari in stats:
                rows.append([model, n_samples, seed, n, ari])

    # Combine the data into a pandas dataframe
    df = pd.DataFrame(rows, columns=cols + metrics)
    print_latex_table(df, "ari", "n_vtx")

    # Cycle through the classes and plot the results
    for metric in metrics:
        flag = f"{cfg.prefix}_{cfg.suffix}_{metric}"
        with open(root / "configs/plotting/plot_configs.yaml") as f:
            plot_kwargs = yaml.safe_load(f)[flag]

        plot_metric(
            df,
            model_list,
            path=Path(cfg.plot_dir, f"{flag}.pdf"),
            **plot_kwargs,
        )

    # Snakemake expects a file to be created
    Path(cfg.plot_dir, f"{cfg.prefix}.pdf").touch()


if __name__ == "__main__":
    main()
