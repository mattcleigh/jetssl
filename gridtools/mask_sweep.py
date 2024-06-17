import numpy as np
import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    frs = list(np.round(np.arange(0.1, 1.0, 0.1), 1))
    standard_job_array(
        job_name="mask_sweep",
        work_dir=root / "scripts",
        log_dir=root / "logs",
        image_path="/srv/fast/share/rodem/images/jetssl_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=6,
        gpu_type="ampere",
        vram_per_gpu=20,
        time_hrs=12,
        mem_gb=20,
        opt_dict={
            "mask_fraction": frs,
            "network_name": [f"mae-kmeans3-mf{fr}" for fr in frs],
            "+model/tasks": "[kmeans,id,probe]",
            "experiment": "pretrain.yaml",
        },
        use_dashes=False,
        is_grid=False,
    )


if __name__ == "__main__":
    main()
