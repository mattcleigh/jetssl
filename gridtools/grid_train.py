import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    standard_job_array(
        job_name="pretraining",
        work_dir="/home/users/l/leighm/DiffBEIT/scripts",
        log_dir="/home/users/l/leighm/DiffBEIT/logs/",
        image_path="/srv/fast/share/rodem/images/diffbeit-image_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=6,
        gpu_type="ampere",
        vram_per_gpu=20,
        time_hrs=4 * 24,
        mem_gb=40,
        opt_dict={
            "network_name": [
                "dino",
                "reg",
                "token",
                "diff",
                "onlyid",
            ],
            "model": [
                "jetdino",
                "mpmreg",
                "mpmtoken",
                "mpmdiff",
                "mpmonlyid",
            ],
        },
        use_dashes=False,
    )


if __name__ == "__main__":
    main()
