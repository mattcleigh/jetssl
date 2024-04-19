import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    standard_job_array(
        job_name="jointembedders",
        work_dir=root / "scripts",
        log_dir=root / "logs",
        image_path="/srv/fast/share/rodem/images/jetssl_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=6,
        gpu_type="ampere",
        vram_per_gpu=20,
        time_hrs=7 * 24,
        mem_gb=40,
        opt_dict={
            "network_name": [
                "dino",
                "jepa",
            ],
            "model": [
                "dino",
                "jepa",
            ],
            "datamodule.loader_config.batch_size": [
                500,
                500,
            ],
            "experiment": [
                "pretrain.yaml",
                "jepa.yaml",
            ],
        },
        use_dashes=False,
        is_grid=False,
    )


if __name__ == "__main__":
    main()
