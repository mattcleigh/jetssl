import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    depths = [1, 2, 3, 4]
    standard_job_array(
        job_name="depth_sweep",
        work_dir=root / "scripts",
        log_dir=root / "logs",
        image_path="/srv/fast/share/rodem/images/jetssl_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=10,
        gpu_type="ampere",
        # vram_per_gpu=20,
        time_hrs=12,
        mem_gb=20,
        opt_dict={
            "network_name": [f"mae-kmeans-depth{d}" for d in depths],
            "+model/tasks": "[kmeans,id,probe]",
            "experiment": "pretrain.yaml",
            "model.decoder_config.num_layers": depths,
            "datamodule.batch_size": 512,
        },
        use_dashes=False,
        is_grid=False,
    )


if __name__ == "__main__":
    main()
