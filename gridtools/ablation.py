import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    standard_job_array(
        job_name="pretraining",
        work_dir=root / "scripts",
        log_dir=root / "logs",
        image_path="/srv/fast/share/rodem/images/jetssl_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=1,
        gpu_type="ampere",
        vram_per_gpu=20,
        time_hrs=12,
        mem_gb=40,
        opt_dict={
            "network_name": [
                # "mpm-reg-3-noID-BERT-0reg",
                "mpm-kmeans-3-noID-BERT-0reg",
                # "mpm-reg-7-noID-BERT-0reg",
                "mpm-kmeans-7-noID-BERT-0reg",
                # "mpm-reg-7-yesID-BERT-0reg",
                "mpm-kmeans-7-yesID-BERT-0reg",
                # "mpm-reg-7-yesID-MAE-0reg",
                # "mpm-kmeans-7-yesID-MAE-0reg",
                # "mpm-reg-7-yesID-MAE-8reg",
                # "mpm-kmeans-7-yesID-MAE-8reg",
            ],
            "+model/tasks": [
                # "[reg,probe]",
                "[kmeans,probe]",
                # "[reg,probe]",
                "[kmeans,probe]",
                # "[reg,id,probe]",
                "[kmeans,id,probe]",
                # "[reg,id,probe]",
                # "[kmeans,id,probe]",
                # "[reg,id,probe]",
                # "[kmeans,id,probe]",
            ],
            "csts_dim": [
                # 3,
                3,
                # 7,
                7,
                # 7,
                7,
                # 7,
                # 7,
                # 7,
                # 7
            ],
            "model.do_mae": [
                # False,
                False,
                # False,
                False,
                # False,
                False,
                # True,
                # True,
                # True,
                # True,
            ],
            "model.encoder_config.num_registers": [
                # 0,
                0,
                # 0,
                0,
                # 0,
                0,
                # 0,
                # 0,
                # 8,
                # 8,
            ],
            "experiment": "mpmv1.yaml",
        },
        use_dashes=False,
        is_grid=False,
    )


if __name__ == "__main__":
    main()
