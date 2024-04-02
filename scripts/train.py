"""Basic training script."""

import logging

import hydra
import pytorch_lightning as pl
import rootutils
import torch as T
from omegaconf import DictConfig

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)
from mltools.mltools.utils import save_declaration

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main training script."""
    log.info("Setting up full job config")
    if cfg.full_resume:
        old_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)
        if old_cfg is not None:
            cfg = old_cfg
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(
        cfg.model,
        data_sample=datamodule.get_data_sample(),
        n_classes=datamodule.n_classes,
    )

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)
        log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Declaring job as finished!")
    save_declaration("train_finished")


if __name__ == "__main__":
    main()
