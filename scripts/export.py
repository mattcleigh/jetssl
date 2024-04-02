import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from pathlib import Path

import h5py
import hydra
import torch as T
from omegaconf import DictConfig

from mltools.mltools.hydra_utils import reload_original_config
from mltools.mltools.torch_utils import to_np

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(ckpt_flag="*best*" if cfg.get_best else "*last*")

    log.info("Loading best checkpoint")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cpu")

    log.info("Instantiating original trainer")
    trainer = hydra.utils.instantiate(orig_cfg.trainer)

    log.info("Instantiating the original datamodule")
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

    log.info("Running inference")
    outputs = trainer.predict(model=model, datamodule=datamodule)

    log.info("Combining predictions across dataset")
    keys = list(outputs[0].keys())
    score_dict = {k: T.vstack([o[k] for o in outputs]) for k in keys}

    log.info("Saving outputs")
    output_dir = Path(orig_cfg.paths.full_path, "outputs")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_dir / "test_set.h5", mode="w") as file:
        for k in keys:
            file.create_dataset(k, data=to_np(score_dict[k]))


if __name__ == "__main__":
    main()
