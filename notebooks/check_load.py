import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import torch as T
from torch import nn

from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

jb = JetBackbone(
    encoder=Transformer(dim=512, num_layers=0),
    csts_emb=nn.Linear(7, 512),
    csts_id_emb=nn.Embedding(8, 512),
)
jb.eval()
T.save(
    jb, "/srv/beegfs/scratch/groups/rodem/jetssl/jetssl_fixed/backbones/no_backbone.pkl"
)

# from pathlib import Path

# import rootutils

# root = rootutils.setup_root(search_from=__file__, pythonpath=True)

# import hydra
# import torch as T

# from mltools.mltools.hydra_utils import reload_original_config
# from src.models.utils import JetBackbone

# model_dir = Path("/srv/beegfs/scratch/groups/rodem/jetssl/jetssl_fixed/mdm")
# target_dir = Path("/srv/beegfs/scratch/groups/rodem/jetssl/jetssl_fixed/backbones/")

# orig_cfg = reload_original_config(file_name="full_config.yaml", path=model_dir)
# model_class = hydra.utils.get_class(orig_cfg.model._target_)
# model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cpu")

# backbone = JetBackbone(
#     csts_emb=model.csts_emb,
#     csts_id_emb=model.csts_id_emb,
#     encoder=model.encoder,
# )
# backbone.eval()
# T.save(backbone, target_dir / "mdm.pkl")
