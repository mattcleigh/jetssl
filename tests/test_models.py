from functools import partial

import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.lightning_utils import linear_warmup
from mltools.mltools.transformers import ClassAttentionPooling
from src.models.jetdino import JetDINO
from src.models.mpm_base import MPMBase
from src.models.mpm_diff import MPMDiff
from src.models.mpm_flow import MPMFlow
from src.models.mpm_reg import MPMReg
from src.models.mpm_token import MPMToken
from src.utils import KMeansLabeller

# All the default arguments for the models
ENCODER = {
    "dim": 32,
    "num_layers": 2,
    "do_input_linear": False,
    "do_absolute_enc": False,
    "num_registers": 2,
    "do_final_norm": False,
    "layer_config": {
        "num_heads": 2,
        "ff_mult": 2,
    },
}
DECODER = {
    "dim": 32,
    "num_layers": 2,
    "do_input_linear": False,
    "num_registers": 0,
    "do_final_norm": False,
    "layer_config": {
        "num_heads": 4,
        "ff_mult": 2,
    },
}
OPT = partial(T.optim.Adam, lr=1e-3)
SCHED = partial(linear_warmup, warmup_steps=50000)
HEAD = partial(
    ClassAttentionPooling, dim=16, do_input_linear=True, do_output_linear=True
)


def dummy_input() -> T.Tensor:
    """Create a dummy input dictionary for jets."""
    T.manual_seed(0)
    batch_size = 3
    num_csts = 32
    jet_dict = {}
    jet_dict["csts"] = T.randn(batch_size, num_csts, 7)
    jet_dict["csts_id"] = T.randint(0, 8, (batch_size, num_csts))
    jet_dict["mask"] = T.rand(batch_size, num_csts) > 0.1
    jet_dict["labels"] = T.randint(0, 3, (batch_size,))
    jet_dict["null_mask"] = (T.rand(batch_size, num_csts) > 0.5) & jet_dict["mask"]
    return jet_dict


def test_base() -> None:
    jet_dict = dummy_input()
    model = MPMBase(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
    )
    model.training_step(jet_dict, 0)


def test_reg() -> None:
    jet_dict = dummy_input()
    model = MPMReg(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
    )
    model.training_step(jet_dict, 0)


def test_diff() -> None:
    jet_dict = dummy_input()
    model = MPMDiff(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
        embed_dim=32,
        diff_config={
            "time_dim": 8,
            "mlp_config": {
                "num_blocks": 2,
                "num_layers_per_block": 2,
                "norm": "LayerNorm",
                "do_res": True,
            },
        },
    )
    model.training_step(jet_dict, 0)


def test_token() -> None:
    jet_dict = dummy_input()
    model = MPMToken(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
        labeller=partial(KMeansLabeller, num_labels=5),
    )
    model.training_step(jet_dict, 0)


def test_flow() -> None:
    jet_dict = dummy_input()
    model = MPMFlow(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
        embed_dim=32,
        flow_config={
            "num_stacks": 6,
            "mlp_width": 64,
            "mlp_depth": 2,
            "mlp_act": "SiLU",
            "tail_bound": 4.0,
            "dropout": 0.0,
            "num_bins": 8,
            "do_lu": False,
            "init_identity": True,
            "do_norm": False,
            "flow_type": "coupling",
        },
    )
    model.training_step(jet_dict, 0)


def test_dino() -> None:
    jet_dict = dummy_input()
    model = JetDINO(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        optimizer=OPT,
        scheduler=SCHED,
        class_head=HEAD,
    )
    model.training_step(jet_dict, 0)
