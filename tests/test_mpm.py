import os
from functools import partial

import pytest
import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.lightning_utils import linear_warmup_exp_decay
from mltools.mltools.transformers import ClassAttentionPooling
from src.models.mpm import MaskedParticleModelling
from src.models.mpm_tasks import (
    DiffTask,
    FlowTask,
    IDTask,
    KmeansTask,
    ProbeTask,
    RegTask,
)

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
SCHED = partial(linear_warmup_exp_decay, warmup_steps=50000)

# The different task inits
id_task = partial(IDTask, name="csts_id")
reg_task = partial(RegTask, name="reg")
flow_task = partial(
    FlowTask,
    name="flow",
    embed_dim=32,
    flow_config={
        "num_stacks": 2,
        "mlp_width": 32,
        "mlp_depth": 1,
        "mlp_act": "SiLU",
        "tail_bound": 4.0,
        "dropout": 0.0,
        "num_bins": 4,
        "do_lu": False,
        "init_identity": True,
        "do_norm": False,
        "flow_type": "coupling",
    },
)
kmeans_task = partial(
    KmeansTask,
    name="kmeans",
    kmeans_config={
        "n_clusters": 5,
        "max_iter": 5,
    },
)
diff_task = partial(
    DiffTask,
    name="diff",
    embed_dim=32,
    diff_config={
        "time_dim": 8,
        "mlp_config": {
            "num_blocks": 1,
            "norm": "LayerNorm",
        },
    },
)
probe_task = partial(
    ProbeTask,
    name="probe",
    class_head=partial(
        ClassAttentionPooling,
        dim=16,
        do_input_linear=True,
        do_output_linear=True,
    ),
)


class DictDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def dummy_input() -> T.Tensor:
    """Create a dummy input dictionary for jets."""
    T.manual_seed(0)
    batch_size = 3
    num_csts = 10
    jet_dict = {}
    jet_dict["csts"] = T.randn(batch_size, num_csts, 7)
    jet_dict["csts_id"] = T.randint(0, 8, (batch_size, num_csts))
    jet_dict["mask"] = T.rand(batch_size, num_csts) > 0.1
    jet_dict["labels"] = T.randint(0, 3, (batch_size,))
    jet_dict["null_mask"] = (T.rand(batch_size, num_csts) > 0.5) & jet_dict["mask"]
    return jet_dict


# Test all tasks individually and then all together
tasks = [
    {"id": id_task},
    {"reg": reg_task},
    {"flow": flow_task},
    {"kmeans": kmeans_task},
    {"diff": diff_task},
    {"probe": probe_task},
]
tasks.append({k: v for t in tasks for k, v in t.items()})


@pytest.mark.parametrize("tasks", tasks)
@pytest.mark.parametrize("do_mae", [True, False])
@pytest.mark.parametrize("use_id", [True, False])
def test_base(tmpdir, tasks, do_mae, use_id) -> None:
    os.chdir(tmpdir)
    jet_dict = dummy_input()
    model = MaskedParticleModelling(
        data_sample={k: v[0] for k, v in jet_dict.items()},
        n_classes=3,
        encoder_config=ENCODER,
        decoder_config=DECODER,
        optimizer=OPT,
        scheduler=SCHED,
        tasks=tasks,
        use_id=do_mae,
        do_mae=use_id,
    )
    trainer = Trainer(fast_dev_run=True, accelerator="cpu")
    loader = DataLoader(DictDataset(jet_dict), batch_size=3)
    trainer.fit(model, loader, loader)
