from functools import partial
from time import perf_counter

import numpy as np
import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.torch_utils import move_dev
from src.models.mpm import MaskedParticleModelling
from src.models.mpm_tasks import IDTask, KmeansTask, RegTask

T.set_float32_matmul_precision("high")


def time_torch(func, *args, steps=100, **kwargs) -> tuple:
    with T.autocast(device_type="cuda", dtype=T.float16):
        for _ in range(steps):
            func(*args)
        T.cuda.synchronize()
        times = []
        for _ in range(steps):
            ts = perf_counter()
            func(*args)
            T.cuda.synchronize()
            te = perf_counter()
            times.append(te - ts)
        return np.mean(times), np.std(times)


batch_size = 128
num_csts = 128
dim = 256
jet_dict = {}
jet_dict["csts"] = T.randn(batch_size, num_csts, 7)
jet_dict["csts_id"] = T.randint(0, 8, (batch_size, num_csts))
jet_dict["mask"] = T.rand(batch_size, num_csts) > 0.5
jet_dict["mask"][:, 0] = True
jet_dict["labels"] = T.randint(0, 3, (batch_size,))
jet_dict["null_mask"] = (T.rand(batch_size, num_csts) > 0.5) & jet_dict["mask"]
jet_dict["jets"] = T.rand(batch_size, 5) > 0.5
jet_dict = move_dev(jet_dict, "cuda")

# Test the model
model = MaskedParticleModelling(
    data_sample={k: v[0] for k, v in jet_dict.items()},
    n_classes=3,
    encoder_config={"num_layers": 2, "dim": dim, "do_packed": True},
    decoder_config={"num_layers": 2, "dim": dim, "do_packed": True},
    optimizer=None,
    scheduler=None,
    tasks={
        "reg": partial(RegTask, name="reg"),
        "id": partial(IDTask, name="id"),
        "kmeans": partial(
            KmeansTask,
            name="kmeans",
            kmeans_path="/home/users/l/leighm/jetssl/resources/kmeans_16384.pkl",
        ),
    },
    use_id=True,
    objective="mae",
)
model = model.to("cuda")
m, v = time_torch(model, jet_dict)

# Clear all gradients
T.cuda.empty_cache()

# Try compiling the model
model_c = T.compile(model)
# out = model_c(jet_dict)

mc, vc = time_torch(model_c, jet_dict)
print(f"Original: {m:.4f} ± {v:.4f}")
print(f"Compiled: {mc:.4f} ± {vc:.4f}")
