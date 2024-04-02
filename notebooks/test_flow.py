# Import required packages
import numpy as np
import rootutils
import torch as T
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch import nn
from torchdiffeq import odeint
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import CosineEncodingLayer


class VectorDiffuser(nn.Module):
    """Flow-Matching MLP for generating a single vector."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        ctxt_dim: int,
        mlp_config: dict,
        time_dim: int = 8,
    ) -> None:
        super().__init__()
        self.time_encoder = CosineEncodingLayer(inpt_dim=1, encoding_dim=time_dim)
        self.mlp = MLP(
            inpt_dim=inpt_dim,
            outp_dim=inpt_dim,
            ctxt_dim=ctxt_dim + time_dim,
            **mlp_config,
        )

    def forward(self, xt: T.Tensor, t: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        """Get the fully denoised estimate."""
        c = T.cat([self.time_encoder(t), ctxt], dim=-1)
        return self.mlp(xt, c)

    def get_loss(self, x0: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        t = T.sigmoid(T.randn(x0.shape[0], 1, device=x0.device))
        x1 = T.randn_like(x0)
        xt = (1 - t) * x0 + t * x1
        v = self.forward(xt, t, ctxt)
        return (v - (x1 - x0)).square().mean()

    def generate(self, x1: T.Tensor, ctxt: T.Tensor, times: T.Tensor) -> T.Tensor:
        """Generate a sample."""

        def fn(t, xt):
            return self(xt, t * xt.new_ones([xt.shape[0], 1]), ctxt)

        return odeint(fn, x1, times, method="midpoint")[-1]


diff = VectorDiffuser(
    inpt_dim=2,
    ctxt_dim=1,
    time_dim=16,
    mlp_config={
        "num_blocks": 4,
        "hddn_dim": 128,
        "act_h": "swish",
        "ctxt_in_inpt": False,
        "ctxt_in_hddn": True,
    },
)

# Train model
max_iter = 100_000
num_samples = 1024
optimizer = T.optim.Adam(diff.parameters(), lr=1e-3)
device = T.device("cuda" if T.cuda.is_available() else "cpu")
diff.to(device)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    x_np, c_np = make_moons(num_samples, noise=0)
    x = T.tensor(x_np).float().to(device)
    c = T.tensor(c_np).float().to(device).unsqueeze(1)

    # Compute loss
    loss = diff.get_loss(x, c)
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        # Sample from the moons and random noise
        x_np, c_np = make_moons(10000, noise=0)
        x1 = T.randn(10000, 2).to(device)
        c = T.tensor(c_np).float().to(device).unsqueeze(1)

        # Generate via integration
        t = T.linspace(1, 0, 50).to(device)

        x0_hat = diff.generate(x1, c, t)
        x0_hat = x0_hat.detach().cpu().numpy()

        # Generate a 2D heatmaps
        bins = [np.linspace(-1.5, 2.5, 50), np.linspace(-1, 1.5, 50)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist2d(x_np[:, 0], x_np[:, 1], bins=bins, cmap="viridis")
        axes[0].set_title("True")
        axes[1].hist2d(x0_hat[:, 0], x0_hat[:, 1], bins=bins, cmap="viridis")
        axes[1].set_title("Sampled")
        plt.savefig("True.png")
        plt.close("all")
