import numpy as np
import rootutils
import torch as T
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.models.utils import VectorDiffuser

diff = VectorDiffuser(
    inpt_dim=2,
    ctxt_dim=1,
    time_dim=16,
    mlp_config={
        "num_blocks": 3,
        "hddn_dim": 128,
        "act_h": "SiLU",
        "norm": "LayerNorm",
        "do_res": True,
        "init_zeros": True,
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
    x_np, c_np = make_moons(num_samples, noise=0.05)
    x = T.tensor(x_np).float().to(device)
    c = T.tensor(c_np).float().to(device).unsqueeze(1)

    # Compute loss
    loss = diff.get_loss(x, c)
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        with T.no_grad():
            # Sample from the moons and random noise
            x_np, c_np = make_moons(10000, noise=0.05)
            x1 = T.randn(10000, 2).to(device)
            c = T.tensor(c_np).float().to(device).unsqueeze(1)

            # Generate via integration
            t = T.linspace(1, 0, 100).to(device)

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
