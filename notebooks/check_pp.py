import matplotlib.pyplot as plt
import torch as T
from torch.nn import functional as F

# Generate some random points
x = T.rand(1000, 32)
x = F.normalize(x, dim=-1)


# Calculate the columb force between each pair of points
def update(x):
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    f_hat = F.normalize(diff, dim=-1)
    f = 1 / (diff.abs() + 1).square()
    f_net = (f * f_hat).sum(dim=1)
    x -= f_net
    return F.normalize(x, dim=-1)


for _i in range(20):
    x = update(x)
    x_np = x.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*x_np.T[:3])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    plt.tight_layout()
    ax.set_box_aspect([1, 1, 1])
    plt.savefig("random_points.png")
    plt.close()
