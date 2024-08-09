import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

fig, axes = plt.subplots(2, 1, figsize=(6, 4))
depth_values = [1, 2, 3, 4]
depth_acc = (
    np.array([
        0.8295,
        0.8304,
        0.8307,
        0.8323,
    ])
    * 100
)
axes[0].plot(depth_values, depth_acc, "ro-")
for x, y in zip(depth_values, depth_acc):
    axes[0].text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom")
axes[0].set_xlabel("decoder depth")
# axes[0].set_ylabel("probe accuracy (%)")
axes[0].set_ylim(82.8, 83.4)

mask_values = np.linspace(10, 90, 9)
mask_acc = (
    np.array([
        0.8192,
        0.8284,
        0.8304,
        0.8316,
        0.8308,
        0.8290,
        0.8252,
        0.8184,
        0.8017,
    ])
    * 100
)
axes[1].plot(mask_values, mask_acc, "ro-")
for x, y in zip(mask_values, mask_acc):
    axes[1].text(
        x,
        y + 0.1,
        f"{y:.2f}",
        ha="center",
        va="bottom",
    )
axes[1].set_xlabel("masking rate (%)")
# axes[1].set_ylabel("probe accuracy (%)")
axes[1].set_ylim(80, 84)
axes[1].yaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
fig.text(0.06, 0.5, "accuracy (%)", ha="center", va="center", rotation="vertical")
for ax in axes:
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.subplots_adjust(left=0.15)
fig.savefig("plots/final/sweep.pdf")
plt.close(fig)
