import torch as T
from pytorch_lightning import LightningModule

from mltools.mltools.clustering import kmeans


def apply_mask(inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
    """Optional mask to apply to the input."""
    if mask is None:
        return inpt
    return inpt[mask]


def apply_unmask(
    inpt: T.Tensor, labels: T.Tensor, mask: T.BoolTensor | None = None
) -> T.Tensor:
    """Optionally pad the labels with zeros to match the input shape."""
    if mask is None:
        return labels
    padded_labels = T.zeros(inpt.shape[:-1], device=labels.device, dtype=labels.dtype)
    padded_labels[mask] = labels
    return padded_labels


class Labeller(LightningModule):
    def __init__(self, *, inpt_dim: int, num_labels: int = 512, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.inpt_dim = inpt_dim
        self.num_labels = num_labels

    def fit(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        raise NotImplementedError


class KMeansLabeller(Labeller):
    """Fit an unconditional kmeans to the input representation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "cluster_centers",
            T.zeros((self.num_labels, self.inpt_dim), device=self.device),
        )
        self.register_buffer("initialised", T.zeros(1))

    @T.no_grad()
    def fit(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        inpt = apply_mask(inpt, mask).flatten(start_dim=1)  # Apply mask and flatten
        cluster_centers = kmeans(inpt, self.num_labels)
        self.cluster_centers = cluster_centers.to(self.device)

    @T.no_grad()
    def forward(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        sel_inpt = apply_mask(inpt, mask)
        labels = T.argmin(T.cdist(sel_inpt, self.cluster_centers), dim=-1)
        return apply_unmask(inpt, labels, mask)

    @T.no_grad()
    def probabilities_to_code(self, probabilities: T.Tensor) -> T.Tensor:
        """Given a distribution over the classes, return the cluster centers."""
        idxes = T.multinomial(probabilities, 1, replacement=True)
        return self.cluster_centers[idxes]

    @T.no_grad()
    def idx_to_code(self, idx: T.Tensor) -> T.Tensor:
        return self.cluster_centers[idx]
