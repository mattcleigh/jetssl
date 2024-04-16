"""My kmeans is much slower than the new library.
So for now I am no longer using any of the code in this file.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as T
from torch import nn

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


class Labeller(nn.Module, ABC):
    """Base class for labellers."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, **kwargs) -> T.Tensor:
        pass

    @abstractmethod
    def probs_to_code(self, **kwargs) -> T.Tensor:
        pass

    @abstractmethod
    def idx_to_code(self, **kwargs) -> T.Tensor:
        pass


class KMeansLabeller(Labeller):
    """Fit an unconditional kmeans to the input representation."""

    def __init__(self, inpt_dim: int, num_labels: int, device: str = "cpu") -> None:
        super().__init__()
        self.inpt_dim = inpt_dim
        self.num_labels = num_labels

        # Register the cluster centers as a buffer
        self.register_buffer(
            "cluster_centers",
            T.zeros((num_labels, inpt_dim), device=device),
        )
        self.register_buffer("initialised", T.zeros(1, device=device))

    @property
    def device(self) -> str:
        return self.cluster_centers.device

    @T.no_grad()
    def fit(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        sel_inpt = apply_mask(inpt, mask).flatten(start_dim=1)
        cluster_centers = kmeans(sel_inpt, self.num_labels)
        self.cluster_centers.data.copy_(cluster_centers)

    @T.no_grad()
    def forward(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        sel_inpt = apply_mask(inpt, mask)
        labels = T.argmin(T.cdist(sel_inpt, self.cluster_centers), dim=-1)
        return apply_unmask(inpt, labels, mask)

    @T.no_grad()
    def probs_to_code(self, probabilities: T.Tensor) -> T.Tensor:
        """Given a distribution over the classes, return the cluster centers."""
        idxes = T.multinomial(probabilities, 1)
        return self.idx_to_code(idxes)

    @T.no_grad()
    def idx_to_code(self, idx: T.Tensor) -> T.Tensor:
        return self.cluster_centers[idx]


class MultiKMeansLabeller(Labeller):
    """Fit multiple kmeans by slicing the inputs."""

    def __init__(
        self,
        inpt_dim: int,
        slices: list | tuple = (),
        labels_per_slice: list | None = None,
    ) -> None:
        super().__init__()
        assert sum(slices) == self.inpt_dim
        assert len(labels_per_slice) == len(slices)

        # Attributes
        self.inpt_dim = inpt_dim
        self.slices = slices
        self.labels_per_slice = labels_per_slice
        self.num_labels = np.prod(labels_per_slice)
        self.register_buffer("initialised", T.zeros(1))

        # Overflows is a useful attribute for converting individual labels
        # To the combined code
        self.overflows = [1, *list(self.labels_per_slice)]

        # For each slice create a separate kmeans sublabeller
        self.sub_labellers = nn.ModuleList([
            KMeansLabeller(inpt_dim=s, num_labels=lab)
            for s, lab in zip(slices, self.labels_per_slice, strict=False)
        ])

    @T.no_grad()
    def fit(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        """Split the input and fit each sublabeller."""
        split_inpt = T.split(inpt, tuple(self.slices), dim=-1)
        for x, labeller in zip(split_inpt, self.sub_labellers, strict=False):
            labeller.fit(x, mask)

    @T.no_grad()
    def forward(
        self, inpt: T.Tensor, mask: T.BoolTensor | None = None, **_kwargs
    ) -> T.Tensor:
        """Build the combined labels from the sublabellers."""
        split_inpt = T.split(inpt, tuple(self.slices), dim=-1)
        combined_labels = T.zeros(inpt.shape[:-1]).to(inpt.device)
        for x, labeller, ov in zip(
            split_inpt, self.sub_labellers, self.overflows, strict=False
        ):
            labels = labeller(x, mask)
            combined_labels += labels * ov
        return labels

    @T.no_grad()
    def probabilities_to_code(self, probabilities: T.Tensor) -> T.Tensor:
        combined_indx = T.multinomial(probabilities, self.n_samples, replacement=True)
        return self.idx_to_code(combined_indx)

    @T.no_grad()
    def idx_to_code(self, idx: T.Tensor) -> T.Tensor:
        codes = []
        for i in range(len(self.slices))[::-1]:
            temp_i = idx // self.overflows[i]
            idx -= temp_i * self.overflows[i]
            codes.insert(0, self.sub_labellers[i].cluster_centers[temp_i])
        return T.cat(codes, dim=-1)
