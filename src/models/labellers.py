from functools import partial

import numpy as np
import torch as T
from pytorch_lightning import LightningModule
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


class Labeller(LightningModule):
    def __init__(self, *, inpt_dim: int, num_labels: int = 512, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.inpt_dim = inpt_dim
        self.num_labels = num_labels

    def fit(self: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        raise NotImplementedError


class KMeansLabeller(Labeller):
    """Fit an unconditional kmeans to the input representation."""

    def __init__(self, n_samples: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer(
            "cluster_centers",
            T.zeros((self.num_labels, self.inpt_dim), device=self.device),
        )
        self.register_buffer("initialised", T.zeros(1))
        self.n_samples = n_samples

    @T.no_grad()
    def fit(self, inpt: T.Tensor, mask: T.BoolTensor | None = None) -> None:
        inpt = apply_mask(inpt, mask).flatten(start_dim=1)  # Apply mask and flatten
        cluster_centers = kmeans(inpt, self.num_labels)
        self.cluster_centers = cluster_centers.to(self.device)

    @T.no_grad()
    def forward(
        self, inpt: T.Tensor, mask: T.BoolTensor | None = None, **_kwargs
    ) -> T.Tensor:
        sel_inpt = apply_mask(inpt, mask)
        labels = T.argmin(T.cdist(sel_inpt, self.cluster_centers), dim=-1)
        return apply_unmask(inpt, labels, mask)

    @T.no_grad()
    def probabilities_to_code(self, probabilities: T.Tensor) -> T.Tensor:
        """Given a distribution over the classes, return the cluster centers."""
        idxes = T.multinomial(probabilities, self.n_samples, replacement=True)
        return self.cluster_centers[idxes]

    @T.no_grad()
    def idx_to_code(self, idx: T.Tensor) -> T.Tensor:
        return self.cluster_centers[idx]


class MultiKMeansLabeller(Labeller):
    """Fit multiple kmeans by slicing the inputs."""

    def __init__(
        self,
        n_samples: int = 5,
        slices: list | tuple = (),
        labels_per_slice: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.slices = slices
        self.register_buffer("initialised", T.zeros(1))

        # Make sure the slices add up to the input dimension
        assert sum(slices) == self.inpt_dim

        # Get the number of labels used by each slice
        if labels_per_slice is not None:
            self.labels_per_slice = labels_per_slice
        else:
            labels_per_dim = round(self.num_labels ** (1 / self.inpt_dim))
            self.labels_per_slice = [labels_per_dim**s for s in slices]
        if np.prod(self.labels_per_slice) != self.num_labels:
            raise ValueError(
                f"Labels per slice: {self.labels_per_slice} does not match ",
                f"the total number of labels: {self.num_labels}",
            )

        # Overflows is used to fold and unfold the labels
        self.overflows = [1, *list(self.labels_per_slice)]

        # For each slice create a separate kmeans sublabeller
        self.sub_labellers = nn.ModuleList([
            KMeansLabeller(inpt_dim=s, num_labels=lab, n_samples=n_samples)
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


class ContinuousClassLabeller(Labeller):
    """Use a combination of a labeller on continous inputs and the data class."""

    def __init__(
        self,
        class_dim: int = 1,
        continuous_labeller: partial = KMeansLabeller,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.class_dim = class_dim
        self.continuous_labeller = continuous_labeller(
            inpt_dim=self.inpt_dim,
            num_labels=self.num_labels,
        )
        self.n_samples = self.continuous_labeller.n_samples

        # Update the number of labels as we broadcast with the class labels
        self.num_labels *= class_dim

    @T.no_grad()
    def fit(self, *args, **kwargs) -> None:
        self.continuous_labeller.fit(*args, **kwargs)

    @T.no_grad()
    def forward(
        self,
        inpt: T.Tensor,
        mask: T.BoolTensor | None = None,
        inpt_label: T.Tensor | None = None,
    ) -> T.Tensor:
        # Get the continuous labels from the labeller
        labels = self.continuous_labeller(inpt, mask)

        # Combine the two labels
        return labels + inpt_label * self.continuous_labeller.num_labels

    @T.no_grad()
    def probabilities_to_code(self, probabilities: T.Tensor) -> tuple:
        combined_indx = T.multinomial(probabilities, self.n_samples, replacement=True)
        return self.idx_to_code(combined_indx)

    @T.no_grad()
    def idx_to_code(self, idx: T.Tensor) -> T.Tensor:
        # Use the remainder to get the continuous label
        cont_indx = idx % self.continuous_labeller.num_labels
        cont_code = self.continuous_labeller.idx_to_code(cont_indx)

        # use the divisor to get the class index
        class_indx = idx // self.continuous_labeller.num_labels

        # Combine the codes
        return cont_code, class_indx
