from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from torchpq.clustering import KMeans

from mltools.mltools.flows import rqs_flow
from src.models.utils import VectorDiffuser
from src.plotting_new import plot_continuous, plot_labels

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class TaskBase(nn.Module):
    """Base class for all tasks."""

    def __init__(
        self,
        mpm: nn.Module,
        *,  # Force keyword arguments
        name: str,
        weight: float = 1.0,
        detach: bool = False,
        apply_every: int = 1,
    ) -> None:
        super().__init__()
        self.mpm = mpm
        self.name = name
        self.weight = weight
        self.detach = detach
        self.apply_every = apply_every

    def get_loss(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Get the loss for the task."""
        # Return early, always run on validation
        if batch_idx % self.apply_every != 0 and prefix == "train":
            return T.tensor(0.0, requires_grad=True)

        # Calculate the loss, detaching if necessary
        with T.no_grad() if self.detach else nullcontext():
            loss = self._get_loss(data, prefix)

        # Log using the parent
        self.mpm.log(f"{prefix}/{self.name}_loss", loss)

        # Return with the weight
        return self.weight * loss

    @T.no_grad()
    def visualise(self, data: dict) -> None:
        """Visualise the task."""
        Path("plots").mkdir(exist_ok=True)
        self._visualise(deepcopy(data))  # Don't want to modify the original

    def on_fit_start(self) -> None:
        """At the start of the fit, allow to pass without error."""

    def _get_loss(self, data: dict) -> T.Tensor:
        """Get the loss for the task."""
        raise NotImplementedError

    def _visualise(self, data: dict) -> None:
        """Visualise the task, optional."""


class IDTask(TaskBase):
    """Task for predicting the ID of the constituent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.mpm.outp_dim, CSTS_ID)

    def _get_loss(self, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        pred = self.head(data["outputs"][data["null_mask"]])
        target = data["csts_id"][data["null_mask"]]
        return cross_entropy(pred, target, label_smoothing=0.1)

    def _visualise(self, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(data["outputs"][data["null_mask"]])
        pred = T.softmax(pred, dim=-1)
        pred = T.multinomial(pred, 1).squeeze(1)
        plot_labels(data, pred)


class RegTask(TaskBase):
    """Task for regressing the properties of the constituent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.mpm.outp_dim, self.mpm.csts_dim)

    def _get_loss(self, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        pred = self.head(data["outputs"][data["null_mask"]])
        target = data["normed_csts"][data["null_mask"]]
        return (pred - target).abs().mean()

    def _visualise(self, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(data["outputs"][data["null_mask"]])
        pred = self.mpm.normaliser.reverse(pred)
        plot_continuous(data, pred)


class FlowTask(TaskBase):
    """Estimating the density of the constituents using a normalising flow."""

    def __init__(
        self, *args, embed_dim: int = 128, flow_config: dict, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.mpm.outp_dim, embed_dim)
        self.flow = rqs_flow(
            xz_dim=self.mpm.csts_dim, ctxt_dim=embed_dim, **flow_config
        )

    @T.autocast("cuda", enabled=False)  # Autocasting is bad for flows
    @T.autocast("cpu", enabled=False)
    def _get_loss(self, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Unpack the data
        normed_csts = data["normed_csts"]
        null_mask = data["null_mask"]
        outputs = data["outputs"]

        # The flow can't handle discrete targets which unfortunately affects the
        # impact paramters. Even for charged particles, there are discrete values
        # particularly in d0_err and dz_err. So we will add a tiny bit of noise.
        # At this stage these variables should be normalised, so hopefully adding a
        # little extra noise won't hurt.
        normed_csts[..., -4:] += 0.05 * T.randn_like(normed_csts[..., -4:])

        # Calculate the conditional likelihood under the flow
        inpt = normed_csts[null_mask].float()
        ctxt = self.head(outputs[null_mask]).float()
        return self.flow.forward_kld(inpt, context=ctxt)

    def _visualise(self, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        ctxt = self.head(data["outputs"][data["null_mask"]])
        pred = self.flow.sample(ctxt.shape[0], context=ctxt)[0]
        pred = self.mpm.normaliser.reverse(pred)
        plot_continuous(data, pred)


class KmeansTask(TaskBase):
    """Task for modelling the properties of the constituent using kmeans clustering."""

    def __init__(self, *args, kmeans_config: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kmeans = KMeans(**kmeans_config)
        self.head = nn.Linear(self.mpm.outp_dim, self.kmeans.n_clusters)

    def _get_loss(self, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Get the target using the kmeans and the original csts
        target = data["normed_csts"][data["null_mask"]].T.contiguous()
        target = self.kmeans.predict(target).long()

        # Get the predictions and calculate the loss
        pred = self.head(data["outputs"][data["null_mask"]])
        return cross_entropy(pred, target, label_smoothing=0.1)

    def _visualise(self, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(data["outputs"][data["null_mask"]])
        pred = T.softmax(pred, dim=-1)
        pred = T.multinomial(pred, 1).squeeze(1)
        pred = self.kmeans.centroids.index_select(1, pred).T
        pred = self.normaliser.reverse(pred)
        plot_continuous(data, pred)

    def on_fit_start(self) -> None:
        """At the start of the fit, fit the kmeans."""
        # Skip kmeans has already been initialised (e.g. from a checkpoint)
        if self.kmeans.centroids is not None:
            return

        # Load the first 50 batches of training data
        csts = []
        mask = []
        loader = self.mpm.trainer.datamodule.train_dataloader()
        for i, data in enumerate(loader):
            csts.append(data["csts"])
            mask.append(data["mask"])
            if i > 50:
                break
        csts = T.vstack(csts).to(self.mpm.device)
        mask = T.vstack(mask).to(self.mpm.device)

        # Fit the normaliser such that the labeller uses scaled inputs
        self.mpm.normaliser.fit(csts, mask, freeze=True)  # 50 batches should be enough
        csts = self.mpm.normaliser(csts, mask)  # Pass through the normaliser
        inputs = csts[mask].T.contiguous()
        self.kmeans.fit(inputs)


class DiffTask(TaskBase):
    """Use conditional diffusion to model the properties of the constituent."""

    def __init__(self, *args, embed_dim: int, diff_config: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.mpm.outp_dim, embed_dim)
        self.diff = VectorDiffuser(
            inpt_dim=self.mpm.csts_dim, ctxt_dim=embed_dim, **diff_config
        )

    def _get_loss(self, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        ctxt = self.head(data["outputs"][data["null_mask"]])
        target = data["normed_csts"][data["null_mask"]]
        return self.diff.get_loss(target, ctxt)

    def _visualise(self, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        ctxt = self.head(data["outputs"][data["null_mask"]])
        x1 = T.randn((ctxt.shape[0], self.mpm.csts_dim), device=self.device)
        times = T.linspace(1, 0, 100, device=ctxt.device)
        pred = self.diff.generate(x1, ctxt, times)
        pred = self.normaliser.reverse(pred)
        plot_continuous(data, pred)


class ProbeTask(TaskBase):
    """Classify the jet using the full outputs and the labels."""

    def __init__(self, *args, class_head: partial, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head = class_head(inpt_dim=self.mpm.outp_dim, outp_dim=self.mpm.n_classes)
        self.acc = Accuracy("multiclass", num_classes=self.mpm.n_classes)

    def _get_loss(self, data: dict, prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Unpack the data
        normed_csts = data["normed_csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        labels = data["labels"]

        # We must pass the full outputs through the backbone (no masking)
        full, full_mask = self.mpm.full_encode(normed_csts, csts_id, mask)
        preds = self.head(full, mask=full_mask)
        loss = cross_entropy(preds, labels)

        # Update and log the accuracy
        self.acc(preds, labels)
        self.mpm.log(f"{prefix}/{self.name}_accuracy", loss)

        return loss
