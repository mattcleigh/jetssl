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
from src.plotting import plot_continuous, plot_labels

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class TaskBase(nn.Module):
    """Base class for all tasks."""

    def __init__(
        self,
        *,  # Force keyword arguments
        name: str,
        input_dim: int = 0,
        weight: float = 1.0,
        detach: bool = False,
        apply_every: int = 1,
        id_conditional: bool = False,
        id_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.weight = weight
        self.detach = detach
        self.apply_every = apply_every
        self.id_conditional = id_conditional
        self.id_embed_dim = id_embed_dim

        # For conditional tasks we add the csts_id to the input
        if id_conditional:
            self.id_one_hot = nn.Embedding(CSTS_ID, self.id_embed_dim)
            self.input_dim += self.id_embed_dim

    def get_loss(
        self, parent: nn.Module, data: dict, batch_idx: int, prefix: str
    ) -> T.Tensor:
        """Get the loss for the task."""
        # Return early, always run on validation
        if batch_idx % self.apply_every != 0 and prefix == "train":
            return T.tensor(0.0, requires_grad=True)

        # Calculate the loss, detaching if necessary
        if self.detach:
            new_dict = {k: v.detach() for k, v in data.items()}  # Detach doesnt copy
            loss = self._get_loss(parent, new_dict, prefix)
        else:
            loss = self._get_loss(parent, data, prefix)

        # Log using the parent
        parent.log(f"{prefix}/{self.name}_loss", loss)

        # Return with the weight
        return self.weight * loss

    def get_head_input(self, data: dict) -> T.Tensor:
        """Get the inputs for the task head.

        Only works if the head acts on the flattened outputs of the model.

        If applicable combines it with the true ID of the constituent.
        """
        outputs = data["outputs"][data["null_mask"]]
        if self.id_conditional:
            true_id = data["csts_id"][data["null_mask"]]
            true_id = self.id_one_hot(true_id)
            return T.concat([outputs, true_id], dim=-1)
        return outputs

    @T.no_grad()
    def visualise(self, parent: nn.Module, data: dict) -> None:
        """Visualise the task."""
        Path("plots").mkdir(exist_ok=True)
        self._visualise(parent, deepcopy(data))  # Don't want to modify the original

    def on_fit_start(self, parent: nn.Module) -> None:
        """At the start of the fit, allow to pass without error."""

    def _get_loss(self, parent: nn.Module, data: dict) -> T.Tensor:
        """Get the loss for the task."""
        raise NotImplementedError

    def _visualise(self, parent: nn.Module, data: dict) -> None:
        """Visualise the task, optional."""


class IDTask(TaskBase):
    """Task for predicting the ID of the constituent."""

    def __init__(self, parent: nn.Module, **kwargs) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.head = nn.Linear(self.input_dim, CSTS_ID)

    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        pred = self.head(self.get_head_input(data))
        target = data["csts_id"][data["null_mask"]]
        return cross_entropy(pred, target, label_smoothing=0.1)

    def _visualise(self, parent: nn.Module, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(self.get_head_input(data))
        pred = T.softmax(pred, dim=-1)
        pred = T.multinomial(pred, 1).squeeze(1)
        plot_labels(data, pred)


class RegTask(TaskBase):
    """Task for regressing the properties of the constituent."""

    def __init__(self, parent: nn.Module, **kwargs) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.head = nn.Linear(self.input_dim, parent.csts_dim)

    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        pred = self.head(self.get_head_input(data))
        target = data["csts"][data["null_mask"]]
        return (pred - target).abs().mean()

    def _visualise(self, parent: nn.Module, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(self.get_head_input(data))
        plot_continuous(data, pred)


class FlowTask(TaskBase):
    """Estimating the density of the constituents using a normalising flow."""

    def __init__(
        self, parent: nn.Module, embed_dim: int, flow_config: dict, **kwargs
    ) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.head = nn.Linear(self.input_dim, embed_dim)
        self.flow = rqs_flow(xz_dim=parent.csts_dim, ctxt_dim=embed_dim, **flow_config)

    @T.autocast("cuda", enabled=False)  # Autocasting is bad for flows
    @T.autocast("cpu", enabled=False)
    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Unpack the data
        csts = data["csts"]
        null_mask = data["null_mask"]

        # The flow can't handle discrete targets which unfortunately affects the
        # impact paramters. Even for charged particles, there are discrete values
        # particularly in d0_err and dz_err. So we will add a tiny bit of noise.
        # At this stage these variables should be normalised, so hopefully adding a
        # little extra noise won't hurt.
        # As this is an inplace operation, we need to clone the tensor
        csts = csts.clone()
        csts[..., -4:] += 0.05 * T.randn_like(csts[..., -4:])

        # Calculate the conditional likelihood under the flow
        inpt = csts[null_mask].float()
        ctxt = self.head(self.get_head_input(data)).float()
        return self.flow.forward_kld(inpt, context=ctxt)

    def _visualise(self, parent: nn.Module, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        ctxt = self.head(self.get_head_input(data))
        pred = self.flow.sample(ctxt.shape[0], context=ctxt)[0]
        plot_continuous(data, pred)


class KmeansTask(TaskBase):
    """Task for modelling the properties of the constituent using kmeans clustering."""

    def __init__(self, parent: nn.Module, kmeans_config: dict, **kwargs) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.kmeans = KMeans(**kmeans_config)
        self.head = nn.Linear(self.input_dim, self.kmeans.n_clusters)

        # Populate the centroids or they wont be caputured in the state_dict
        self.kmeans.centroids = T.zeros((parent.csts_dim, self.kmeans.n_clusters))

    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Get the target using the kmeans and the original csts
        target = data["csts"][data["null_mask"]].T.contiguous()
        target = self.kmeans.predict(target).long()

        # Get the predictions and calculate the loss
        pred = self.head(self.get_head_input(data))
        return cross_entropy(pred, target, label_smoothing=0.1)

    def _visualise(self, parent: nn.Module, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        pred = self.head(self.get_head_input(data))
        pred = T.softmax(pred, dim=-1)
        pred = T.multinomial(pred, 1).squeeze(1)
        pred = self.kmeans.centroids.index_select(1, pred).T
        plot_continuous(data, pred)

    def on_fit_start(self, parent: nn.Module) -> None:
        """At the start of the fit, fit the kmeans."""
        # Skip kmeans has already been initialised (e.g. from a checkpoint)
        if self.kmeans.centroids is not None and (self.kmeans.centroids > 0).any():
            return

        # Load the first 50 batches of training data
        csts = []
        mask = []
        loader = parent.trainer.train_dataloader
        if loader is None:
            parent.trainer.fit_loop.setup_data()
            loader = parent.trainer.train_dataloader
        for i, data in enumerate(loader):
            csts.append(data["csts"])
            mask.append(data["mask"])
            if i > 50:
                break
        csts = T.vstack(csts).to(parent.device)
        mask = T.vstack(mask).to(parent.device)

        # Fit the kmeans
        inputs = csts[mask].T.contiguous()
        self.kmeans.fit(inputs)


class VQVAETask(TaskBase):
    """Task for distilation using a pretrained vq-vae."""

    def __init__(self, parent: nn.Module, vae_path: str, **kwargs) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.vae = T.load(vae_path, map_location=parent.device)
        self.head = nn.Linear(self.input_dim, self.vae.quantizer.codebook_size)

        # Make sure that the vae is not trainable
        self.vae.requires_grad_(False)
        self.vae.eval()

    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # Get the target using the vae and the original csts
        target = self.vae(data["csts"], data["null_mask"])[0].long()

        # Get the predictions and calculate the loss
        pred = self.head(self.get_head_input(data))
        return cross_entropy(pred, target, label_smoothing=0.05)


class DiffTask(TaskBase):
    """Use conditional diffusion to model the properties of the constituent."""

    def __init__(
        self, parent: nn.Module, embed_dim: int, diff_config: dict, **kwargs
    ) -> None:
        super().__init__(input_dim=parent.outp_dim, **kwargs)
        self.head = nn.Linear(self.input_dim, embed_dim)
        self.diff = VectorDiffuser(
            inpt_dim=parent.csts_dim, ctxt_dim=embed_dim, **diff_config
        )

    def _get_loss(self, parent: nn.Module, data: dict, _prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        ctxt = self.head(self.get_head_input(data))
        target = data["csts"][data["null_mask"]]
        return self.diff.get_loss(target, ctxt)

    def _visualise(self, parent: nn.Module, data: dict) -> dict:
        """Sample and plot the outputs of the head."""
        ctxt = self.head(self.get_head_input(data))
        x1 = T.randn((ctxt.shape[0], parent.csts_dim), device=ctxt.device)
        times = T.linspace(1, 0, 50, device=ctxt.device)
        pred = self.diff.generate(x1, ctxt, times)
        plot_continuous(data, pred)


class ProbeTask(TaskBase):
    """Classify the jet using the full outputs and the labels."""

    def __init__(self, parent: nn.Module, class_head: partial, **kwargs) -> None:
        super().__init__(input_dim=parent.encoder.dim, **kwargs)
        self.head = class_head(inpt_dim=parent.encoder.dim, outp_dim=parent.n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=parent.n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=parent.n_classes)

    def _get_loss(self, parent: nn.Module, data: dict, prefix: str) -> T.Tensor:
        """Get the loss for this task."""
        # We must pass the full outputs through the backbone (no masking)
        full, full_mask = parent.full_encode(data)
        preds = self.head(full, mask=full_mask)
        loss = cross_entropy(preds, data["labels"])

        # Update and log the accuracy
        acc = getattr(self, f"{prefix}_acc")
        acc(preds, data["labels"])
        parent.log(f"{prefix}/{self.name}_accuracy", acc)

        return loss
