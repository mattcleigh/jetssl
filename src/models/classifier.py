from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.loss import bce_with_label_smoothing
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.transformers import Transformer

if TYPE_CHECKING:
    from src.models.utils import JetBackbone

import logging

log = logging.getLogger(__name__)

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class Classifier(LightningModule):
    """A class for fine tuning a classifier based on a model with an encoder.

    This should be paired with a scheduler for unfreezing/freezing the backbone.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        class_head: partial,
        optimizer: partial,
        scheduler: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        log.info(f"Loading backbone from {backbone_path}")
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        # Use logistic regression for binary classification
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes if n_classes > 2 else 1,
        )

        # Metrics
        task = "multiclass" if n_classes > 2 else "binary"
        self.train_acc = Accuracy(task, num_classes=n_classes)
        self.valid_acc = Accuracy(task, num_classes=n_classes)

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.BoolTensor) -> tuple:
        x, mask = self.backbone(csts, csts_id, mask)
        return self.class_head(x, mask=mask)

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]

        # Pass through the backbone and head
        output = self.forward(csts, csts_id, mask)

        # Get the loss either by cross entropy or logistic regression
        if self.n_classes > 2:
            loss = cross_entropy(output, labels, label_smoothing=0.1)
        else:
            target = labels.float().view_as(output)
            loss = bce_with_label_smoothing(output, target)
            output = T.sigmoid(output)  # For the accuracy metric

        # Calculate the accuracy
        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)

        # Log the loss and accuracy
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", acc)

        return loss

    def training_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "train")

    def validation_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "valid")

    def predict_step(self, sample: tuple) -> T.Tensor:
        output = self.forward(sample["csts"], sample["csts_id"], sample["mask"])
        return {"output": output, "label": sample["labels"].unsqueeze(-1)}

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)


class OnlyHeadClass(LightningModule):
    """A classifier without a backbone, only a head.

    Useful for testing the head on its own and when the backbone is not needed.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        class_head: partial,
        backbone_path: str,
        optimizer: partial,
        scheduler: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Save the data sample information
        self.n_classes = n_classes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # Delete the backbone string (not needed and linter complains)
        del backbone_path

        # Create the head for the downstream task
        self.class_head = class_head(
            inpt_dim=128,
            outp_dim=n_classes
            if n_classes > 2
            else 1,  # Logistic regression for binary
        )

        # Still need the embedding networks
        self.csts_id_embedder = nn.Embedding(CSTS_ID, self.class_head.dim)
        self.ctst_embedder = nn.Linear(self.csts_dim, self.class_head.dim)

        # Loss and metrics
        task = "multiclass" if n_classes > 2 else "binary"
        self.trian_acc = Accuracy(task, num_classes=n_classes)
        self.valid_acc = Accuracy(task, num_classes=n_classes)

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.BoolTensor) -> tuple:
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)
        return self.class_head(x, mask=mask)

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]

        # Pass through the backbone and head
        output = self.forward(csts, csts_id, mask)

        # Get the loss either by cross entropy or logistic regression
        if self.n_classes > 2:
            loss = cross_entropy(output, labels, label_smoothing=0.1)
        else:
            target = labels.float().view(output.shape)
            loss = binary_cross_entropy_with_logits(output, target)
            output = T.sigmoid(output)  # For the accuracy metric

        # Update the accuracy
        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)

        # Log
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", acc)

        return loss

    def training_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "train")

    def validation_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "valid")

    def predict_step(self, sample: tuple) -> T.Tensor:
        output = self.forward(sample["csts"], sample["csts_id"], sample["mask"])
        return {"output": output, "label": sample["labels"].unsqueeze(-1)}

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)


class CWoLaClassifier(Classifier):
    """Extra classifier for the CWoLa task."""

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        cwola_labels = sample["cwola_labels"]

        # Pass through the network
        output = self.forward(csts, csts_id, mask)

        # Use the cwola labels for the loss with label smoothing
        cwola_target = cwola_labels.float().view(output.shape)
        loss = bce_with_label_smoothing(output, cwola_target)

        # Use the true labels for the accuracy
        true_target = labels.view(output.shape)
        acc = getattr(self, f"{prefix}_acc")
        acc(T.sigmoid(output), true_target)

        # Log
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", acc)

        return loss


class TopTagger(LightningModule):
    """Classifier with an extra embedding layer for toptagging.

    This should be paired with a scheduler for unfreezing/freezing the backbone.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        class_head: partial,
        optimizer: partial,
        scheduler: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")
        self.backbone.csts_id_emb.requires_grad_(False)  # Freeze the csts_id embedding

        # Single transformer layer for the extra variables
        self.embedder = Transformer(
            inpt_dim=3,
            num_layers=1,
            outp_dim=self.backbone.dim + 4,
            dim=128,
            do_input_linear=True,
            do_output_linear=True,
        )

        # Create the head for the downstream task
        # Use logistic regression for binary classification
        self.jet_normaliser = IterativeNormLayer(5)
        self.jet_embedder = nn.Linear(5, 16)
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=1,
            ctxt_dim=16,
        )

        # Loss and metrics
        self.train_acc = Accuracy("binary")
        self.valid_acc = Accuracy("binary")

    def forward(self, csts: T.Tensor, mask: T.BoolTensor, jets: T.Tensor) -> tuple:
        # Embed the extra variables
        csts = csts[..., :3]
        emb_1, csts_id = self.embedder(csts, mask=mask).split(
            [4, self.backbone.dim], dim=-1
        )
        csts = T.cat([csts, emb_1], dim=-1)

        # Pass through the backbone but without the csts_id embedding
        x = self.backbone.csts_emb(csts) + csts_id
        x = self.backbone.encoder(x, mask=mask)
        mask = self.backbone.encoder.get_combined_mask(mask)

        # Pass through the head using the jet kinematics as context
        ctxt = self.jet_embedder(self.jet_normaliser(jets))
        return self.class_head(x, mask=mask, ctxt=ctxt)

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        labels = sample["labels"]
        mask = sample["mask"]
        jets = sample["jets"]

        # Pass through the backbone and head
        output = self.forward(csts, mask, jets)

        # Get the loss by logistic regression with label smoothing
        loss = bce_with_label_smoothing(output, labels.float())
        output = T.sigmoid(output).squeeze()  # For the accuracy metric

        # Update the internal state of the accuracy metric
        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)

        # Log the loss and accuracy
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", acc)

        return loss

    def training_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "train")

    def validation_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "valid")

    def predict_step(self, sample: tuple) -> T.Tensor:
        output = self.forward(sample["csts"], sample["mask"])[0]
        return {"output": output, "label": sample["labels"].unsqueeze(-1)}

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)
