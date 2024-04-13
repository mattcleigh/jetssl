from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched

if TYPE_CHECKING:
    from src.models.utils import JetBackbone


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
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes
            if n_classes > 2
            else 1,  # Logistic regression for binary
        )

        # Loss and metrics
        self.acc = Accuracy(
            "multiclass" if n_classes > 2 else "binary", num_classes=n_classes
        )

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
            target = labels.float().view(output.shape)
            loss = binary_cross_entropy_with_logits(output, target)
            output = T.sigmoid(output)  # For the accuracy metric

        # Update the internal state of the accuracy metric
        self.acc(output, labels)

        # Log the loss and accuracy
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", self.acc)

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

        # Use the cwola labels for the loss
        cwola_target = cwola_labels.float().view(output.shape)
        loss = binary_cross_entropy_with_logits(output, cwola_target)

        # Use the true labels for the accuracy
        true_target = labels.view(output.shape)
        self.acc(T.sigmoid(output), true_target)

        # Log the loss and accuracy
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", self.acc)

        return loss
