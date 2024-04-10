from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy
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

        # Load the pretrained and pickled JetBackbone object.
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes,
        )

        # Loss and metrics
        self.acc = Accuracy("multiclass", num_classes=n_classes)

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
        loss = cross_entropy(output, labels, label_smoothing=0.1)
        self.acc(output, labels)  # updates internal state

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

        # Train on cwola labels but evaluate on the true labels
        output = self.forward(csts, csts_id, mask)
        loss = cross_entropy(output, cwola_labels)  # No smoothing for CWoLa
        true_loss = cross_entropy(output, labels)
        acc = self.acc(output, labels)

        # Log the loss and accuracy
        self.log(f"{prefix}/total_loss", true_loss)  # Used for early stopping
        self.log(f"{prefix}/cwola_loss", loss)
        self.log(f"{prefix}/acc", acc)

        return loss
