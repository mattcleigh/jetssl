from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.mlp import MLP

if TYPE_CHECKING:
    from src.models.utils import JetBackbone

import logging

log = logging.getLogger(__name__)


class Tracker(LightningModule):
    """A class for fine tuning a track type classifier on top of a JetBackbone."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        mlp_config: partial,
        optimizer: partial,
        scheduler: dict,
        class_weights: list | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        log.info(f"Loading backbone from {backbone_path}")
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        # Use logistic regression for binary classification
        self.mlp = MLP(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes,
            **mlp_config,
        )

        # Store the class weights as a buffer
        class_weights = class_weights or [1.0] * n_classes
        assert len(class_weights) == n_classes
        self.register_buffer("class_weights", T.tensor(class_weights))

        # Metrics
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.BoolTensor) -> tuple:
        x, mask = self.backbone(csts, csts_id, mask)
        x = x[:, -csts.shape[1] :]  # Trim off registers
        return self.mlp(x)

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        mask = sample["mask"]
        track_type = sample["track_type"]

        # Pass through the backbone and head
        output = self.forward(csts, csts_id, mask)[mask]
        track_type = track_type[mask]

        # Get the loss either by cross entropy or logistic regression
        loss = cross_entropy(
            output, track_type, label_smoothing=0.1, weight=self.class_weights
        )

        # Calculate the accuracy and f1 scores
        acc = getattr(self, f"{prefix}_acc")
        acc(output, track_type)

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
        return {
            "output": output,
            "track_type": sample["track_type"],
            "mask": sample["mask"],
            "labels": sample["labels"].unsqueeze(-1),
        }

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)
