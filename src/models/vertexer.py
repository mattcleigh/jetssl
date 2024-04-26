from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import Accuracy, F1Score

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.mlp import MLP

if TYPE_CHECKING:
    from src.models.utils import JetBackbone


class Vertexer(LightningModule):
    """A class for fine tuning a vertex finder based on a model with an encoder.

    This should be paired with a scheduler for unfreezing/freezing the backbone.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        vertex_config: dict,
        optimizer: partial,
        scheduler: dict,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.n_classes = n_classes
        self.pos_weight = T.tensor(pos_weight)

        # Load the pretrained and pickled JetBackbone object.
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")
        self.outp_dim = self.backbone.outp_dim

        # Create the head for the vertexing task
        self.vertex_head = MLP(
            inpt_dim=self.outp_dim, outp_dim=self.outp_dim, **vertex_config
        )

        # Loss and metrics
        self.accs = nn.ModuleList([Accuracy("binary") for _ in range(n_classes + 1)])
        self.f1s = nn.ModuleList([F1Score("binary") for _ in range(n_classes + 1)])

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.BoolTensor) -> tuple:
        x, _ = self.backbone(csts, csts_id, mask)
        x = x[:, -csts.shape[1] :]  # Trim off registers
        x = self.vertex_head(x)  # Pass through the head

        # Return the scaled dot product of all the pairs
        return x @ x.transpose(-1, -2) * (x.shape[-1] ** -0.5)

    def _shared_step(self, sample: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        vtx_id = sample["vtx_id"]

        # Calculate the mask for which edges are needed
        # Note we only need the upper triangle of the matrix (symmetric)
        vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        vtx_mask = T.triu(vtx_mask, diagonal=1)  # No diagonal = self edges

        # Calculate the target based on if the vtx id matches
        target = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)
        target = target[vtx_mask].float()

        # Pass through the backbone and head
        output = self.forward(csts, csts_id, mask)[vtx_mask].squeeze(-1)

        # Calculate the loss and accuracy
        loss = binary_cross_entropy_with_logits(
            output, target, pos_weight=self.pos_weight
        )

        # Calculate the metrics (last index is for all)
        self.accs[-1](output, target)
        self.f1s[-1](output, target)

        # Log the loss
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/acc", self.accs[-1])
        self.log(f"{prefix}/f1", self.f1s[-1])

        # Log the metrics per jet type
        exp_labels = labels[..., None, None].expand_as(vtx_mask)[vtx_mask]
        for i in range(self.n_classes):
            class_mask = exp_labels == i
            sel_out = output[class_mask]
            sel_tar = target[class_mask]
            self.accs[i](sel_out, sel_tar)
            self.f1s[i](sel_out, sel_tar)
            self.log(f"{prefix}/acc_{i}", self.accs[i])
            self.log(f"{prefix}/f1_{i}", self.f1s[i])

        return loss

    def training_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "train")

    def validation_step(self, sample: tuple) -> T.Tensor:
        return self._shared_step(sample, "valid")

    def predict_step(self, sample: tuple) -> T.Tensor:
        output = self.forward(sample["csts"], sample["csts_id"], sample["mask"])
        return {
            "output": output.squeeze(-1),
            "mask": sample["mask"],
            "vtx_id": sample["vtx_id"],
            "labels": sample["labels"].unsqueeze(-1),
        }

    def configure_optimizers(self) -> dict:
        return simple_optim_sched(self)
