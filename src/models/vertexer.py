from functools import partial
from typing import TYPE_CHECKING

import torch as T
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.mlp import MLP

if TYPE_CHECKING:
    from src.models.utils import JetBackbone

BELL_NUMBERS = [
    1,
    1,
    2,
    5,
    15,
    52,
    203,
    877,
    4140,
    21147,
    115975,
    678570,
    4213597,
    27644437,
    190899322,
    1382958545,
]


def get_ari(mask: T.Tensor, vtx_id: T.Tensor, output: T.Tensor):
    """Calculate the expectation value for randomly assigned vertex ids."""
    # First get the number particles and edges in the batch
    n = mask.sum(-1)
    n_edges = n * (n - 1) / 2
    vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    vtx_mask = T.triu(vtx_mask, diagonal=1)
    targets = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)

    # Calculate the RI score: number of correct edges / number of edges
    preds = output > 0
    correct = (preds == targets) & vtx_mask
    ri = correct.sum(-1).sum(-1) / n_edges

    # Calculate the bell numbers for the batch
    bell_numbers = T.tensor(BELL_NUMBERS, device=mask.device)
    Bn = bell_numbers[n]
    Bnm1 = bell_numbers[n - 1]

    # Make the vtx_id of padded nodes -1 so they cant be counted
    vtx_id[~mask] = -1

    # Calculate the sum over the vertices using the number of tracks per vertex
    g = [(vtx_id == i).sum(-1) for i in range(7)]  # tracks per vertex
    g = [v * (v - 1) / 2 for v in g]  # n choose 2
    g = T.stack(g, dim=-1).sum(-1)  # sum over vertices

    # Get the expectation value
    b_ratio = Bnm1 / Bn
    g_ratio = g / (n * (n - 1) / 2)
    exp = b_ratio * g_ratio + (1 - b_ratio) * (1 - g_ratio)

    # Return the ARI
    return (ri - exp) / (1 - exp)


def get_perf(mask: T.Tensor, vtx_id: T.Tensor, output: T.Tensor):
    vtx_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    vtx_mask = T.triu(vtx_mask, diagonal=1)
    target = vtx_id.unsqueeze(1) == vtx_id.unsqueeze(2)
    target = target & vtx_mask
    output = (output > 0) & vtx_mask
    return (target == output).all((-1, -2)).float().mean()


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
        loss_fn: partial,
        optimizer: partial,
        scheduler: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")
        self.outp_dim = self.backbone.outp_dim

        # Create the head for the vertexing task
        self.vertex_head = MLP(
            inpt_dim=self.outp_dim, outp_dim=self.outp_dim, **vertex_config
        )
        self.dist_head = nn.Linear(self.outp_dim, 1)
        self.loss_fn = loss_fn()

        # Metrics
        self.train_metrics = nn.ModuleDict({
            "acc": Accuracy("binary"),
            "f1": F1Score("binary"),
        })
        self.valid_metrics = nn.ModuleDict({
            "acc": Accuracy("binary"),
            "f1": F1Score("binary"),
        })

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.BoolTensor) -> tuple:
        x, _ = self.backbone(csts, csts_id, mask)
        x = x[:, -csts.shape[1] :]  # Trim off registers
        x = self.vertex_head(x)  # Pass through the head

        # Get the L1 distance between all combinations
        d = (x.unsqueeze(1) - x.unsqueeze(2)).abs()

        # Return a weighted sum of the distances
        return self.dist_head(d).squeeze(-1)

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
        targ_masked = target[vtx_mask].float()

        # Pass through the backbone and head
        output = self.forward(csts, csts_id, mask).squeeze(-1)

        # Calculate the loss and log
        loss = self.loss_fn(output[vtx_mask], targ_masked)
        self.log(f"{prefix}/total_loss", loss)

        # Custom Metrics per class
        if prefix == "valid":
            for i in range(self.n_classes):  # Metric calculated per class
                c_mask = labels == i
                c_lab = ["light", "charm", "bottom"][i]
                perf = get_perf(mask[c_mask], vtx_id[c_mask], output[c_mask]).mean()
                ari = get_ari(mask[c_mask], vtx_id[c_mask], output[c_mask]).mean()
                self.log(f"{prefix}/perf_{c_lab}", perf)
                self.log(f"{prefix}/ari_{c_lab}", ari)

        # Standard Metrics
        metrics = getattr(self, f"{prefix}_metrics")
        for k in metrics:
            metrics[k](T.sigmoid(output[vtx_mask]), targ_masked)
            self.log(f"{prefix}/{k}", metrics[k])

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
