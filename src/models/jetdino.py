import random
from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    log_softmax,
    normalize,
    pairwise_distance,
    softmax,
)
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.torch_utils import ema_param_sync, set_eval
from mltools.mltools.transformers import Transformer
from src.models.utils import MLP, JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class DINOv2Loss(nn.Module):
    """DINOv2 loss with sinkhorn-knopp centering."""

    def __init__(
        self,
        dim=int,
        s_temp: float = 0.1,
        t_temp: float = 0.05,
        momentum: float = 0.9,
        centering_type: str = "swav",
    ) -> None:
        super().__init__()
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.momentum = momentum
        self.centering_type = centering_type
        self.register_buffer("t_center", T.zeros(1, dim))

    @T.no_grad()
    def swav_center(self, t_out: T.Tensor) -> T.Tensor:
        """Apply sinkhorn-Knopp centering to the teacher outputs."""
        Q = T.exp(t_out.float() / self.t_temp).t()
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes
        Q /= Q.sum()
        for _ in range(3):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()

    @T.no_grad()
    def momentum_center(self, t_out: T.Tensor) -> T.Tensor:
        """Apply momentum centering to the teacher outputs."""
        t_out = softmax((t_out - self.center) / self.t_temp, dim=-1)
        if self.training:
            self.t_center *= self.momentum
            self.t_center += (1 - self.momentum) * t_out.mean(dim=0)
        return t_out

    def forward(self, s_out: T.Tensor, t_out: T.Tensor) -> T.Tensor:
        """Calculate the loss given the pre-computed teacher centers."""
        # Center the teacher outputs
        if self.centering_type == "swav":
            t_centered = self.swav_center(t_out)
        elif self.centering_type == "momentum":
            t_centered = self.momentum_center(t_out)
        elif self.centering_type == "none":
            t_centered = softmax(t_out / self.t_temp, dim=-1)
        else:
            raise ValueError(f"Unknown centering type: {self.centering_type}")

        # Calculate the loss
        s_lsm = log_softmax(s_out / self.s_temp, dim=-1)
        return -(t_centered * s_lsm).sum(dim=-1).mean()


def mean_entropy(x: T.Tensor, temp: float = 0.1) -> T.Tensor:
    """The average entropy of a batch of logits."""
    marginal = softmax(x / temp, dim=-1).mean(dim=0)
    return (marginal ** (-marginal)).log().sum()


@T.autocast("cuda", enabled=False)
def koleo_loss(x: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """
    # Normalize the input
    x = normalize(x, eps=eps, dim=-1)

    # Calculate the matching pair idxes via the max inner produce
    with T.no_grad():
        dots = T.mm(x, x.t())
        dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)  # Fill the diagonal with -1
        min_idx = T.argmax(dots, dim=1)

    # Get the distance between closest pairs
    distances = pairwise_distance(x, x[min_idx])

    # Return the kozachenko-leonenko entropy
    return -T.log(distances + eps).mean()


class DINOHead(nn.Module):
    """The projection head for DINO-v2.

    Adapted from:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        bottleneck_dim: int = 0,
    ) -> None:
        super().__init__()
        bottleneck_dim or inpt_dim // 2
        self.mlp = MLP(
            inpt_dim=inpt_dim,
            outp_dim=bottleneck_dim or inpt_dim // 2,
            hddn_dim=inpt_dim,
            num_blocks=2,
            act_h="SiLU",
        )
        self.apply(self.reset_params)
        self.last_layer = weight_norm(nn.Linear(self.mlp.outp_dim, outp_dim))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def reset_params(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == T.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


class JetDINO(pl.LightningModule):
    """Dino-v2 (really iBOT) model for jets."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        embed_dim: int = 4096,
        t_ema: float = 0.992,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # The normalisation layer (for the continuous features only)
        self.normaliser = IterativeNormLayer(self.csts_dim)

        # The transformer encoder for the constituents
        self.encoder = Transformer(**encoder_config)
        self.t_encoder = Transformer(**encoder_config)

        # The different linear embedding layers
        self.csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)
        self.t_csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.t_ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)

        # The projection head for into the contrastive space
        self.projector = DINOHead(inpt_dim=self.encoder.dim, outp_dim=embed_dim)
        self.t_projector = DINOHead(inpt_dim=self.encoder.dim, outp_dim=embed_dim)

        # The "contrastive" loss function
        self.dino_loss = DINOv2Loss(dim=embed_dim)

        # Turn off gradients for teacher components as it "learns" via EMA
        self.t_ema = t_ema
        self.t_csts_id_embedder.requires_grad_(False)
        self.t_ctst_embedder.requires_grad_(False)
        self.t_encoder.requires_grad_(False)
        self.t_projector.requires_grad_(False)

        # Save the output dimensions
        self.output_dim = self.encoder.dim
        self.embed_dim = embed_dim

        # The learnable null token (unique for positional encoding)
        self.null_token = nn.Parameter(T.randn((self.num_csts, self.output_dim)) * 1e-3)

        # Basic classifier and accuracy for evaluating encoder
        self.classifier_head = class_head(inpt_dim=self.encoder.dim, outp_dim=n_classes)
        self.acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, sample: T.Tensor, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        null_mask = sample["null_mask"]

        # Normalise the continuous features and update the stats
        normed_csts = self.normaliser(csts, mask)

        # Get the inverse mask so the student gets two views
        inv_mask = mask & ~null_mask

        # Pass through the student model with dropped nodes under both configs
        cls_s1, x_s1 = self.mask_and_encode(normed_csts, csts_id, mask, null_mask)
        cls_s2, x_s2 = self.mask_and_encode(normed_csts, csts_id, mask, inv_mask)

        # Pass through the teacher model without dropping
        with set_eval(self), T.no_grad():
            e_t, e_mask = self.pass_teacher(normed_csts, csts_id, mask)
            w_t = self.t_projector(e_t)
            cls_t = w_t[:, 0]
            x_t = w_t[:, self.t_encoder.num_registers :]

        # Get the dino losses for the class tokens using both student views
        loss_dino = self.dino_loss(cls_s1, cls_t)
        loss_dino += self.dino_loss(cls_s2, cls_t)

        # Get the ibot losses for the constituent using both student views
        # Ibot losses only happen half the time
        if random.random() < 0.5:
            loss_ibot = self.dino_loss(x_s1[mask], x_t[mask])
            loss_ibot += self.dino_loss(x_s2[mask], x_t[mask])
        else:
            loss_ibot = T.zeros(1, device=self.device)

        # Do an intervention for the loss sometimes being NaN
        if T.isnan(loss_dino):
            loss_dino = T.zeros(1, device=self.device)
            print("Intervention: loss_dino was NaN")
        if T.isnan(loss_ibot):
            loss_ibot = T.zeros(1, device=self.device)
            print("Intervention: loss_ibot was NaN")

        # Combine the losses
        total_loss = loss_dino + loss_ibot

        # Perform the ema updates (training only)
        if self.training:
            ema_param_sync(self.csts_id_embedder, self.t_csts_id_embedder, self.t_ema)
            ema_param_sync(self.ctst_embedder, self.t_ctst_embedder, self.t_ema)
            ema_param_sync(self.encoder, self.t_encoder, self.t_ema)
            ema_param_sync(self.projector, self.t_projector, self.t_ema)

        # Measure the occupancy of the teacher outputs
        cls_occ = T.unique(T.argmax(cls_t, dim=-1)).size(0) / self.embed_dim
        x_occ = T.unique(T.argmax(x_t[mask], dim=-1)).size(0) / self.embed_dim
        self.log(f"{prefix}/cls_occ", cls_occ)
        self.log(f"{prefix}/x_occ", x_occ)

        # Dont run probe too often as we must be fair to MPM!
        if batch_idx % 20 == 0 or prefix == "valid":
            class_out = self.classifier_head(e_t.detach(), mask=e_mask)
            probe_loss = cross_entropy(class_out, labels)
            total_loss += probe_loss
            self.acc(class_out, labels)  # Updates internal state
            self.log(f"{prefix}/probe_loss", probe_loss)
            self.log(f"{prefix}/accuracy", self.acc)

        # Log the metrics
        self.log(f"{prefix}/total_loss", total_loss)
        self.log(f"{prefix}/dino_loss", loss_dino)
        self.log(f"{prefix}/ibot_loss", loss_ibot)
        return total_loss

    def mask_and_encode(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.Tensor,
    ) -> tuple:
        """Drop the input nodes and pass through the encoder."""
        # Pass the inputs through their respective embedding layers and sum
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)

        # Create array which allows us to index the null_mask in order per jet
        nt = self.null_token[: x.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Give positional encoding to the inputs
        x[null_mask] = nt[null_sorted].type(x.dtype)

        # Pass through the encoder (might gain registers)
        x = self.encoder(x, mask=mask)
        x = self.projector(x)  # Project into the contrastive space

        # Split off the registers, keep 1 for the cls token, others are dropped
        return x[:, 0], x[:, self.encoder.num_registers :]

    def pass_teacher(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Get the outputs of the teacher."""
        x = self.t_ctst_embedder(csts) + self.t_csts_id_embedder(csts_id)
        x = self.t_encoder(x, mask=mask)
        mask = self.t_encoder.get_combined_mask(mask)
        return x, mask

    def pass_student(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Get the outputs of the student."""
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)
        x = self.encoder(x, mask=mask)
        mask = self.encoder.get_combined_mask(mask)
        return x, mask

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_train_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        backbone = JetBackbone(
            normaliser=self.normaliser,
            ctst_embedder=self.t_ctst_embedder,
            id_embedder=self.t_csts_id_embedder,
            encoder=self.t_encoder,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
