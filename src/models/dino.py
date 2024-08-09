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
from mltools.mltools.torch_utils import ema_param_sync
from src.models.utils import MLP, JetEncoder

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
        loss = -(t_centered * s_lsm).sum(dim=-1)
        loss = T.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss.mean()


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
        self.mlp = MLP(
            inpt_dim=inpt_dim,
            outp_dim=bottleneck_dim or inpt_dim // 2,
            hddn_dim=bottleneck_dim or inpt_dim // 2,
            num_blocks=1,
            act_h="SiLU",
            act_o="SiLU",
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


@T.autocast("cuda", enabled=False)
def koleo_loss(x: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """
    # Normalize the input
    x = normalize(x, eps=eps, dim=-1)

    # Calculate the matching pair idxes via the max inner product
    with T.no_grad():
        dots = T.mm(x, x.t())
        dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)  # Fill the diagonal with -1
        min_idx = T.argmax(dots, dim=1)

    # Get the distance between closest pairs
    distances = pairwise_distance(x, x[min_idx])

    # Return the kozachenko-leonenko entropy
    return -T.log(distances + eps).mean()


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
        backbone_path: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.embed_dim = embed_dim
        self.t_ema = t_ema

        # The student and teacher models
        if backbone_path is not None:
            self.student = T.load(backbone_path)
            self.teacher = T.load(backbone_path)
        else:
            self.student = JetEncoder(
                csts_dim=self.csts_dim, encoder_config=encoder_config
            )
            self.teacher = JetEncoder(
                csts_dim=self.csts_dim, encoder_config=encoder_config
            )
        self.teacher.requires_grad_(False)  # Teacher is an EMA of student

        # Save the dimensions used in the student
        self.dim = self.student.dim
        self.outp_dim = self.student.outp_dim

        # Create the additional projection heads
        self.student_proj = DINOHead(inpt_dim=self.outp_dim, outp_dim=embed_dim)
        self.teacher_proj = DINOHead(inpt_dim=self.outp_dim, outp_dim=embed_dim)
        self.teacher_proj.requires_grad_(False)

        # The non-contrastive loss function
        self.dino_loss = DINOv2Loss(dim=embed_dim)

        # The learnable null token (unique for positional encoding)
        self.null_token = nn.Parameter(T.randn((self.num_csts, self.dim)) * 1e-3)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, sample: T.Tensor, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        null_mask = sample["null_mask"]

        # Pass through the student model with dropped nodes
        cls_s, x_s = self.mask_and_encode(csts, csts_id, mask, null_mask)

        # Pass through the teacher model without dropping
        # We want to keep the raw backbone output for the probe
        with T.no_grad():
            self.teacher.eval()
            self.teacher_proj.eval()
            _B, S, _D = csts.shape
            e_t, e_mask = self.teacher(csts, csts_id, mask)
            w_t = self.teacher_proj(e_t)
            cls_t = w_t[:, 0]
            x_t = w_t[:, -S:]

        # Get the dino losses using the class tokens
        loss_dino = self.dino_loss(cls_s, cls_t)

        # Get the ibot losses using the constituents
        # Ibot losses only happen half the time
        if random.random() < 0.5:
            loss_ibot = self.dino_loss(x_s[mask], x_t[mask])
        else:
            loss_ibot = 0.0

        # Perform the ema updates (while training only)
        if self.training:
            ema_param_sync(self.student, self.teacher, self.t_ema)
            ema_param_sync(self.student_proj, self.teacher_proj, self.t_ema)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(e_t.detach(), mask=e_mask.detach())
            probe_loss = cross_entropy(class_out, labels)

            # Log the probe accuracy
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
        else:
            probe_loss = 0.0

        # Combine and log the losses
        total_loss = loss_dino + loss_ibot + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        self.log(f"{prefix}/loss_dino", loss_dino)
        self.log(f"{prefix}/loss_ibot", loss_ibot)
        self.log(f"{prefix}/probe_loss", probe_loss)

        # Log the occupancy of the teacher outputs
        cls_occ = T.unique(T.argmax(cls_t, dim=-1)).size(0) / self.embed_dim
        x_occ = T.unique(T.argmax(x_t[mask], dim=-1)).size(0) / self.embed_dim
        self.log(f"{prefix}/cls_occ", cls_occ)
        self.log(f"{prefix}/x_occ", x_occ)

        return total_loss

    def mask_and_encode(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.Tensor,
    ) -> tuple:
        """Drop the input nodes and pass through the encoder."""
        # Embed into the transformer dimension
        x = self.student.csts_emb(csts) + self.student.csts_id_emb(csts_id)

        # Create array which allows us to index the null_mask in order per jet
        B, S, _D = x.shape
        nt = self.null_token[:S]
        nt = nt.unsqueeze(0).expand(B, S, -1)
        null_sorted = T.arange(S, device=self.device).unsqueeze(0).expand(B, S)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Give positional encoding to the inputs
        x[null_mask] = nt[null_sorted].type(x.dtype)

        # Pass through the encoder (might gain registers)
        x = self.student.encoder(x, mask=mask)
        x = self.student_proj(x)  # Project into the contrastive space

        # Split off the registers, keep 1 for the cls token, others are dropped
        return x[:, 0], x[:, -S:]

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        self.teacher.eval()
        T.save(self.teacher, "backbone.pkl")
