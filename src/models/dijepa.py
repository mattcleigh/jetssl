from functools import partial

import pytorch_lightning as pl
import torch as T
import wandb
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    log_softmax,
)
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.torch_utils import ema_param_sync
from mltools.mltools.transformers import Transformer
from src.models.utils import MLP, JetEncoder

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class Predictor(nn.Module):
    """Predictor for the JEPA model."""

    def __init__(self, inpt_dim: int, num_csts: int, **kwargs) -> None:
        super().__init__()

        # The transformer encoder for the constituents
        self.encoder = Transformer(**kwargs)
        self.dim = self.encoder.dim

        # The input and output projection layers
        self.input_proj = nn.Linear(inpt_dim, self.dim)
        self.output_proj = nn.Linear(self.dim, inpt_dim)

        # The learnable parameters for the dropped nodes in the decoder (1 per seq)
        self.null_token = nn.Parameter(T.randn((num_csts, self.dim)) * 1e-3)

    def forward(
        self, x: T.Tensor, mask: T.BoolTensor, null_mask: T.BoolTensor
    ) -> T.Tensor:
        """Pass through the predictor.

        Parameters
        ----------
        x : T.Tensor
            The embedded jet constituents. The output of the student encoder.
            May contain cls tokens and registers.
        mask : T.BoolTensor
            The mask for the input. Which nodes are real (T) and which are padding (F).
        null_mask : T.BoolTensor
            A mask which tells us which nodes were hidden from the student.
            Allows us to parameterise the predictor with the masking.
        """
        # Embed the nodes into the predictor space
        x = self.input_proj(x)

        # Get the shape of the input
        B, S = null_mask.shape

        # Trim the null tokens to seq_len and expand to match batch size
        nt = self.null_token[:S].unsqueeze(0).expand(B, S, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(S, device=x.device)  # Simple counter
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)  # Dupl for batch
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)  # Mask

        # Insert the null tokens so they are ordered wrt each other
        x[:, -S:][null_mask] = nt[null_sorted].type(x.dtype)

        # Pass through the transformer
        x = self.encoder(x, mask=mask)

        # Project back to the original space (dont need registers)
        return self.output_proj(x)


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


@T.no_grad()
def sk_center(x: T.Tensor, temp: float) -> T.Tensor:
    """Apply sinkhorn-Knopp centering, ensures that rows and columns sum to 1."""
    Q = T.exp(x.float() / temp)
    B = Q.shape[0]  # batch size
    K = Q.shape[1]  # number of prototypes
    Q /= Q.sum()
    for _ in range(3):
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize the columns
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True)  # Normalize the rows
        Q /= B
    Q *= B
    return Q


def dinov2_loss(
    s_out: T.Tensor, t_out: T.Tensor, s_temp: float = 0.1, t_temp: float = 0.05
) -> T.Tensor:
    t_centered = sk_center(t_out, t_temp)
    s_lsm = log_softmax(s_out / s_temp, dim=-1)
    loss = -(t_centered * s_lsm).sum(dim=-1)
    loss = T.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


class DiJepa(pl.LightningModule):
    """Dino-v2 model with IJEPA type predictor and an MAE type encoder."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        predictor_config: dict,
        optimizer: partial,
        scheduler: partial,
        class_head: partial,
        t_ema: float = 0.992,
        embed_dim: int = 4096,
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

        # Create the projection heads
        self.student_head = DINOHead(self.dim, outp_dim=embed_dim)
        self.teacher_head = DINOHead(self.dim, outp_dim=embed_dim)
        self.teacher_head.requires_grad_(False)

        # The predictor for mapping between the student and teacher spaces
        self.predictor = Predictor(self.dim, self.num_csts, **predictor_config)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.dim, outp_dim=n_classes)
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

        # Pass the inputs through the student with nulls dropped
        _B, S = mask.shape
        s_out, _s_mask = self.student(csts, csts_id, mask & ~null_mask)

        # Pass the inputs through the teacher without any dropping
        with T.no_grad():
            self.teacher.eval()
            t_out, t_mask = self.teacher(csts, csts_id, mask)
            t_cst = t_out[:, -S:][null_mask]
            t_reg = t_out[:, :-S]

        # Pass through the predictor using the t_mask (non-dropped with reg)
        s_out = self.predictor(s_out, t_mask, null_mask)
        s_cst = s_out[:, -S:][null_mask]
        s_reg = s_out[:, :-S]

        # Get the outputs of the heads
        t_cst = self.teacher_head(t_cst)
        t_reg = self.teacher_head(t_reg)
        s_cst = self.student_head(s_cst)
        s_reg = self.student_head(s_reg)

        # Get the loss using the teacher and predictor outputs
        cst_loss = dinov2_loss(s_cst, t_cst)
        self.log(f"{prefix}/dino_cst_loss", cst_loss)

        # Get the loss using the teacher and student registers
        reg_loss = dinov2_loss(s_reg, t_reg)
        self.log(f"{prefix}/dino_reg_loss", reg_loss)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        probe_loss = T.tensor(0.0, device=self.device)
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(t_out.detach(), mask=t_mask.detach())
            probe_loss = cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
            self.log(f"{prefix}/probe_loss", probe_loss)

        # Log the total loss
        total_loss = cst_loss + reg_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)

        # Log the occupancy of the teacher outputs
        if batch_idx % 100 == 0:
            cst_ids = T.argmax(t_cst, dim=-1)
            reg_ids = T.argmax(t_reg, dim=-1)
            cst_occ = T.unique(cst_ids).size(0) / self.embed_dim
            reg_occ = T.unique(reg_ids).size(0) / self.embed_dim
            self.log(f"{prefix}/cst_occ", cst_occ)
            self.log(f"{prefix}/reg_occ", reg_occ)
            if wandb.run is not None:
                wandb.log({"cst_ids": wandb.Histogram(cst_ids.cpu().numpy())})
                wandb.log({"reg_ids": wandb.Histogram(reg_ids.cpu().numpy())})

        # Check each of the losses for NaNs
        for loss in [("cst", cst_loss), ("reg", reg_loss), ("probe", probe_loss)]:
            if T.isnan(loss[1]):
                raise ValueError(f"NaN in {loss[0]} loss.")

        return total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        ema_param_sync(self.student, self.teacher, self.t_ema)
        ema_param_sync(self.student_head, self.teacher_head, self.t_ema)
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Have a seperate optimizer for the student and the class head."""
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        self.teacher.eval()
        T.save(self.teacher, "backbone.pkl")
