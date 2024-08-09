from copy import deepcopy
from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    smooth_l1_loss,
)
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import get_max_steps, simple_optim_sched
from mltools.mltools.torch_utils import ema_param_sync, occupancy
from mltools.mltools.transformers import Transformer
from src.models.utils import DINOHead, JetEncoder, dinov2_loss


class JetJEPA(pl.LightningModule):
    """JEPA for running on jets."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        decoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        backbone_path: str | None = None,
        ema_start: float = 0.992,
        do_cls_loss: bool = False,
        do_dino: bool = False,
        dino_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ema_start = ema_start
        self.do_cls_loss = do_cls_loss
        self.do_dino = do_dino
        self.dino_dim = dino_dim

        # The student (online) model, support loading from a backbone
        self.student = (
            T.load(backbone_path)
            if backbone_path is not None
            else JetEncoder(csts_dim=self.csts_dim, encoder_config=encoder_config)
        )

        # The teacher (ema) model, official code is a copy not a reinit
        self.teacher = deepcopy(self.student)
        self.teacher.requires_grad_(False)  # No direct optimisation

        # The predictor (decoder) for mapping between the student and teacher spaces
        self.decoder = Transformer(**decoder_config)

        # The linear layers for mapping between the encoder and predictor spaces
        self.outp_dim = self.student.outp_dim
        self.enc_to_dec = nn.Linear(self.outp_dim, self.decoder.dim)
        self.dec_to_enc = nn.Linear(self.decoder.dim, self.outp_dim)

        # The learnable parameters for the dropped nodes in the predictor (1 per seq)
        self.null_token = nn.Parameter(T.randn((self.num_csts, self.decoder.dim)))

        # If we are using the DINO, we need to create the heads
        if do_dino:
            self.student_head = DINOHead(self.outp_dim, outp_dim=dino_dim)
            self.teacher_head = deepcopy(self.student_head)
            self.teacher_head.requires_grad_(False)

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

        # Pass the inputs through the student model, dropping some nodes
        _B, S, _D = csts.shape
        s_out, _s_mask = self.student(csts, csts_id, mask & ~null_mask)

        # Pass the inputs through the teacher model, without the null mask
        with T.no_grad():
            t_out, t_mask = self.teacher(csts, csts_id, mask)
            target = t_out[:, -S:][mask]  # Strip registers and unmask

        # Resize to the predictor space and store the number of registers
        dec_inpts = self.enc_to_dec(s_out)

        # Trim the null tokens to seq_len and expand to match batch size
        nt = self.null_token[: null_mask.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Insert the null tokens so they are ordered wrt each other
        dec_inpts[:, -S:][null_mask] = nt[null_sorted].type(dec_inpts.dtype)

        # Pass through the predictor, we dont need registers after
        p_out = self.decoder(dec_inpts, mask=t_mask)
        p_out = p_out[:, -S:][mask]  # Strip registers and unmask
        p_out = self.dec_to_enc(p_out)

        # Calculate the prediction losses
        if self.do_dino:
            p_out = self.student_head(p_out)
            target = self.teacher_head(target)
            pred_loss = dinov2_loss(p_out, target)
            self.log(f"{prefix}/x_occ", occupancy(target))
        else:
            pred_loss = smooth_l1_loss(p_out, target)  # paper=mse, official code=sl1
        self.log(f"{prefix}/pred_loss", pred_loss)

        # Inlcude loss from the cls token (first register token)
        cls_loss = 0
        if self.do_cls_loss:
            if self.do_dino:
                s_cls = self.student_head(s_out[:, 0])
                t_cls = self.teacher_head(t_out[:, 0])
                cls_loss = dinov2_loss(s_cls, t_cls)
                self.log(f"{prefix}/cls_occ", occupancy(t_cls))
            else:
                cls_loss = smooth_l1_loss(s_out[:, 0], t_out[:, 0])
            self.log(f"{prefix}/cls_loss", cls_loss)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        probe_loss = 0
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(t_out.detach(), mask=t_mask.detach())
            probe_loss = cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
            self.log(f"{prefix}/probe_loss", probe_loss)

        # Combine the losses
        total_loss = pred_loss + cls_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        ema_param_sync(self.student, self.teacher, self.get_ema())
        if self.do_dino:
            ema_param_sync(self.student_head, self.teacher_head, self.get_ema())
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        self.max_steps = get_max_steps(self)
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        self.teacher.eval()
        T.save(self.teacher, "backbone.pkl")

    def get_ema(self) -> None:
        """Method to calculate the EMA decay for the teacher network."""
        step = self.trainer.global_step
        return min(1, self.ema_start + step * (1 - self.ema_start) / self.max_steps)
