from copy import deepcopy
from functools import partial

import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import get_max_steps, simple_optim_sched
from mltools.mltools.torch_utils import ema_param_sync, occupancy
from mltools.mltools.transformers import Transformer
from src.models.utils import (
    DINOHead,
    JetEncoder,
    dinov2_loss,
    repulse_loss,
    varcov_loss,
)


class JetJEPA(pl.LightningModule):
    """JEPA for running on jets."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        decoder_config: dict,
        ctxt_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        backbone_path: str | None = None,
        ema_start: float = 0.992,
        do_cls_loss: bool = False,
        do_dino: bool = False,
        do_repulse: bool = True,
        dino_dim: int = 4096,
        use_ctxt: bool = False,
        do_varcov: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ctxt_dim = data_sample["jets"].shape[-1]
        self.ema_start = ema_start
        self.do_cls_loss = do_cls_loss
        self.do_dino = do_dino
        self.dino_dim = dino_dim
        self.use_ctxt = use_ctxt
        self.do_varcov = do_varcov
        self.do_repulse = do_repulse

        # The student (online) model, support loading from a backbone
        self.student = (
            T.load(backbone_path)
            if backbone_path is not None
            else JetEncoder(
                csts_dim=self.csts_dim,
                encoder_config=encoder_config,
                ctxt_dim=self.ctxt_dim * use_ctxt,
                ctxt_config=ctxt_config,
            )
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
        jets = sample["jets"] if self.use_ctxt else None

        # Pass the inputs through the student model, dropping some nodes
        _B, S, _D = csts.shape
        s_out, _s_mask = self.student.forward(csts, csts_id, mask & ~null_mask, jets)

        # Pass the inputs through the teacher model, without the null mask
        with T.no_grad():
            t_out, t_mask = self.teacher(csts, csts_id, mask, jets)

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
        p_out = self.dec_to_enc(p_out)

        # Calculate the prediction losses
        p_seq = p_out[:, -S:][mask]
        t_seq = t_out[:, -S:][mask]
        if self.do_dino:
            p_seq = self.student_head(p_seq)
            t_seq = self.teacher_head(t_seq)
            tok_loss = dinov2_loss(p_seq, t_seq)
            self.log(f"{prefix}/x_occ", occupancy(t_seq))
        elif self.do_repulse:
            tok_loss = repulse_loss(p_seq, t_seq)
        else:
            tok_loss = F.smooth_l1_loss(p_seq, t_seq)  # paper=mse, official code=sl1
        self.log(f"{prefix}/sequence_loss", tok_loss)

        # If we are using the variance-covariance loss, include it
        # This loss comes directly from the student so we only use the visible nodes
        varcov_seq = 0
        if self.do_varcov:
            varcov_seq, var_seq, cov_seq = varcov_loss(s_out[:, -S:][mask & ~null_mask])
            self.log(f"{prefix}/var_seq", var_seq)
            self.log(f"{prefix}/cov_seq", cov_seq)

        # Inlcude loss from the cls token (first register token)
        cls_loss = 0
        varcov_cls = 0
        if self.do_cls_loss:
            p_cls = p_out[:, 0]
            t_cls = t_out[:, 0]
            if self.do_dino:
                p_cls = self.student_head(p_cls)
                t_cls = self.teacher_head(t_cls)
                cls_loss = dinov2_loss(p_cls, t_cls)
                self.log(f"{prefix}/cls_occ", occupancy(t_cls))
            elif self.do_repulse:
                cls_loss = repulse_loss(p_cls, t_cls)
            else:
                cls_loss = F.smooth_l1_loss(p_cls, t_cls)
            if self.do_varcov:
                varcov_cls, var_cls, cov_cls = varcov_loss(s_out[:, 0])
                self.log(f"{prefix}/var_cls", var_cls)
                self.log(f"{prefix}/cov_cls", cov_cls)
            self.log(f"{prefix}/cls_loss", cls_loss)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        probe_loss = 0
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(t_out.detach(), mask=t_mask.detach())
            probe_loss = F.cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
            self.log(f"{prefix}/probe_loss", probe_loss)

        # Combine the losses
        total_loss = tok_loss + cls_loss + probe_loss + varcov_seq + varcov_cls
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        ema = self.get_ema()
        ema_param_sync(self.student, self.teacher, ema)
        if self.do_dino:
            ema_param_sync(self.student_head, self.teacher_head, ema)
        self.log("ema", ema)
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
