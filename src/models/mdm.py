from functools import partial

import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.modules import CosineEncodingLayer
from mltools.mltools.torch_utils import append_dims
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class MaskedDiffusionModelling(pl.LightningModule):
    """Class for all masked diffusino modelling pre-training."""

    def __init__(
        self,
        *,
        data_sample: dict,
        n_classes: int,
        encoder_config: dict,
        decoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        time_dim: int = 32,
        class_head: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # Attributes
        self.n_classes = n_classes

        # The transformer encoder
        self.encoder = Transformer(**encoder_config)
        self.decoder = Transformer(
            inpt_dim=self.csts_dim + self.encoder.outp_dim,
            ctxt_dim=time_dim,
            outp_dim=self.csts_dim,
            **decoder_config,
        )
        self.n_reg = self.encoder.num_registers
        self.outp_dim = self.encoder.outp_dim

        # The embedding layers
        self.csts_emb = nn.Linear(self.csts_dim, self.encoder.dim)
        self.csts_id_emb = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.time_encoder = CosineEncodingLayer(inpt_dim=1, encoding_dim=time_dim)

        # Additional linear head for the ID task
        self.id_head = nn.Linear(self.outp_dim, CSTS_ID)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.Tensor) -> T.Tensor:
        x = self.csts_emb(csts) + self.csts_id_emb(csts_id)
        enc_out = self.encoder(x, mask=mask)
        full_mask = self.encoder.get_combined_mask(mask)
        return enc_out, full_mask

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        null_mask = data["null_mask"]
        labels = data["labels"]

        # Get the encoded output with null mask
        enc_out, _ = self.forward(csts, csts_id, mask & ~null_mask)
        full_mask = self.encoder.get_combined_mask(mask)  # Other is only for null

        # Get and log the losses
        id_loss = self.get_id_loss(prefix, csts_id, enc_out, null_mask)
        diff_loss = self.get_diff_loss(prefix, csts, enc_out, full_mask, null_mask)
        probe_loss = self.get_probe_loss(prefix, csts, csts_id, mask, labels, batch_idx)

        # Combine and return the losses
        total_loss = id_loss + diff_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def get_probe_loss(
        self,
        prefix: str,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
        labels: T.Tensor,
        batch_idx: int,
    ) -> T.Tensor:
        """Calculate the detached probe loss and accuracy."""
        if prefix == "train" and batch_idx % 50 != 0:  # Skip most training steps
            return T.tensor(0, device=csts.device, dtype=csts.dtype)

        # Do a full pass without null masking and detach the output
        with T.no_grad():
            out, out_mask = self.forward(csts, csts_id, mask)

        # Calculate the probe loss and accuracy
        class_out = self.class_head(out.detach(), mask=out_mask)
        loss = F.cross_entropy(class_out, labels)
        acc = getattr(self, f"{prefix}_acc")
        acc(class_out, labels)
        self.log(f"{prefix}/probe_loss", loss)
        self.log(f"{prefix}/probe_accuracy", acc)
        return loss

    def get_id_loss(
        self,
        prefix: str,
        csts_id: T.Tensor,
        enc_out: T.Tensor,
        null_mask: T.BoolTensor,
    ) -> T.Tensor:
        id_out = self.id_head(enc_out[:, self.n_reg :])
        loss = F.cross_entropy(id_out[null_mask], csts_id[null_mask])
        self.log(f"{prefix}/id_loss", loss)
        return loss

    def get_diff_loss(
        self,
        prefix: str,
        csts: T.Tensor,
        enc_out: T.Tensor,
        full_mask: T.BoolTensor,
        null_mask: T.BoolTensor,
    ) -> T.Tensor:
        """Get the loss for the diffusion reconstruction of the jet."""
        # Sample time and noise for the diffuser
        t = T.sigmoid(T.randn(csts.shape[0], 1, device=csts.device))
        ctxt_t = self.time_encoder(t)
        t = append_dims(t, csts.ndim)
        x1 = T.randn_like(csts)
        xt = (1 - t) * csts + t * x1

        # Appropriate casting
        xt = xt.type(enc_out.dtype)
        ctxt_t = ctxt_t.type(enc_out.dtype)

        # Combine the outputs of the encoder with the noisy samples
        xt = F.pad(xt, (0, 0, self.n_reg, 0), value=0)
        xt = T.cat([xt, enc_out], dim=-1)
        v = self.decoder(xt, mask=full_mask, ctxt=ctxt_t)[:, self.n_reg :]
        loss = (v - (x1 - csts))[null_mask].square().mean()
        self.log(f"{prefix}/diff_loss", loss)
        return loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "train")

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(
            csts_emb=self.csts_emb,
            csts_id_emb=self.csts_id_emb,
            encoder=self.encoder,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")