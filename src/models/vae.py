from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    huber_loss,
)
from torchmetrics import Accuracy
from vector_quantize_pytorch import VectorQuantize

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone


class JetVQVAE(pl.LightningModule):
    """Transformer based vector quantised autoencoder for point cloud data.
    We use a symmetric encoder and decoder with a VQ layer in the middle.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        quantizer_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # The encoder and decoder
        self.encoder = Transformer(**encoder_config)
        self.decoder = Transformer(**encoder_config)

        # Save the dimensions used in the student
        self.dim = self.encoder.dim
        self.outp_dim = self.encoder.outp_dim
        self.latent_dim = latent_dim
        self.n_reg = self.encoder.num_registers

        # The embedding layers and output layers
        self.csts_emb = nn.Linear(self.csts_dim, self.dim)
        self.csts_out = nn.Linear(self.dim, self.csts_dim)

        # The vector quantizing layer
        self.quantizer = VectorQuantize(self.dim, **quantizer_config)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(self, csts: T.Tensor, mask: T.Tensor) -> T.Tensor:
        """Forward pass through the network, needed for hooks."""
        # Encode
        x = self.csts_emb(csts)
        enc_outs = self.encoder(x, mask=mask)[:, self.n_reg :]  # Trim the registers

        # Quantize
        z_masked, indices, vq_loss = self.quantizer(enc_outs[mask])
        z = T.zeros_like(enc_outs, dtype=z_masked.dtype)
        z[mask] = z_masked

        # Decode
        dec_outs = self.decoder(z, mask=mask)[:, self.n_reg :]
        dec_outs = self.csts_out(dec_outs)

        return enc_outs, dec_outs, indices, vq_loss

    def _shared_step(self, sample: T.Tensor, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        labels = sample["labels"]
        mask = sample["mask"]

        # Pass through the model
        enc_outs, dec_outs, indices, vq_loss = self.forward(csts, mask)
        self.log(f"{prefix}/vq_loss", vq_loss)

        # Calculate the lossses
        loss_csts = huber_loss(dec_outs[mask], csts[mask])
        self.log(f"{prefix}/loss_csts", loss_csts)

        # Run the probe to evaluate the embedding (once every 50 batches)
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(enc_outs.detach(), mask=mask.detach())
            probe_loss = cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
            self.log(f"{prefix}/probe_loss", probe_loss)
        else:
            probe_loss = T.zeros(1, device=self.device)

        # Log the occupancy of the codebook
        occ = len(T.unique(indices)) / self.quantizer.codebook_size
        self.log(f"{prefix}/codebook_occ", occ)

        # Combine and log the losses
        total_loss = loss_csts + probe_loss + vq_loss * 10
        self.log(f"{prefix}/total_loss", total_loss)

        return total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(self.csts_emb, None, self.encoder)
        quantizer = ConstituentQuantizer(backbone, self.quantizer)
        quantizer.eval()
        T.save(quantizer, "quantizer.pkl")


class ConstituentQuantizer(nn.Module):
    """Combination of a backbone and a quantizer for the constituents."""

    def __init__(self, backbone: JetBackbone, quantizer: VectorQuantize) -> None:
        super().__init__()
        self.backbone = backbone
        self.quantizer = quantizer
        self.n_reg = self.backbone.encoder.num_registers

    def forward(self, csts: T.Tensor, mask: T.Tensor) -> T.Tensor:
        # Encode
        x = self.backbone.csts_emb(csts)
        enc_outs = self.backbone.encoder(x, mask=mask)[:, self.n_reg :]

        # Quantize
        z_masked, indices, vq_loss = self.quantizer(enc_outs[mask])
        z = T.zeros_like(enc_outs, dtype=z_masked.dtype)
        z[mask] = z_masked

        return indices, z, enc_outs, vq_loss
