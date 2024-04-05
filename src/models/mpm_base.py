from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch as T
import wandb
from torch import nn
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.torch_utils import set_eval
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone
from src.plotting import plot_continuous, plot_labels

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class MPMBase(pl.LightningModule):
    """Base class for masked particle modelling.

    The base class can only be used for modelling the csts_id features using basic cross
    entropy loss.

    To learn the csts features, the user must inherit from this class and define a
    masked_recovery_loss method. .
    """

    def __init__(
        self,
        *,
        data_sample: dict,
        n_classes: int,
        encoder_config: dict,
        decoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # The normalisation layer (for the continuous features only)
        self.normaliser = IterativeNormLayer(self.csts_dim)

        # The transformer encoder (encoder, no positional encoding)
        self.encoder = Transformer(**encoder_config)

        # The decoder used for the MAE objective (no positional encoding)
        self.decoder = Transformer(**decoder_config)

        # The different linear embedding layers for inputs, between enc, and outputs
        self.csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)
        self.enc_to_dec = nn.Linear(self.encoder.dim, self.decoder.dim)
        self.csts_id_head = nn.Linear(self.decoder.dim, CSTS_ID)

        # The learnable parameters for the dropped nodes in the decoder (1 per seq)
        self.null_token = nn.Parameter(
            T.randn((self.num_csts, self.decoder.dim)) * 1e-3
        )

        # Basic classifier and accuracy for evaluating encoder
        self.classifier_head = class_head(inpt_dim=self.encoder.dim, outp_dim=n_classes)
        self.acc = Accuracy("multiclass", num_classes=n_classes)

        # Test the save epoch end callback which saves an untrained backbone
        self.on_train_epoch_end()

    def _shared_step(self, sample: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        null_mask = sample["null_mask"]

        # Normalise the continuous features and update the stats
        normed_csts = self.normaliser(csts, mask)

        # Encode and decode
        enc_outs, enc_mask = self.drop_and_encode(normed_csts, csts_id, mask, null_mask)
        decoder_outs = self.combine_and_decode(enc_outs, enc_mask, mask, null_mask)

        # Pass through the different heads and get the reconstruction losses
        id_loss = self.masked_id_loss(csts_id, null_mask, decoder_outs)
        cst_loss = self.masked_cst_loss(normed_csts, csts_id, null_mask, decoder_outs)

        # Combine the losses
        loss = id_loss + cst_loss

        # Use the probe on all inputs! Not very often!
        if batch_idx % 20 == 0 or prefix == "valid":
            with set_eval(self), T.no_grad():
                full, full_mask = self.full_pass(normed_csts, csts_id, mask)
            probe_loss = self.get_probe_loss(full.detach(), full_mask.detach(), labels)
            loss = loss + probe_loss
            self.log(f"{prefix}/accuracy", self.acc)
            self.log(f"{prefix}/probe_loss", probe_loss)

        # Despite my best efforts sometimes the loss is NaN :(
        if T.isnan(loss):
            print("Intervention!")
            loss = T.tensor(0.0, device=self.device)

        # Log the metrics
        self.log(f"{prefix}/total_loss", id_loss + cst_loss)
        self.log(f"{prefix}/id_loss", id_loss)
        self.log(f"{prefix}/cst_loss", cst_loss)

        # Do inpainting for validation
        if prefix == "valid" and batch_idx == 0:
            self.inpaint(csts, csts_id, mask, null_mask, decoder_outs)

        return loss

    def drop_and_encode(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.Tensor,
    ) -> tuple:
        """Drop the input nodes and pass through the encoder."""
        # Pass the inputs through their respective embedding layers and sum
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)

        # Pass through the encoder (might gain registers)
        x = self.encoder(x, mask=mask & ~null_mask)
        mask = self.encoder.get_combined_mask(mask)
        return x, mask

    def combine_and_decode(
        self,
        enc_outs: T.Tensor,
        enc_mask: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.BoolTensor,
    ) -> T.Tensor:
        """Recombine the encoder outputs with the dropped nodes and decode."""
        dec_inpts = self.enc_to_dec(enc_outs)
        n_reg = self.encoder.num_registers

        # Trim and expand the null tokens to match the decoder input sequence
        nt = self.null_token[: dec_inpts.size(1) - n_reg]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Insert the null tokens so they are ordered wrt each other
        dec_inpts[:, n_reg:][null_mask] = nt[null_sorted].type(dec_inpts.dtype)

        # Pass through the decoder dont need registers after
        return self.decoder(dec_inpts, mask=enc_mask)[:, n_reg:]

    def masked_id_loss(
        self,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calculate particle ID loss used to train the encoder."""
        csts_id_out = self.csts_id_head(decoder_outs[null_mask])
        return cross_entropy(csts_id_out, csts_id[null_mask], label_smoothing=0.1)

    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calculate the particle loss used to train the encoder.

        This is what must be implemented in the sub-classes!!!!.
        """
        return T.tensor(0.0, device=self.device)

    def sample_csts_id(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Return a collection of csts_id samples under the learned distribution."""
        id_outs = self.csts_id_head(decoder_outs)
        id_probs = T.softmax(id_outs, dim=-1)
        return T.multinomial(id_probs, 1).squeeze(1)

    def sample_csts(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Return a collection of csts samples under the learned distribution."""
        return T.tensor(0.0, device=self.device)

    @T.no_grad()
    def inpaint(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor | None,
        mask: T.BoolTensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> None:
        """For visualising the masked recovery process."""
        # Dont do anything if there is no active logger
        if wandb.run is None:
            return

        # Sample to recover the masked features
        csts_samples = self.sample_csts(decoder_outs[null_mask])
        csts_id_samples = self.sample_csts_id(decoder_outs[null_mask])

        # Place the sampled values back into the original tensors
        rec_csts = T.clone(csts)
        rec_csts_id = T.clone(csts_id)
        rec_csts[null_mask] = csts_samples
        rec_csts_id[null_mask] = csts_id_samples

        # Plot the distributions
        Path("plots").mkdir(exist_ok=True)
        plot_continuous(csts, mask, null_mask, rec_csts)
        plot_labels(csts_id, mask, null_mask, rec_csts_id)

    def get_probe_loss(
        self, encoder_outputs: T.Tensor, encoder_mask: T.BoolTensor, labels: T.Tensor
    ) -> tuple:
        """Run the linear classifier using the encoder outputs."""
        class_out = self.classifier_head(encoder_outputs, mask=encoder_mask)
        loss = cross_entropy(class_out, labels)
        self.acc(class_out, labels)  # Updates internal state
        return loss

    def full_pass(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Full forward pass for inference."""
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)
        x = self.encoder(x, mask=mask)
        new_mask = self.encoder.get_combined_mask(mask)
        return x, new_mask

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_train_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(
            normaliser=self.normaliser,
            ctst_embedder=self.ctst_embedder,
            id_embedder=self.csts_id_embedder,
            encoder=self.encoder,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
