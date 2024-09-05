from functools import partial

import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.mlp import MLP
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
        ctxt_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.jets_dim = data_sample["jets"].shape[-1]

        # Attributes
        self.time_dim = time_dim
        self.ctxt_dim = ctxt_config["outp_dim"]
        self.n_classes = n_classes

        # The transformer encoder
        self.encoder = Transformer(ctxt_dim=self.ctxt_dim, **encoder_config)
        self.decoder = Transformer(
            inpt_dim=self.csts_dim + CSTS_ID,  # We concatenate the csts and csts_id
            outp_dim=self.csts_dim + CSTS_ID,  # Denoising, so inpt = outp
            ctxt_dim=time_dim,
            use_decoder=True,
            **decoder_config,
        )
        self.n_reg = self.encoder.num_registers
        self.dim = self.encoder.dim
        self.outp_dim = self.encoder.outp_dim

        # Keep everything tight!
        self.encoder.do_packed = True
        self.decoder.do_packed = True
        self.encoder.unpack_output = False
        self.decoder.unpack_output = False

        # The embedding layers
        self.csts_emb = nn.Linear(self.csts_dim, self.dim)
        self.csts_id_emb = nn.Embedding(CSTS_ID, self.dim)
        self.enc_do_dec = nn.Linear(self.outp_dim, self.decoder.dim)
        self.time_encoder = CosineEncodingLayer(inpt_dim=1, encoding_dim=time_dim)
        self.jets_emb = MLP(inpt_dim=self.jets_dim, **ctxt_config)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        null_mask = data["null_mask"]
        labels = data["labels"]
        jets = data["jets"]

        # Split the jets into the two sets (done by masking only)
        mask_1 = mask & ~null_mask
        mask_2 = mask & null_mask

        # Representations for the models (masking/splitting is later)
        x = self.csts_emb(csts) + self.csts_id_emb(csts_id)  # Inputs to encoder
        y = T.cat([csts, F.one_hot(csts_id, CSTS_ID)], dim=-1)  # Inputs to decoder

        # Embed the jets
        ctxt = self.jets_emb(jets)

        # Get the output of the encoder with the null mask
        enc_out, enc_culens, enc_maxlen = self.encoder(x, mask=mask_1, ctxt=ctxt)

        # Sample diffusion time and the noise
        t = T.sigmoid(T.randn(csts.shape[0], 1, device=csts.device))  # Batch x 1
        ctxt_t = self.time_encoder(t)  # Model needs embedded time
        t = append_dims(t, csts.ndim)
        x1 = T.randn_like(y)
        xt = (1 - t) * y + t * x1  # Mix the two
        v = x1[mask_2] - y[mask_2]  # Target velocity

        # Get the output of the decoder using the opposite mask, time and context
        v_hat, _, _ = self.decoder(
            xt,
            mask=mask_2,
            ctxt=ctxt_t,
            kv=self.enc_do_dec(enc_out),  # Resize for the decoder
            kv_culens=enc_culens,
            kv_maxlen=enc_maxlen,
        )

        # Calculate the loss based on the velocity vector
        diff_loss = (v_hat - v).square().mean()
        self.log(f"{prefix}/diff_loss", diff_loss)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        probe_loss = 0
        if batch_idx % 50 == 0 or prefix == "valid":
            with T.no_grad():
                self.encoder.unpack_output = True
                # Times the context by zero to indicate nothing was dropped
                t_out = self.encoder(x, mask=mask, ctxt=ctxt * 0).detach()
                t_mask = self.encoder.get_combined_mask(mask)
                self.encoder.unpack_output = False
            class_out = self.class_head(t_out, mask=t_mask)
            probe_loss = F.cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_accuracy", acc)
            self.log(f"{prefix}/probe_loss", probe_loss)

        # Combine and return the losses
        total_loss = diff_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

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
            ctxt_emb=self.jets_emb,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
