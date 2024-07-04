from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class JetGPT(pl.LightningModule):
    """Class for GPT style jet pre-training."""

    def __init__(
        self,
        *,
        data_sample: dict,
        n_classes: int,
        encoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        vae_path: str | None = None,
        kmeans_path: str | None = None,
    ) -> None:
        """Initialise the model.

        Parameters
        ----------
        data_sample : dict
            A sample of the data to be used for initialising the model.
        n_classes : int
            The number of classes for the classifier head.
        encoder_config : dict
            The configuration for the encoder transformer.
        optimizer : partial
            The optimizer to be used.
        scheduler : dict
            The scheduler to be used.
        tasks : dict
            A dictionary of tasks to be used. Sould be a list of partials.
        vae_path : str
            The path to the VAE model to get the target tokens.
        kmeans_path : str
            The path to the kmeans model to get the target tokens.
        class_head : partial
            The class head to be used for the probe.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.n_classes = n_classes
        self.do_kmeans = kmeans_path is not None
        self.do_vae = vae_path is not None
        assert self.do_kmeans != self.do_vae, "Need one of VAE or KMeans"

        # The transformer encoder
        encoder_config["num_registers"] = 0  # GPT does not use registers
        encoder_config["max_seq_len"] = self.num_csts + 1  # One for the start token
        encoder_config["do_absolute_enc"] = True  # Needs positional encodings!
        self.encoder = Transformer(**encoder_config)
        self.outp_dim = self.encoder.outp_dim

        # The embedding layers
        self.csts_emb = nn.Linear(self.csts_dim, self.encoder.dim)
        self.csts_id_emb = nn.Embedding(CSTS_ID, self.encoder.dim)

        # Add the learnable start token
        self.start_token = nn.Parameter(T.randn((1, 1, self.encoder.dim)) * 1e-3)

        # Load clustering model
        if self.do_kmeans:
            self.kmeans = T.load(kmeans_path, map_location="cpu")
            self.num_clusters = self.kmeans.centroids.shape[1]
        else:
            self.vae = T.load(vae_path, map_location="cpu")
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.num_clusters = self.vae.quantizer.codebook_size

        # Load the objective heads with extra class for the end token
        self.head = nn.Linear(self.outp_dim, self.num_clusters + 1)
        self.id_head = nn.Linear(self.outp_dim, CSTS_ID + 1)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(self, csts: T.Tensor, csts_id: T.Tensor, mask: T.Tensor) -> T.Tensor:
        # Initial embedding
        x = self.csts_emb(csts) + self.csts_id_emb(csts_id)

        # Add the start token to the input and the mask
        st = self.start_token.expand(x.size(0), 1, -1)  # Duplicate for batch
        x = T.cat([st, x], dim=1)
        mask = F.pad(mask, (1, 0), value=True)

        # Pass with causal mask
        enc_out = self.encoder(x, mask=mask, causal=True)
        return enc_out, mask

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        labels = data["labels"]

        # Get the encoded output
        enc_out, mwstart = self.forward(csts, csts_id, mask)

        # Get and log the losses
        id_loss = self.get_id_loss(prefix, csts_id, enc_out, mwstart)
        clus_loss = self.get_clus_loss(prefix, csts, enc_out, mwstart)
        probe_loss = self.get_probe_loss(prefix, enc_out, labels, mwstart, batch_idx)

        # Combine and return the losses
        total_loss = id_loss + clus_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def get_clus_loss(
        self,
        prefix: str,
        csts: T.Tensor,
        enc_out: T.Tensor,
        mask_wstart: T.BoolTensor,
    ) -> T.Tensor:
        """Get the clustering loss using either the VAE or the KMeans."""
        mask = mask_wstart[:, 1:]  # Get the old mask for the targets

        # Get the target clusters from the VAE or the KMeans
        target = (
            self.kmeans.predict(csts[mask].T.contiguous()).long()
            if self.do_kmeans
            else self.vae(csts, mask=mask)[0].long()
        )
        clus_id = T.zeros_like(mask, dtype=T.long)  # Redo padding
        clus_id[mask] = target

        # Insert target for end token
        n_csts = mask.sum(dim=1)[:, None]
        clus_id = F.pad(clus_id, (0, 1)).scatter_(1, n_csts, self.num_clusters)

        # Get the prediction and return the loss
        clus_out = self.head(enc_out)
        loss = F.cross_entropy(
            clus_out[mask_wstart], clus_id[mask_wstart], label_smoothing=0.1
        )
        self.log(f"{prefix}/clus_loss", loss)
        return loss

    def get_id_loss(
        self,
        prefix: str,
        csts_id: T.Tensor,
        enc_out: T.Tensor,
        mask_wstart: T.BoolTensor,
    ) -> T.Tensor:
        """Get the loss for the constituent ID."""
        n_csts = mask_wstart.sum(dim=1)[:, None] - 1  # Insert target for end token
        csts_id = F.pad(csts_id, (0, 1)).scatter_(1, n_csts, CSTS_ID)

        # Get the prediction and return the loss
        id_out = self.id_head(enc_out)
        loss = F.cross_entropy(id_out[mask_wstart], csts_id[mask_wstart])
        self.log(f"{prefix}/id_loss", loss)
        return loss

    def get_probe_loss(
        self,
        prefix: str,
        enc_out: T.Tensor,
        labels: T.Tensor,
        mask_wstart: T.BoolTensor,
        batch_idx: int,
    ) -> T.Tensor:
        """Calculate the detached probe loss and accuracy."""
        if prefix == "train" and batch_idx % 50 != 0:  # Skip most training steps
            return T.tensor(0, device=enc_out.device, dtype=enc_out.dtype)

        # Make sure to detach the encoder output!
        class_out = self.class_head(enc_out.detach(), mask=mask_wstart.detach())
        loss = F.cross_entropy(class_out, labels)
        acc = getattr(self, f"{prefix}_acc")
        acc(class_out, labels)
        self.log(f"{prefix}/probe_loss", loss)
        self.log(f"{prefix}/probe_accuracy", acc)
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
            is_causal=True,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
