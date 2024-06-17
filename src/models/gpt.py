from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchpq.clustering import KMeans

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
        vae_path: str,
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
        class_head : partial
            The class head to be used for the probe.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ctxt_dim = data_sample["jets"].shape[-1]
        self.n_classes = n_classes
        self.cemb_dim = 32  # Hardcoded for now
        self.do_kmeans = vae_path is None

        # The transformer encoder
        encoder_config["num_registers"] = 0  # GPT does not use registers
        encoder_config["max_seq_len"] = self.num_csts + 1  # One for the start token
        encoder_config["do_absolute_enc"] = True  # Needs positional encodings!
        self.encoder = Transformer(**encoder_config, ctxt_dim=self.cemb_dim)
        self.outp_dim = self.encoder.outp_dim

        # The embedding layers
        self.csts_emb = nn.Linear(self.csts_dim, self.encoder.dim)
        self.jets_emb = nn.Linear(self.ctxt_dim, self.cemb_dim)
        self.csts_id_emb = nn.Embedding(CSTS_ID, self.encoder.dim)

        # Add the learnable start token
        self.start_token = nn.Parameter(T.randn((1, 1, self.outp_dim)) * 1e-3)

        # Load the VAE model and freeze it
        if self.do_kmeans:
            self.kmeans = KMeans(8192, max_iter=500, verbose=10)
            self.kmeans.centroids = T.zeros((self.csts_dim, self.kmeans.n_clusters))
            self.end_clus_id = self.kmeans.n_clusters
        else:
            self.vae = T.load(vae_path, map_location="cpu")
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.end_clus_id = self.vae.quantizer.codebook_size

        # Load the objective heads with extra class for the end token
        self.head = nn.Linear(self.outp_dim, self.end_clus_id + 1)
        self.id_head = nn.Linear(self.outp_dim, CSTS_ID + 1)
        self.end_id = CSTS_ID

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        jets = data["jets"]
        labels = data["labels"]

        # Used for adding the end tokens to the labels
        n_csts = mask.sum(dim=1)[:, None]
        wstart = F.pad(mask, (1, 0), value=True)

        # Embed
        x = self.csts_emb(csts) + self.csts_id_emb(csts_id)
        ctxt = self.jets_emb(jets)

        # Add positional encodings and append the start token
        st = self.start_token.expand(x.size(0), 1, -1)
        x = T.cat([st, x], dim=1)

        # Pass through the encoder with a causal mask
        enc_out = self.encoder(x, mask=wstart, ctxt=ctxt, causal=True)

        # Get the ID loss
        csts_id = F.pad(csts_id, (0, 1)).scatter_(1, n_csts, self.end_id)
        id_out = self.id_head(enc_out)
        id_loss = F.cross_entropy(id_out[wstart], csts_id[wstart], label_smoothing=0.1)
        self.log(f"{prefix}/id_loss", id_loss)

        # Pass through the vae or kmeans to get the targets
        if self.do_kmeans:
            target = self.kmeans.predict(csts[mask].T.contiguous()).long()
        else:
            target = self.vae(csts, mask=mask)[0].long()

        # Get the loss for the VAE/KMeans
        clus_id = T.zeros((csts.shape[:2]), dtype=T.long, device=x.device)
        clus_id[mask] = target
        clus_id = F.pad(clus_id, (0, 1)).scatter_(1, n_csts, self.end_clus_id)
        clus_out = self.head(enc_out)
        clus_loss = F.cross_entropy(
            clus_out[wstart], clus_id[wstart], label_smoothing=0.1
        )
        self.log(f"{prefix}/clus_loss", clus_loss)

        # Run the probe to evaluate the embedding
        probe_loss = T.tensor(0, device=csts.device, dtype=clus_loss.dtype)
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(enc_out, mask=wstart)
            probe_loss = F.cross_entropy(class_out, labels)
            acc = getattr(self, f"{prefix}_acc")
            acc(class_out, labels)
            self.log(f"{prefix}/probe_loss", probe_loss)
            self.log(f"{prefix}/probe_accuracy", acc)

        # Combine and return the losses
        total_loss = id_loss + clus_loss + probe_loss
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
            jets_emb=self.jets_emb,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")

    def on_fit_start(self) -> None:
        """At the start of the fit, fit the kmeans."""
        if not self.do_kmeans:
            return

        # Skip kmeans has already been initialised (e.g. from a checkpoint)
        if self.kmeans.centroids is not None and (self.kmeans.centroids != 0).any():
            return

        # Load the first 50 batches of training data
        csts = []
        mask = []
        loader = self.trainer.train_dataloader
        if loader is None:
            self.trainer.fit_loop.setup_data()
            loader = self.trainer.train_dataloader
        for i, data in enumerate(loader):
            csts.append(data["csts"])
            mask.append(data["mask"])
            if i > 50:
                break
        csts = T.vstack(csts).to(self.device)
        mask = T.vstack(mask).to(self.device)

        # Fit the kmeans
        inputs = csts[mask].T.contiguous()
        self.kmeans.fit(inputs)
