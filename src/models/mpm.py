from functools import partial
from typing import Literal

import pytorch_lightning as pl
import torch as T
from torch import nn

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class MaskedParticleModelling(pl.LightningModule):
    """Class for all masked particle modelling pre-training.

    Is either setup as a BERT style encoder only or a MAE with a decoder.
    List of tasks defines the various masked objectives to be used.
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
        tasks: dict,
        objective: Literal["mae", "bert"] = "mae",
        use_id: bool = True,
        use_hlv: bool = False,
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
        decoder_config : dict
            The configuration for the decoder transformer.
        optimizer : partial
            The optimizer to be used.
        scheduler : dict
            The scheduler to be used.
        tasks : dict
            A dictionary of tasks to be used. Sould be a list of partials.
        objective : str, optional
            The type of objective to be used, by default "mae".
            Can be "mae" or "bert"
        use_id : bool, optional
            Whether to include the ID information in the network inputs,
            by default True.
        use_hlv : bool, optional
            Whether to use the HLV-Jet information as context, by default False.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Make sure the tasks dict is not empty
        assert tasks, "No tasks were provided."

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.ctxt_dim = data_sample["jets"].shape[-1] if use_hlv else 0

        # Attributes
        self.objective = objective
        self.use_id = use_id
        self.use_hlv = use_hlv
        self.n_classes = n_classes
        self.cemb_dim = 32 if use_hlv else 0  # Hardcoded for now

        # Hack but it turns out that the flow training is more unstable than the others
        if "flow" in tasks:
            encoder_config["do_final_norm"] = True

        # The transformer encoder
        self.encoder = Transformer(**encoder_config, ctxt_dim=self.cemb_dim)

        # The decoder used for the MAE objective (no positional encoding)
        if self.objective == "mae":
            self.decoder = Transformer(**decoder_config)
            self.enc_to_dec = nn.Linear(self.encoder.dim, self.decoder.dim)

        # The embedding layers
        self.csts_emb = nn.Linear(self.csts_dim, self.encoder.dim)
        self.jets_emb = nn.Linear(self.ctxt_dim, self.cemb_dim) if use_hlv else None
        self.csts_id_emb = nn.Embedding(CSTS_ID, self.encoder.dim) if use_id else None

        # The output dimension (input for the tasks)
        self.outp_dim = self.decoder.dim if objective == "mae" else self.encoder.dim

        # The learnable parameters for the dropped nodes in the decoder (1 per seq)
        self.null_token = nn.Parameter(T.randn((self.num_csts, self.outp_dim)) * 1e-3)

        # Initialise each of the tasks
        self.tasks = nn.ModuleDict({k: v(self, name=k) for k, v in tasks.items()})
        self.on_validation_epoch_end()

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Pass through the model using the appropriate method
        data["outputs"] = self.apply_pass(data)

        # Calculate the losses per task and log
        loss = T.tensor(0.0, device=self.device)
        for task in self.tasks.values():
            loss = loss + task.get_loss(self, data, batch_idx, prefix)
        self.log(f"{prefix}/total_loss", loss)

        # Call the visualisation method for each task
        if prefix == "valid" and batch_idx == 0:
            for task in self.tasks.values():
                task.visualise(self, data)

        return loss

    def apply_pass(self, data: dict) -> T.Tensor:
        """Apply the correct pass through the model."""
        if self.objective == "mae":
            return self.mae_pass(data)
        if self.objective == "bert":
            return self.bert_pass(data)
        raise ValueError(f"Objective {self.objective} not recognised.")

    def mae_pass(self, data: dict) -> T.Tensor:
        """Pass through the masked autoencoder using and get the decoder outputs."""
        # Unpack the inputs
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        null_mask = data["null_mask"]
        jets = data["jets"]

        # Embed the inputs
        x = self.csts_emb(csts)
        if self.use_id:
            x = x + self.csts_id_emb(csts_id)
        ctxt = self.jets_emb(jets) if self.use_hlv else None

        # Pass through the encoder (might gain registers)
        x = self.encoder(x, mask=mask & ~null_mask, ctxt=ctxt)
        mask = self.encoder.get_combined_mask(mask)

        # Resize to the decoder and store the number of registers
        dec_inpts = self.enc_to_dec(x)
        n_reg = self.encoder.num_registers

        # Trim the null tokens to seq_len and expand to match batch size
        nt = self.null_token[: null_mask.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Insert the null tokens so they are ordered wrt each other
        dec_inpts[:, n_reg:][null_mask] = nt[null_sorted].type(dec_inpts.dtype)

        # Pass through the decoder dont need registers after
        return self.decoder(dec_inpts, mask=mask)[:, n_reg:]

    def bert_pass(self, data: dict) -> T.Tensor:
        """Pass through the encoder only with the null tokens."""
        # Unpack the inputs
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        null_mask = data["null_mask"]
        jets = data["jets"]

        # Embed the inputs
        x = self.csts_emb(csts)
        if self.use_id:
            x = x + self.csts_id_emb(csts_id)
        ctxt = self.jets_emb(jets) if self.use_hlv else None

        # Trim the null tokens to seq_len and expand to match batch size
        nt = self.null_token[: x.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Insert the null tokens so they are ordered wrt each other
        x[null_mask] = nt[null_sorted].type(x.dtype)

        # Pass through the encoder dont need registers after
        n_reg = self.encoder.num_registers
        return self.encoder(x, mask=mask, ctxt=ctxt)[:, n_reg:]

    def forward(self, data: dict) -> T.Tensor:
        """Full forward pass for inference without null tokens."""
        # Unpack the inputs
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        jets = data["jets"]

        x = self.csts_emb(csts)
        if self.use_id:
            x = x + self.csts_id_emb(csts_id)
        ctxt = self.jets_emb(jets) if self.use_hlv else None

        x = self.encoder(x, mask=mask, ctxt=ctxt)
        new_mask = self.encoder.get_combined_mask(mask)
        return x, new_mask

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "train")

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_fit_start(self) -> None:
        """Call the on_fit_start method for each task."""
        for task in self.tasks.values():
            task.on_fit_start(self)

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
