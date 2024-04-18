from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
)
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.torch_utils import ema_param_sync, set_eval
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class JEPAPredictor(nn.Module):
    """Predictor for the JEPA model."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim

        # The transformer encoder for the constituents
        self.encoder = Transformer(**kwargs)
        dim = self.encoder.dim

        # The input and output projection layers
        self.input_proj = nn.Linear(inpt_dim, dim)
        self.output_proj = nn.Linear(dim, inpt_dim)

        # The tokens for the null and target nodes, includes positional encoding
        self.null_token = nn.Parameter(T.randn((1, 1, dim)) * 1e-3)
        self.target_token = nn.Parameter(T.randn((1, 1, dim)) * 1e-3)

    def forward(
        self,
        x: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.BoolTensor,
        target_mask: T.BoolTensor,
        n_reg: int,
    ) -> T.Tensor:
        """Pass through the predictor.

        Parameters
        ----------
        x : T.Tensor
            The embedded jet constituents. The output of the student encoder.
        mask : T.BoolTensor
            The mask for the input. Which nodes are real (T) and which are padding (F).
        null_mask : T.BoolTensor
            A mask which tells us which nodes were hidden from the student.
            Allows us to parameterise the predictor with the masking.
        target_mask : T.BoolTensor
            The mask for the target. Which of the teacher's nodes are we tryuing to
            predict.
        n_reg : int
            The number of registers that have been added to the front of x.
        """
        # Embed the nodes into the predictor space
        x = self.input_proj(x)

        # Via adding the null and target tokens we parameterise the predictor into
        # Which nodes were hidden from the student (augmentation)
        # Which nodes are the target
        b_size = x.size(0)
        n_seq = x.size(1) - n_reg
        null_tokens = self.null_token.expand(b_size, n_seq, -1)
        targ_tokens = self.target_token.expand(b_size, n_seq, -1)
        null_tokens = null_tokens * null_mask.unsqueeze(-1)
        targ_tokens = targ_tokens * target_mask.unsqueeze(-1)

        # Add to the sequence part of x (not the registers)
        x[:, n_reg:] = x[:, n_reg:] + null_tokens + targ_tokens

        # Pass through the transformer after dropping all the null nodes
        x = self.encoder(x, mask=mask)
        return self.output_proj(x)


class JetJEPA(pl.LightningModule):
    """JEPA for running on jets."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        predictor_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        t_ema: float = 0.990,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # The transformer encoder for the constituents (And the teacher version)
        self.encoder = Transformer(**encoder_config)
        self.t_encoder = Transformer(**encoder_config)
        self.n_reg = self.encoder.num_registers

        # The different linear embedding layers
        self.csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)
        self.t_csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.t_ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)

        # The predictor for mapping between the student and teacher spaces
        self.predictor = JEPAPredictor(
            inpt_dim=self.encoder.dim,
            outp_dim=self.encoder.dim,
            **predictor_config,
        )

        # Turn off gradients for teacher components as it "learns" via EMA
        self.t_ema = t_ema
        self.t_csts_id_embedder.requires_grad_(False)
        self.t_ctst_embedder.requires_grad_(False)
        self.t_encoder.requires_grad_(False)

        # Save the output dimensions
        self.output_dim = self.encoder.dim

        # The learnable null token (unique for positional encoding)
        self.null_token = nn.Parameter(T.randn((self.num_csts, self.output_dim)) * 1e-3)

        # Basic classifier and accuracy for evaluating encoder
        self.classifier_head = class_head(inpt_dim=self.encoder.dim, outp_dim=n_classes)
        self.acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, sample: T.Tensor, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        # Unpack the sample
        csts = sample["csts"]
        csts_id = sample["csts_id"]
        labels = sample["labels"]
        mask = sample["mask"]
        null_mask = sample["null_mask"]
        target_mask = sample["target_mask"]

        # Pass the inputs through the student model, masking some nodes
        s_out, s_mask = self.mask_and_encode(csts, csts_id, mask, null_mask)

        # Pass the outputs through the predictor
        p_out = self.predictor(s_out, s_mask, null_mask, target_mask, self.n_reg)

        # Pass through the teacher model without dropping
        with set_eval(self), T.no_grad():
            t_out, t_mask = self.pass_teacher(csts, csts_id, mask)

        # Calculate the prediction losses for the target mask
        loss = (
            (p_out[:, self.n_reg :][target_mask] - t_out[:, self.n_reg :][target_mask])
            .square()
            .mean()
        )
        self.log(f"{prefix}/total_loss", loss)

        # Perform the ema updates (while training only)
        if self.training:
            ema_param_sync(self.csts_id_embedder, self.t_csts_id_embedder, self.t_ema)
            ema_param_sync(self.ctst_embedder, self.t_ctst_embedder, self.t_ema)
            ema_param_sync(self.encoder, self.t_encoder, self.t_ema)

        # Run a detached probe every 50 batches to evaluate the encoder
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.classifier_head(t_out.detach(), mask=t_mask.detach())
            probe_loss = cross_entropy(class_out, labels)
            loss += probe_loss
            self.acc(class_out, labels)  # Updates internal state
            self.log(f"{prefix}/probe_loss", probe_loss)
            self.log(f"{prefix}/probe_accuracy", self.acc)

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

        # Pass through the encoder while masking away null_nodes (might gain registers)
        x = self.encoder(x, mask=mask & ~null_mask)
        mask = self.encoder.get_combined_mask(mask)

        return x, mask

    def pass_teacher(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Get the outputs of the teacher."""
        x = self.t_ctst_embedder(csts) + self.t_csts_id_embedder(csts_id)
        x = self.t_encoder(x, mask=mask)
        mask = self.t_encoder.get_combined_mask(mask)
        return x, mask

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_train_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        backbone = JetBackbone(
            csts_emb=self.t_ctst_embedder,
            csts_id_emb=self.t_csts_id_embedder,
            encoder=self.t_encoder,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
