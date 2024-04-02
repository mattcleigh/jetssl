from copy import deepcopy
from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    log_softmax,
    normalize,
    pairwise_distance,
    softmax,
)
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.modules import IterativeNormLayer
from mltools.mltools.torch_utils import ema_param_sync, set_eval
from mltools.mltools.transformers import Transformer
from src.models.utils import JetBackbone

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


def dino_loss(
    t_out: T.Tensor,
    s_out: T.Tensor,
    t_temp: float,
    s_temp: float,
    center: T.Tensor,
) -> T.Tensor:
    """Calculate the loss used in DINO-v1 including centering and sharpening."""
    t_out = t_out.detach() - center
    s_out = log_softmax(s_out / s_temp, dim=-1)  # log softmax improves stability
    t_out = softmax(t_out / t_temp, dim=-1)
    return -(t_out * s_out).sum(dim=-1).mean()


@T.autocast("cuda", enabled=False)
def koleo_loss(x: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """
    # Normalize the input
    x = normalize(x, eps=eps, dim=-1)

    # Calculate the matching pair idxes via the max inner produce
    with T.no_grad():
        dots = T.mm(x, x.t())
        dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)  # Fill the diagonal with -1
        min_idx = T.argmax(dots, dim=1)

    # Get the distance between closest pairs
    distances = pairwise_distance(x, x[min_idx])

    # Return the kozachenko-leonenko entropy
    return -T.log(distances + eps).mean()


def marginal_regularise(x: T.Tensor, temp: float) -> T.Tensor:
    """Regularise the total entropy of a batch output."""
    marginal = softmax(x / temp, dim=-1).mean(dim=0)
    return -(marginal * T.log(marginal + 1e-6)).sum()


class JetDINO(pl.LightningModule):
    """Dino-v2 (really iBOT) model for jets."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        encoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        class_head: partial,
        t_temp: float = 0.05,
        s_temp: float = 0.1,
        t_ema: float = 0.995,
        c_ema: float = 0.9,
        reg_strenth: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]

        # The normalisation layer (for the continuous features only)
        self.normaliser = IterativeNormLayer(self.csts_dim)

        # The transformer encoder for the constituents
        self.encoder = Transformer(**encoder_config)
        self.output_dim = self.encoder.dim  # For fine tuning

        # The different linear embedding layers
        self.csts_id_embedder = nn.Embedding(CSTS_ID, self.encoder.dim)
        self.ctst_embedder = nn.Linear(self.csts_dim, self.encoder.dim)

        # The projection layer into the contrastive space
        self.projector = nn.Linear(self.output_dim, self.output_dim)

        # Make a copy of the full network for the teacher
        self.t_csts_id_embedder = deepcopy(self.csts_id_embedder)
        self.t_ctst_embedder = deepcopy(self.ctst_embedder)
        self.t_encoder = deepcopy(self.encoder)
        self.t_projector = deepcopy(self.projector)
        self.t_csts_id_embedder.requires_grad_(False)
        self.t_ctst_embedder.requires_grad_(False)
        self.t_encoder.requires_grad_(False)
        self.t_projector.requires_grad_(False)

        # The learnable null token and positional encoding
        self.null_token = nn.Parameter(T.randn(self.output_dim) * 1e-3)
        self.pos_enc = nn.Parameter(T.randn((self.num_csts, self.output_dim)) * 1e-3)

        # Dino parameters
        self.t_temp = t_temp
        self.s_temp = s_temp
        self.t_ema = t_ema
        self.c_ema = c_ema
        self.reg_strenth = reg_strenth
        self.register_buffer("center", T.zeros(1, self.output_dim))

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

        # Normalise the continuous features and update the stats
        normed_csts = self.normaliser(csts, mask)

        # Pass through the student model with dropped nodes under both configs
        x_s1 = self.mask_and_encode(normed_csts, csts_id, mask, null_mask)
        x_s2 = self.mask_and_encode(normed_csts, csts_id, mask, ~null_mask & mask)

        # Expand the null mask to also include the registers in the loss
        exp_null_mask = self.encoder.get_combined_mask(null_mask)
        exp_inv_mask = self.encoder.get_combined_mask(~null_mask & mask)

        # Pass through the teacher model without dropping
        with set_eval(self), T.no_grad():
            x_t, t_mask = self.pass_teacher(normed_csts, csts_id, mask)
            x_t = self.t_projector(x_t)

        # Use each combination to get the loss with the register nodes
        loss = 0.5 * (
            dino_loss(
                x_t[exp_null_mask],
                x_s1[exp_null_mask],
                self.t_temp,
                self.s_temp,
                self.center,
            )
            + dino_loss(
                x_t[exp_inv_mask],
                x_s2[exp_inv_mask],
                self.t_temp,
                self.s_temp,
                self.center,
            )
        )

        # Regularise using the koleo loss of the student with all inputs
        x_s, s_mask = self.pass_student(normed_csts, csts_id, mask)
        x_s = self.projector(x_t)
        reg = koleo_loss(x_s[s_mask])

        # Combine the losses and regularisations
        total_loss = loss + reg * self.reg_strenth

        # Do an intervention for the loss sometimes being NaN
        if T.isnan(total_loss).any():
            print("Intervention!")
            total_loss = T.nan_to_num(total_loss)

        # Perform the ema updates (training only)
        if self.training:
            self.center = self.center * self.c_ema + x_t[t_mask].mean(dim=0) * (
                1 - self.c_ema
            )
            ema_param_sync(self.csts_id_embedder, self.t_csts_id_embedder, self.t_ema)
            ema_param_sync(self.ctst_embedder, self.t_ctst_embedder, self.t_ema)
            ema_param_sync(self.encoder, self.t_encoder, self.t_ema)
            ema_param_sync(self.projector, self.t_projector, self.t_ema)

        # Dont run probe too often as we must be fair to MPM!
        if batch_idx % 10 == 0 or prefix == "valid":
            class_out = self.classifier_head(x_t.detach(), kv_mask=t_mask)
            probe_loss = cross_entropy(class_out, labels)
            total_loss += probe_loss
            self.acc(class_out, labels)  # Updates internal state
            self.log(f"{prefix}/probe_loss", probe_loss)
            self.log(f"{prefix}/accuracy", self.acc)

        # Log the metrics
        self.log(f"{prefix}/total_loss", loss)
        self.log(f"{prefix}/reg_loss", reg)

        return total_loss

    def mask_and_encode(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.Tensor,
    ) -> tuple:
        """Drop the input nodes and pass through the encoder."""
        # Pass the inputs through their respective embedding layers and sum
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)

        # Replace all null nodes with the same null token
        x[null_mask] = self.null_token.type(x.dtype)

        # Calculate the sorted positional encoding for the dropped nodes
        pos_enc = self.pos_enc[: x.size(1)].unsqueeze(0).expand_as(x)
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Give positional encoding to the inputs
        x[null_mask] = x[null_mask] + pos_enc[null_sorted].type(x.dtype)

        # Pass through the encoder (might gain registers)
        x = self.encoder(x, kv_mask=mask)
        return self.projector(x)  # Project into the contrastive space

    def pass_teacher(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Get the outputs of the teacher."""
        x = self.t_ctst_embedder(csts) + self.t_csts_id_embedder(csts_id)
        x = self.t_encoder(x, kv_mask=mask)
        new_mask = self.t_encoder.get_combined_mask(mask)
        return x, new_mask

    def pass_student(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
    ) -> T.Tensor:
        """Get the outputs of the student."""
        x = self.ctst_embedder(csts) + self.csts_id_embedder(csts_id)
        x = self.encoder(x, kv_mask=mask)
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
        """Create the pickled object for the backbone out of the teacher components."""
        backbone = JetBackbone(
            normaliser=self.normaliser,
            ctst_embedder=self.t_ctst_embedder,
            id_embedder=self.t_csts_id_embedder,
            encoder=self.t_encoder,
        )
        backbone.eval()
        T.save(backbone, "backbone.pkl")
