from functools import partial

import pytorch_lightning as pl
import torch as T
from torch import nn
from torch.nn.functional import (
    cross_entropy,
    smooth_l1_loss,
)
from torchmetrics import Accuracy

from mltools.mltools.lightning_utils import simple_optim_sched
from mltools.mltools.torch_utils import ema_param_sync, set_eval
from mltools.mltools.transformers import Transformer
from src.models.utils import JetEncoder


class VarCovRegLoss(nn.Module):
    """Variance-Covariance regularisation loss.

    From the VICReg paper: https://arxiv.org/pdf/2105.04906.pdf
    """

    def __init__(
        self,
        inpt_dim: int,
        expand_dim: int = 8192,
        std_weight: float = 1,
        cov_weight: float = 0.04,  # Paper used ratio of 25:1
    ) -> None:
        super().__init__()
        self.std_weight = std_weight
        self.cov_weight = cov_weight
        self.fn = nn.Sequential(
            nn.Linear(inpt_dim, (expand_dim + inpt_dim) // 2),
            nn.SiLU(),
            nn.Linear((expand_dim + inpt_dim) // 2, expand_dim),
        )

    def forward(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        """Calculate the loss."""
        # Map to the expanded space
        z = self.fn(x[mask])
        _N, D = z.shape

        # Calculate the variance loss
        std = (z.var(dim=0) + 1e-4).sqrt()
        std_loss = (T.relu(1 - std)).mean()  # Clamp as we only want to penalise low std

        # Calculate the covariance loss
        cov = T.cov(z.T)
        cov.diagonal().fill_(0)  # Remove the diagonal
        cov_loss = cov.square().sum().div(D)

        # Return the weighted sum
        total = self.std_weight * std_loss + self.cov_weight * cov_loss
        return total, std_loss, cov_loss


class JEPAPredictor(nn.Module):
    """Predictor for the JEPA model."""

    def __init__(self, inpt_dim: int, **kwargs) -> None:
        super().__init__()

        # The transformer encoder for the constituents
        self.encoder = Transformer(**kwargs)
        self.dim = self.encoder.dim

        # The input and output projection layers
        self.input_proj = nn.Linear(inpt_dim, self.dim)
        self.output_proj = nn.Linear(self.dim, inpt_dim)

        # The tokens for the null and target nodes
        self.null_token = nn.Parameter(T.randn((1, 1, self.dim)) * 1e-3)
        self.target_token = nn.Parameter(T.randn((1, 1, self.dim)) * 1e-3)

    def forward(
        self,
        x: T.Tensor,
        mask: T.BoolTensor,
        null_mask: T.BoolTensor,
        target_mask: T.BoolTensor,
    ) -> T.Tensor:
        """Pass through the predictor.

        Parameters
        ----------
        x : T.Tensor
            The embedded jet constituents. The output of the student encoder.
            May contain cls tokens and registers.
        mask : T.BoolTensor
            The mask for the input. Which nodes are real (T) and which are padding (F).
        null_mask : T.BoolTensor
            A mask which tells us which nodes were hidden from the student.
            Allows us to parameterise the predictor with the masking.
        target_mask : T.BoolTensor
            The mask for the target. Which of the teacher's nodes are we tryuing to
            predict.
        """
        # Embed the nodes into the predictor space
        x = self.input_proj(x)

        # The predictor needs to know:
        # 1) Which nodes were hidden from the student (augmentation)
        null_tokens = self.null_token.expand(*null_mask.shape, -1)
        null_tokens = null_tokens * null_mask.unsqueeze(-1)
        # 2) Which nodes are the target
        targ_tokens = self.target_token.expand(*target_mask.shape, -1)
        targ_tokens = targ_tokens * target_mask.unsqueeze(-1)

        # Add to the sequence part of x (not the registers)
        S = null_mask.shape[1]
        x[:, -S:] = x[:, -S:] + null_tokens + targ_tokens

        # Pass through the transformer
        x = self.encoder(x, mask=mask)

        # We only need the target nodes
        x = x[:, -S:][target_mask]

        # Project back to the original space
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
        t_ema: float = 0.992,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Attributes
        self.num_csts = data_sample["csts"].shape[0]
        self.csts_dim = data_sample["csts"].shape[-1]
        self.t_ema = t_ema

        # The student and teacher models
        self.student = JetEncoder(csts_dim=self.csts_dim, encoder_config=encoder_config)
        self.teacher = JetEncoder(csts_dim=self.csts_dim, encoder_config=encoder_config)
        self.teacher.requires_grad_(False)  # Teacher is an EMA of student

        # Save the embedding dimension
        self.outp_dim = self.student.outp_dim

        # The predictor for mapping between the student and teacher spaces
        self.predictor = JEPAPredictor(self.outp_dim, **predictor_config)

        # Regularisation loss for the predictor to prevent collapse
        self.reg_loss = VarCovRegLoss(self.outp_dim)

        # Basic classifier and accuracy for evaluating encoder
        self.class_head = class_head(inpt_dim=self.outp_dim, outp_dim=n_classes)
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

        # Make sure that the target mask is a subset of the null mask
        # (dont predict it if it wasnt hidden in the first place)
        target_mask[~null_mask] = False

        # Pass the inputs through the student model, masking some nodes
        s_out, s_mask = self.student(csts, csts_id, mask & ~null_mask)

        # Pass the inputs through the teacher model, without the null mask
        with set_eval(self), T.no_grad():
            t_out, t_mask = self.teacher(csts, csts_id, mask)
            S = csts.shape[1]  # Number of nodes
            target = t_out[:, -S:]  # Strip the cls/register tokens
            target = target[target_mask]  # Only need the target nodes

        # Pass the student outputs through the predictor (t_mask includes registers)
        p_out = self.predictor(s_out, s_mask, null_mask, target_mask)

        # Calculate the prediction losses for the target mask and log
        # Official JEPA code uses smooth_l1_loss over mse
        pred_loss = smooth_l1_loss(p_out, target)

        # Calculate the regularisation loss for the predictor
        reg_loss, std_loss, cov_loss = self.reg_loss(s_out, s_mask)

        # Perform the ema updates (while training only)
        if self.training:
            ema_param_sync(self.student, self.teacher, self.t_ema)

        # Run the probe to evaluate the embedding using the teacher's output
        # In MPM we typically do it every 50 batches.
        if batch_idx % 50 == 0 or prefix == "valid":
            class_out = self.class_head(t_out.detach(), mask=t_mask.detach())
            probe_loss = cross_entropy(class_out, labels)
            self.acc(class_out, labels)  # Updates internal state
            self.log(f"{prefix}/probe_accuracy", self.acc)
        else:
            probe_loss = T.zeros(1, device=self.device)

        # Combine and log the losses
        total_loss = pred_loss + reg_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        self.log(f"{prefix}/pred_loss", pred_loss)
        self.log(f"{prefix}/std_loss", std_loss)
        self.log(f"{prefix}/cov_loss", cov_loss)
        self.log(f"{prefix}/probe_loss", probe_loss)
        return total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "train")

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        return self._shared_step(sample, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        """Use the mltools optimiser and scheduler."""
        return simple_optim_sched(self)

    def on_train_epoch_end(self) -> None:
        """Create the pickled object for the backbone out of the teacher components."""
        self.teacher.eval()
        T.save(self.teacher, "backbone.pkl")
