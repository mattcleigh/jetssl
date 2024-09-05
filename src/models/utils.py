from copy import deepcopy

import torch as T
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from torchdiffeq import odeint

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import CosineEncodingLayer
from mltools.mltools.torch_utils import ema_param_sync
from mltools.mltools.transformers import Transformer

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class JetBackbone(nn.Module):
    """Generalised backbone for the jet models.

    Simply wraps the constituent embedding, constituent id embedding and encoder
    together in a single module.
    Easy for saving and loading using the pickle module
    """

    def __init__(
        self,
        csts_emb: nn.Module,
        csts_id_emb: nn.Module | None,
        encoder: nn.Module,
        ctxt_emb: nn.Module | None = None,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.csts_emb = csts_emb
        self.csts_id_emb = csts_id_emb
        self.encoder = encoder
        self.ctxt_emb = ctxt_emb
        self.is_causal = is_causal

    @property
    def dim(self) -> int:
        return self.encoder.dim

    @property
    def outp_dim(self) -> int:
        return self.encoder.outp_dim

    def forward(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
        jets: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the complete backbone."""
        x = self.csts_emb(csts)
        if self.csts_id_emb is not None:
            x = x + self.csts_id_emb(csts_id)

        # Need the hasattr check as the older pickled models dont have this
        ctxt = (
            self.ctxt_emb(jets)
            if hasattr(self, "ctxt_emb") and self.ctxt_emb is not None
            else None
        )
        x = self.encoder(x, mask=mask, ctxt=ctxt, causal=self.is_causal)
        new_mask = self.encoder.get_combined_mask(mask)
        return x, new_mask


class JetEncoder(JetBackbone):
    """Generalised transformer encoder for the jets.

    Same as above but we initialise via configs, not modules.
    """

    def __init__(
        self,
        *,
        csts_dim: int,
        encoder_config: dict,
        use_csts_id: bool = True,
        is_causal: bool = False,
        ctxt_dim: int = 0,
        ctxt_config: dict | None = None,
    ) -> None:
        if ctxt_dim:
            ctxt_emb = MLP(ctxt_dim, **(ctxt_config or {}))
            ctxt_dim = ctxt_emb.outp_dim
        else:
            ctxt_emb = None
        encoder = Transformer(**encoder_config, ctxt_dim=ctxt_dim)
        csts_emb = nn.Linear(csts_dim, encoder.dim)
        csts_id_emb = nn.Embedding(CSTS_ID, encoder.dim) if use_csts_id else None
        super().__init__(csts_emb, csts_id_emb, encoder, ctxt_emb, is_causal)


class VectorDiffuser(nn.Module):
    """Flow-Matching MLP for generating a single vector."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        ctxt_dim: int,
        time_dim: int = 8,
        mlp_config: dict,
    ) -> None:
        super().__init__()
        self.time_encoder = CosineEncodingLayer(inpt_dim=1, encoding_dim=time_dim)
        self.mlp = MLP(
            inpt_dim=inpt_dim,
            outp_dim=inpt_dim,
            ctxt_dim=ctxt_dim + time_dim,
            **mlp_config,
        )
        self.ema_mlp = deepcopy(self.mlp)
        self.ema_mlp.requires_grad_(False)

    def forward(
        self, xt: T.Tensor, t: T.Tensor, ctxt: T.Tensor, use_ema: bool = False
    ) -> T.Tensor:
        """Get the fully denoised estimate."""
        c = T.cat([self.time_encoder(t), ctxt], dim=-1)
        mlp = self.ema_mlp if use_ema else self.mlp
        return mlp(xt, c)

    def get_loss(self, x0: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        t = T.sigmoid(T.randn(x0.shape[0], 1, device=x0.device))
        x1 = T.randn_like(x0)
        xt = (1 - t) * x0 + t * x1
        v = self.forward(xt, t, ctxt)

        if self.training:
            ema_param_sync(self.mlp, self.ema_mlp, 0.999)

        return (v - (x1 - x0)).square().mean()

    # Turn off autocast
    @T.autocast("cuda", enabled=False)  # Dont autocast during integration
    @T.autocast("cpu", enabled=False)
    def generate(self, x1: T.Tensor, ctxt: T.Tensor, times: T.Tensor) -> T.Tensor:
        """Generate a sample."""

        def ode_fn(t, xt):
            t = t * xt.new_ones([xt.shape[0], 1])
            return self.forward(xt, t, ctxt, use_ema=True)

        return odeint(ode_fn, x1, times, method="midpoint")[-1]


def minimize_padding(x: T.Tensor, mask: T.BoolTensor) -> tuple:
    """Minimise the padding of a batched tensor."""
    # Calculate the minimum mask required per jet
    max_csts = mask.sum(axis=-1).max()

    # Check if the mask is already minimal
    if max_csts == mask.shape[-1]:
        return x, mask

    # Get the array that sorts the mask and expand it to x shape
    sort_mask = T.argsort(-mask.float(), dim=-1)[:, :max_csts]  # CUDA cant sort bools
    sort_x = sort_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])

    # Use gather to get the new mask and x
    mask = T.gather(mask, 1, sort_mask)
    x = T.gather(x, 1, sort_x)

    return x, mask


class LinearHead(nn.Module):
    """Very basic linear pooling head."""

    def __init__(self, inpt_dim: int, outp_dim: int, nonlin: bool = False) -> None:
        super().__init__()
        self.lin1 = nn.Linear(inpt_dim, inpt_dim)
        self.lin2 = nn.Linear(inpt_dim, outp_dim)
        self.nonlin = nonlin

    def forward(self, x: T.Tensor, mask: T.BoolTensor | None = None) -> T.Tensor:
        x = self.lin1(x)
        if self.nonlin:
            x = F.silu(x)
        if mask is None:
            return self.lin2(x.mean(dim=1))
        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.lin2(x)


def varcov_loss(
    x: T.Tensor,
    std_weight: float = 1,
    cov_weight: float = 0.01,
) -> tuple:
    """Variance-Covariance regularisation loss.

    From the VICReg paper: https://arxiv.org/pdf/2105.04906.pdf
    We also use the default weights from the paper when they applied it to BYOL
    """
    _N, D = x.shape

    # Calculate the variance loss
    std = (x.var(dim=0) + 1e-4).sqrt()
    std_loss = (T.relu(1 - std)).mean()  # Clamp as we only want to penalise low std

    # Calculate the covariance loss
    cov = T.cov(x.T)
    cov.diagonal().fill_(0)
    cov_loss = cov.square().sum().div(D)

    # Return the weighted sum
    total = std_weight * std_loss + cov_weight * cov_loss
    return total, std_loss, cov_loss


class DINOHead(nn.Module):
    """The projection head for DINO-v2.

    Adapted from:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        bottleneck_dim: int = 0,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            inpt_dim=inpt_dim,
            outp_dim=bottleneck_dim or inpt_dim // 4,
            hddn_dim=bottleneck_dim or inpt_dim // 4,
            num_blocks=1,
            act_h="SiLU",
            act_o="SiLU",
        )
        self.apply(self.reset_params)
        self.last_layer = weight_norm(nn.Linear(self.mlp.outp_dim, outp_dim))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def reset_params(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == T.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


@T.no_grad()
def sk_center(x: T.Tensor, temp: float) -> T.Tensor:
    """Apply sinkhorn-Knopp centering, ensures that rows and columns sum to 1."""
    Q = T.exp(x.float() / temp)
    B = Q.shape[0]  # batch size
    K = Q.shape[1]  # number of prototypes / classes / dimension of output
    Q /= Q.sum()
    for _ in range(3):
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize the columns
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True)  # Normalize the rows
        Q /= B
    Q *= B
    return Q


def dinov2_loss(
    s_out: T.Tensor,
    t_out: T.Tensor,
    s_temp: float = 0.1,
    t_temp: float = 0.05,
) -> T.Tensor:
    t_centered = sk_center(t_out, t_temp)
    s_lsm = F.log_softmax(s_out / s_temp, dim=-1)
    loss = -(t_centered * s_lsm).sum(dim=-1)
    loss = T.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()


@T.no_grad
@T.cuda.amp.custom_fwd(cast_inputs=T.float32)
def repulse(x: T.Tensor, num_iter: int = 1):
    """Run the repulsion algorithm for a number of iterations."""
    for _ in range(num_iter):
        r = T.randn_like(x) * 1e-6
        x = F.normalize(x + r)
        a = T.cdist(x, x)
        a.add_(1e-12).reciprocal_().fill_diagonal_(0).clamp_max_(1e6)
        b = x * a.sum(1, keepdim=True)
        c = T.mm(a, x)
        x = b - c
    return F.normalize(x)


def repulse_loss(s_out: T.Tensor, t_out: T.Tensor) -> T.Tensor:
    s_out = F.normalize(s_out)
    t_out = repulse(t_out)
    return F.smooth_l1_loss(s_out, t_out)
