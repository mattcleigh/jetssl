import torch as T
from torch import nn
from torchdiffeq import odeint

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import CosineEncodingLayer


class JetBackbone(nn.Module):
    """Basic backbone for the jet models."""

    def __init__(
        self,
        normaliser: nn.Module,
        ctst_embedder: nn.Module,
        id_embedder: nn.Module,
        encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.normaliser = normaliser
        self.ctst_embedder = ctst_embedder
        self.id_embedder = id_embedder
        self.encoder = encoder

    def forward(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
        do_norm: bool = True,
    ) -> T.Tensor:
        """Pass through the complete backbone."""
        if do_norm:
            csts = self.normaliser(csts, mask)
        x = self.ctst_embedder(csts) + self.id_embedder(csts_id)
        x = self.encoder(x, kv_mask=mask)
        new_mask = self.encoder.get_combined_mask(mask)
        return x, new_mask


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

    def forward(self, xt: T.Tensor, t: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        """Get the fully denoised estimate."""
        c = T.cat([self.time_encoder(t), ctxt], dim=-1)
        return self.mlp(xt, c)

    def get_loss(self, x0: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        t = T.sigmoid(T.randn(x0.shape[0], 1, device=x0.device))
        x1 = T.randn_like(x0)
        xt = (1 - t) * x0 + t * x1
        v = self.forward(xt, t, ctxt)
        return (v - (x1 - x0)).square().mean()

    def generate(self, x1: T.Tensor, ctxt: T.Tensor, times: T.Tensor) -> T.Tensor:
        """Generate a sample."""

        def ode_fn(t, xt):
            t = t * xt.new_ones([xt.shape[0], 1])
            return self(xt, t, ctxt)

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
