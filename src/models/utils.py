from copy import deepcopy

import torch as T
from torch import nn
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
        jets_emb: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.csts_emb = csts_emb
        self.csts_id_emb = csts_id_emb
        self.encoder = encoder
        self.jet_emb = jets_emb

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
            self.jet_emb(jets)
            if hasattr(self, "jet_emb") and self.jet_emb is not None
            else None
        )
        x = self.encoder(x, mask=mask, ctxt=ctxt)
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
        use_hlv: bool = False,
    ) -> None:
        cemb_dim = 64 if use_hlv else 0
        encoder = Transformer(**encoder_config, ctxt=cemb_dim)
        csts_emb = nn.Linear(csts_dim, encoder.dim)
        csts_id_emb = nn.Embedding(CSTS_ID, encoder.dim) if use_csts_id else None
        jets_emb = nn.Linear(self.ctxt_dim, cemb_dim) if use_hlv else None
        super().__init__(csts_emb, csts_id_emb, encoder, jets_emb)


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
