import torch as T
from torch import nn

from src.models.mpm_base import MPMBase
from src.models.utils import VectorDiffuser


class MPMDiff(MPMBase):
    """Using a diffusion model to recover the masked particles."""

    def __init__(self, *, embed_dim: int, diff_config: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.csts_head = nn.Linear(self.decoder.dim, embed_dim)
        self.diff = VectorDiffuser(
            inpt_dim=self.csts_dim, ctxt_dim=embed_dim, **diff_config
        )

    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calulate the diffusion loss."""
        return self.diff.get_loss(
            normed_csts[null_mask], self.csts_head(decoder_outs[null_mask])
        )

    @T.no_grad()
    def sample_csts(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Get an estimate of the dropped csts using the outputs."""
        x1 = T.randn((decoder_outs.shape[0], self.csts_dim), device=self.device)
        ctxt = self.csts_head(decoder_outs)
        times = T.linspace(1, 0, 100, device=decoder_outs.device)
        return self.diff.generate(x1, ctxt, times)
