import torch as T
from torch import nn

from src.models.mpm_base import MPMBase


class MPMReg(MPMBase):
    """Direct regression for the constituents."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csts_head = nn.Linear(self.decoder.dim, self.csts_dim)

    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calculate the loss using direct regression."""
        csts_out = self.csts_head(decoder_outs[null_mask])
        return (normed_csts[null_mask] - csts_out).abs().mean()

    def sample_csts(self, decoder_outs: T.Tensor) -> list:
        """Get an estimate of the dropped csts using the outputs."""
        csts_out = self.csts_head(decoder_outs)
        return self.normaliser.reverse(csts_out)
