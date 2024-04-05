import torch as T
from torch import nn

from mltools.mltools.flows import rqs_flow
from src.models.mpm_base import MPMBase


class MPMFlow(MPMBase):
    """Using a flow to calculate the recover the masked particles."""

    def __init__(self, *, embed_dim: int, flow_config: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.csts_head = nn.Linear(self.decoder.dim, embed_dim)
        self.flow = rqs_flow(xz_dim=self.csts_dim, ctxt_dim=embed_dim, **flow_config)

    @T.autocast("cuda", enabled=False)  # Autocasting is bad for flows
    @T.autocast("cpu", enabled=False)
    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calulate the loss under the flow."""
        # The flow can't handle discrete targets
        # So we need to add noise to the neutral impact parameters (last 4)
        # This is a hack because it is hardcoded but works for now
        # This crap below is all because using multiple slices no longer allows inplace
        neut = null_mask & ((csts_id == 0) | (csts_id == 2))
        neut = neut.unsqueeze(-1).expand_as(normed_csts).clone()
        neut[..., :-4] = False  # Which variables we want to change
        normed_csts[neut] = T.randn_like(normed_csts[neut])

        return (
            self.flow.forward_kld(
                normed_csts[null_mask].float(),
                context=self.csts_head(decoder_outs[null_mask]).float(),
            )
            / 3
        )  # Arbitrary scaling factor

    def sample_csts(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Get an estimate of the dropped csts using the outputs."""
        samples = self.flow.sample(
            decoder_outs.shape[0], context=self.csts_head(decoder_outs)
        )[0]
        return self.normaliser.reverse(samples)
