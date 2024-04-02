from functools import partial

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data.dataloader import default_collate

from src.models.mpm_base import MPMBase


class MPMToken(MPMBase):
    """Clustering and CE for the constituents."""

    def __init__(
        self,
        *args,
        labeller: partial,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.labeller = labeller(inpt_dim=self.csts_dim)
        self.csts_head = nn.Linear(self.decoder.dim, self.labeller.num_labels)

    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calulate the loss using the labeller."""
        labels = self.labeller(normed_csts[null_mask]).long()
        preds = self.csts_head(decoder_outs[null_mask])
        return cross_entropy(preds, labels, label_smoothing=0.1)

    def sample_csts(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Sample using the labeller."""
        preds = self.csts_head(decoder_outs)
        probs = T.softmax(preds, dim=-1)
        idxes = T.multinomial(probs, 1).squeeze(1)
        samples = self.labeller.idx_to_code(idxes)
        return self.normaliser.reverse(samples)

    def on_fit_start(self) -> None:
        """At the start of the fit fill in the normaliser and labeller.

        Only use 30 batches but this will be sufficient to get a good fit.
        """
        n_store = []
        m_store = []
        loader = self.trainer.datamodule.train_dataloader()
        loader.num_workers = 0  # No multithreading
        loader.collate_fn = default_collate  # Turn off the batch compression
        for i, sample in enumerate(loader):
            n_store.append(sample["csts"])
            m_store.append(sample["mask"])
            if i > 30:
                break
        csts = T.vstack(n_store).to(self.device)
        mask = T.vstack(m_store).to(self.device)

        # Fit the normaliser such that the labeller uses scaled inputs
        self.normaliser.fit(csts, mask, freeze=False)  # Dont freeze yet!
        csts = self.normaliser(csts, mask)  # Pass through the normaliser
        self.labeller.fit(csts, mask)  # Fit the labeller to the normalised inputs
