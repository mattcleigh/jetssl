import torch as T
from torch import nn
from torch.nn.functional import cross_entropy
from torchpq.clustering import KMeans

from src.models.mpm_base import MPMBase


class MPMKmeans(MPMBase):
    """Perform kmeans clustering the use crossent for the constituents."""

    def __init__(
        self,
        *args,
        kmeans_config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kmeans = KMeans(**kmeans_config)
        self.csts_head = nn.Linear(self.decoder.dim, self.kmeans.n_clusters)

    def masked_cst_loss(
        self,
        normed_csts: T.Tensor,
        csts_id: T.Tensor,
        null_mask: T.BoolTensor,
        decoder_outs: T.Tensor,
    ) -> T.Tensor:
        """Calculate the loss using the labeller."""
        labels = self.kmeans.predict(normed_csts[null_mask].T.contiguous()).long()
        preds = self.csts_head(decoder_outs[null_mask])
        return cross_entropy(preds, labels, label_smoothing=0.1)

    def sample_csts(self, decoder_outs: T.Tensor) -> T.Tensor:
        """Sample using the labeller."""
        preds = self.csts_head(decoder_outs)
        probs = T.softmax(preds, dim=-1)
        idxes = T.multinomial(probs, 1).squeeze(1)
        samples = self.kmeans.centroids.index_select(1, idxes).T
        return self.normaliser.reverse(samples)

    def on_fit_start(self) -> None:
        """At the start of the fit, fit the kmeans."""
        # Skip kmeans has already been initialised (e.g. from a checkpoint)
        if self.kmeans.centroids is not None:
            return

        # Load the first 50 batches of training data
        csts = []
        mask = []
        loader = self.trainer.datamodule.train_dataloader()
        for i, sample in enumerate(loader):
            csts.append(sample["csts"])
            mask.append(sample["mask"])
            if i > 50:
                break
        csts = T.vstack(csts).to(self.device)
        mask = T.vstack(mask).to(self.device)

        # Fit the normaliser such that the labeller uses scaled inputs
        self.normaliser.fit(csts, mask, freeze=True)  # 50 batches should be enough
        csts = self.normaliser(csts, mask)  # Pass through the normaliser
        inputs = csts[mask].T.contiguous()
        self.kmeans.fit(inputs)
