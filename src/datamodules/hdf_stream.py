import logging
from collections.abc import Callable, Generator
from functools import partial
from itertools import starmap

import h5py
import numpy as np
import torch as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from src.datamodules.hdf import identity
from src.datamodules.hdf_utils import HDFRead, combine_slices

log = logging.getLogger(__name__)


class BatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Batch sampler for an h5 dataset.

        Parameters
        ----------
        dataset : torch.data.Dataset
            Input dataset, used only to determine the length
        batch_size : int
            Number of objects to batch
        shuffle : bool
            Shuffle the batches
        drop_last : bool
            Drop the last incomplete batch (if present)
        """
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length // self.batch_size  # full batches
        self.incl_last = not drop_last and (self.dataset_length % self.batch_size != 0)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.n_batches + self.incl_last

    def __iter__(self) -> Generator:
        # Create the batch ids
        if self.shuffle:
            self.batch_ids = T.randperm(self.n_batches)
        else:
            self.batch_ids = T.arange(self.n_batches)

        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start = batch_id * self.batch_size
            stop = (batch_id + 1) * self.batch_size
            yield np.s_[int(start) : int(stop)]

        # yield the partial batch at the end
        if self.incl_last:
            start = self.n_batches
            stop = self.dataset_length
            yield np.s_[int(start) : int(stop)]


class JetHDFStream(Dataset):
    """A class for streaming in jets from HDF without loading buffers into memory.

    Should be combined with the RandomBatchSampler for training.
    """

    def __init__(
        self,
        *,
        path: str,
        n_classes: int,
        features: list[list] | None = None,
        csts_dim: int | None = None,
        n_jets: int | list = 0,
        transforms: Callable | list = identity,
    ) -> None:
        """Parameters
        ----------
        path : str
            The path containing all the HDF files.
        features : list of tuples
            The features to be loaded from the dataset.
            Should have three elements: the (key, dtype, slice).
        n_classes : int
            The number of classes in the dataset. Purely for convenience.
            Is not actually used in the class.
        n_jets: int or None, optional
            The total number of jets in the dataset.
        transforms : partial
            A callable function to apply during the getitem method
        """
        # Default features
        if features is None:
            features = [
                ["csts", "f", [128]],
                ["mask", "bool", [128]],
                ["csts_id", "l", [128]],
                ["labels", "l", None],
                ["jets", "f", None],
            ]

        # Insert the csts dim into the features
        # This is a hack but we need the csts to change with hydra for now
        if csts_dim is not None:
            log.info("Warning! Explicitly setting the csts dimension")
            log.info("This is a hack and should be removed in the future!")
            c_idx = [i for i, f in enumerate(features) if f[0] == "csts"][0]
            curr = features[c_idx][-1]
            features[c_idx][-1] = [curr, [csts_dim]]
            log.info("New feature slice for csts:")
            log.info(features[c_idx])

        # Class attributes
        self.n_classes = n_classes
        self.file = h5py.File(path, mode="r")
        self.features = list(starmap(HDFRead, features))

        # Set the number of jets (if not set, use all jets in the file)
        n_jets_in_file = len(next(iter(self.file.values())))
        self.n_jets = min(n_jets, n_jets_in_file) or n_jets_in_file

        # Save the preprocessing as a list
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

        log.info(f"Streaming from {path}")
        log.info(f"- selected {self.n_jets} jets")

    def __len__(self) -> int:
        return self.n_jets

    def __getitem__(self, idx: int | slice) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        data = {
            d.key: self.file[d.key][combine_slices(idx, d.s_)].astype(d.dtype)
            for d in self.features
        }
        for fn in self.transforms:
            data = fn(data)
        return data


class StreamModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_set: partial,
        val_set: partial,
        test_set: partial,
        num_workers: int = 6,
        batch_size: int = 1000,
        pin_memory: bool = True,
        transforms: list | Callable = identity,
    ) -> None:
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set()  # initialise now to calculate data shape
        self.test_set = test_set
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.transforms = transforms
        self.n_classes = self.val_set.n_classes
        self.val_set.transforms = self.transforms

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = self.train_set()
            self.train_set.transforms = self.transforms
        if stage in {"predict", "test"}:
            self.test_set = self.test_set()
            self.test_set.transforms = self.transforms

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            sampler=BatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,  # flag == "train", Honestly its so big...
                drop_last=flag == "train",
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=None,  # batch size is handled by the sampler
            shuffle=False,  # shuffle is handled by the sampler
            collate_fn=None,  # collations should be handled by the dataset
        )

    def get_data_sample(self) -> tuple:
        """Get a data sample to help initialise the network."""
        return next(iter(self.val_set))

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_set, "val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, "test")

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
