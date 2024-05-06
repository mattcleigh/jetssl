"""Pytorch Dataset definitions of various collections training samples."""

import logging
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from mltools.mltools.utils import intersperse
from src.datamodules.hdf_utils import HDFRead, get_file_list, load_h5_into_dict

# This is to prevent the loaders from being killed when loading new files
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 999

log = logging.getLogger(__name__)


def identity(x: Any) -> Any:
    """Placeholder function for no transformation."""
    return x


# Recently update to match the defaults jet_labels
# in src/jets/datamodules/root_utls/read_jetclass_file()
JC_CLASS_TO_LABEL = {
    "ZJetsToNuNu": 0,
    "TTBarLep": 1,
    "TTBar": 2,
    "WToQQ": 3,
    "ZToQQ": 4,
    "HToBB": 5,
    "HToCC": 6,
    "HToGG": 7,
    "HToWW4Q": 8,
    "HToWW2Q1L": 9,
}


class JetMappable(Dataset):
    """The base class for loading jets stored as HDF datasets."""

    def __init__(
        self,
        *,
        path: str,
        features: list[list],
        n_classes: int,
        csts_dim: int | None = None,
        processes: list | str = "all",
        n_files: int | list | None = None,
        n_jets: int | list | None = None,
        transforms: Callable = identity,
        n_jets_total: int | None = None,
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
        processes : list or str
            The processes to be used.
            If a string is provided, it is converted into a list.
        n_files : int, list or None, optional
            The number of files per process. If not provided, all files are used.
        n_jets : int or None, optional
            The number of jets to load per file per process.
            If not provided, all jets are used from each file.
        transforms : partial
            A callable function to apply during the getitem method
        n_jets_total : int or None, optional
            The total number of jets in the dataset.
            If not provided, it is calculated from the files and the n_jets.
        """
        super().__init__()

        # Processes and the number of jets must be a list for generality
        if isinstance(processes, str):
            if processes == "all":
                processes = list(JC_CLASS_TO_LABEL.keys())
            else:
                processes = [processes]
        if n_jets_total is not None:
            n_jets = n_jets_total // len(processes)
        if isinstance(n_jets, int) or n_jets is None:
            n_jets = [n_jets] * len(processes)

        # Insert the csts dim into the features
        # This is a hack but we need the csts to change with hydra for now
        if csts_dim is not None:
            print("Warning! Explicitly setting the csts dimension")
            print("This is a hack and should be removed in the future!")
            c_idx = [i for i, f in enumerate(features) if f[0] == "csts"][0]
            curr = features[c_idx][-1]
            features[c_idx][-1] = [curr, [csts_dim]]
            print("New feature slice for csts:")
            print(features[c_idx])

        # Class attributes
        self.path = Path(path)
        self.processes = processes
        self.n_classes = n_classes
        self.n_jets = n_jets
        self.features = list(starmap(HDFRead, features))
        self.transforms = transforms

        # Get the full paths of every file that makes up this dataset
        self.files_per_proc = get_file_list(processes, self.path, n_files)
        self.n_files = [len(f) for f in self.files_per_proc]

        # Create the file list by evenly distributing the processes
        self.file_list = list(intersperse(*self.files_per_proc))

        # Create a matching list of how many samples to load from each file
        nj_per_proc = [
            [nj] * nf for nj, nf in zip(self.n_jets, self.n_files, strict=False)
        ]
        self.njet_list = list(intersperse(*nj_per_proc))

        # Load the data from the root filess
        self.data_dict = load_h5_into_dict(
            file_list=self.file_list,
            data_types=self.features,
            n_samples=self.njet_list,
            disable=False,
        )

        log.info(f"Loaded {len(self)} jets from {len(self.file_list)} files")

    def __len__(self) -> int:
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        sample_dict = {k: v[idx] for k, v in self.data_dict.items()}
        return self.transforms(sample_dict)


class JetCWola(Dataset):
    """A mappable dataset that loads signal and background jets and mixes the labels."""

    def __init__(
        self,
        num_signal: int = 100_000,
        num_background: int = 1000_000,
        signal_process: str = "TTBar",
        background_process: str = "ZJetsToNuNu",
        **kwargs,
    ) -> None:
        # Needed for the model init
        self.n_classes = 2

        # Load the signal and background datasets
        self.signal = JetMappable(
            n_classes=1,
            processes=signal_process,
            n_files=10,  # Distribute among 10 files
            n_jets=num_signal // 10,
            **kwargs,
        )
        self.n_signal = len(self.signal)
        self.background = JetMappable(
            n_classes=1,
            processes=background_process,
            n_files=10,  # Distribute among 10 files
            n_jets=num_background // 10,
            **kwargs,
        )
        self.n_background = len(self.background)

    def __len__(self) -> int:
        return self.n_background + self.n_signal

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        # Take from signal first
        if idx < len(self.signal):
            sample = self.signal[idx]
            sample["cwola_labels"] = 1
            sample["labels"] = 1
        # Otherwise take from background which label is split in two
        else:
            sample = self.background[idx - self.n_signal]
            sample["cwola_labels"] = idx % 2  # Odd becomes signal
            sample["labels"] = 0
        return sample


class JetDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_set: partial,
        val_set: partial,
        test_set: partial,
        loader_config: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.valid_set = val_set()  # initialise now to calculate data shape
        self.loader_config = loader_config
        self.n_classes = self.valid_set.n_classes

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = self.hparams.train_set()
        if stage in {"predict", "test"}:
            self.test_set = self.hparams.test_set()

    def get_dataloader(self, dataset: Dataset, flag: str) -> DataLoader:
        kwargs = deepcopy(self.loader_config)
        if flag != "train":
            kwargs["drop_last"] = False
            kwargs["shuffle"] = False
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set, "train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.valid_set, "valid")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set, "test")

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> tuple:
        """Get a data sample to help initialise the network."""
        return next(iter(self.valid_set))
