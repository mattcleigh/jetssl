"""Pytorch Dataset definitions of various collections training samples."""

import gc
import logging
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from pyparsing import Generator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from mltools.mltools.utils import batched, intersperse
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


class JetHDFBase:
    """The base class for loading jets stored as HDF datasets."""

    def __init__(
        self,
        *,
        path: str,
        features: list[tuple],
        n_classes: int,
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


class JetMappable(Dataset, JetHDFBase):
    """A pytorch mappable dataset for jets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        num_signal: int = 1000_000,
        num_background: int = 10_000,
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
            n_files=10,
            n_jets=num_signal // 10,
            **kwargs,
        )
        self.n_signal = len(self.signal)
        self.background = JetMappable(
            n_classes=0,
            processes=background_process,
            n_files=10,
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
            sample["cwola_labels"] = idx % 2
            sample["labels"] = 0
        return sample


class JetMappablePartial(Dataset, JetHDFBase):
    """A pytorch mappable dataset that each epoch will load a portion of the jets.

    Alternative to Iterable which keeps failing for some reason :(
    Requires that the trainer is reinitialising the dataloaders each epoch
    """

    def __init__(self, files_per_epoch: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert files_per_epoch <= len(self.file_list)
        self.max_idx = len(self.file_list) // files_per_epoch
        self.files_per_epoch = files_per_epoch
        self.data_dict = {}  # Initalised here for the delete in load_epoch
        self.idx = -1
        self.load_epoch(0)

    def load_epoch(self, epoch: int) -> None:
        """Load a set of files for the relevant epoch."""
        # Check if the files need to be reloaded
        if self.idx == (new_idx := epoch % self.max_idx):
            return

        # Clear the data_dict to prevent memory spike between each epoch
        self.data_dict.clear()
        del self.data_dict
        gc.collect()  # Call the garbage collector

        # Work out the slice of files to load from the list
        self.idx = new_idx
        start = self.files_per_epoch * self.idx
        end = self.files_per_epoch * (self.idx + 1)
        log.info(f"Current epoch count {epoch}, reading files {start} to {end}")

        # Load the data from the hdf filess
        self.data_dict = load_h5_into_dict(
            file_list=self.file_list[slice(start, end)],
            data_types=self.features,
            n_samples=self.njet_list[slice(start, end)],
            disable=False,
            concatenate=False,
        )
        log.info(f"Loaded {len(self)} jets from {self.files_per_epoch} files")

    def __len__(self) -> int:
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves an item and applies the pre-processing function."""
        sample_dict = {k: v[idx] for k, v in self.data_dict.items()}
        return self.transforms(sample_dict)


class JetIterable(IterableDataset, JetHDFBase):
    """A pytorch iterable dataset for jets."""

    def __init__(
        self,
        *args,
        files_per_buffer: int = 0,
        shuffle: bool = False,
        n_steps_per_epoch: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.files_per_buffer = files_per_buffer or len(self.processes)
        self.shuffle = shuffle
        self.n_total = self._count_samples()
        self.n_steps_per_epoch = n_steps_per_epoch
        log.info(f"Streaming {len(self)} jets from {len(self.file_list)} files")

    def _count_samples(self) -> int:
        """Counts the total number of samples in the entire dataset.

        Relatively quick as there is no big I/O
        """
        total = 0
        log.info("Counting the total number of samples in the dataset")
        for file, njets in tqdm(
            zip(self.file_list, self.njet_list, strict=False), total=sum(self.n_files)
        ):
            with h5py.File(file, mode="r") as f:
                key = list(f.keys())[0]
                n = len(f[key])
            if njets is not None:
                n = min(n, njets)
            total += n
        return total

    def __len__(self) -> int:
        return self.n_steps_per_epoch or self.n_total

    def __iter__(self) -> Generator:
        """Called seperately for each worker (thread).

        - Divides up the file for each worker
        - Loads 1 file per process at a time into a buffer
        - Divides up the buffers into batches
        - Returns each batch
        """
        # Check if we are using single-process vs multi process data loading
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Calculate which files this worker is responsible for
        worker_files = np.array_split(self.file_list, num_workers)[worker_id]
        worker_njets = np.array_split(self.njet_list, num_workers)[worker_id]

        # Check that we have enough files for the shufflling
        if self.shuffle and len(worker_files) < self.files_per_buffer:
            log.warning(
                "You have too many workers resulting in incomplete buffers. "
                "If you are trying to shuffle your data this wont work!"
            )

        # Break them up into batches, each batch will be a buffer
        batched_files = batched(worker_files, self.files_per_buffer)
        batched_njets = batched(worker_njets, self.files_per_buffer)

        # Cycle through the files grouped by buffer
        for b_files, b_njets in zip(batched_files, batched_njets, strict=False):
            # Load the data into a buffer
            data_dict = load_h5_into_dict(
                file_list=b_files,
                data_types=self.features,
                n_samples=b_njets,
                disable=True,
                concatenate=False,
            )

            # Generate an order from which to iterate through the buffer
            len_buffer = len(next(iter(data_dict.values())))
            if self.shuffle:
                order = np.random.default_rng().permutation(len_buffer)
            else:
                order = np.arange(len_buffer)

            # Yeild from the buffer one sampe at a time
            for i in order:
                sample_dict = {k: v[i] for k, v in data_dict.items()}
                yield self.transforms(sample_dict)

            # Delete the data_dict to prevent memory spike between each buffer
            # Probably overkill with both clear and del, but both seem to be needed?
            sample_dict.clear()
            data_dict.clear()
            del data_dict
            del sample_dict
            gc.collect()

        # Final task of 1st worker is to shuffle the file order for the next epoch
        # if self.shuffle and worker_id == 0:
        #     for fl in self.files_per_proc:
        #         random.shuffle(fl)
        #     self.file_list = intersperse(*self.files_per_proc)


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

        # Make the datamodule stateful as it helps with checkpointing
        self.state = {"epoch": -1}

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = self.hparams.train_set()
        if stage in {"predict", "test"}:
            self.test_set = self.hparams.test_set()

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state_dict: dict) -> None:
        return self.state.update(state_dict)

    def train_dataloader(self) -> DataLoader:
        self.state["epoch"] += 1
        if hasattr(self.train_set, "load_epoch"):
            self.train_set.load_epoch(self.state["epoch"])
        return DataLoader(self.train_set, **self.loader_config)

    def val_dataloader(self) -> DataLoader:
        val_config = deepcopy(self.loader_config)
        val_config["drop_last"] = False
        val_config["shuffle"] = False
        return DataLoader(self.valid_set, **val_config)

    def test_dataloader(self) -> DataLoader:
        test_config = deepcopy(self.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        return DataLoader(self.test_set, **test_config)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> tuple:
        """Get a data sample to help initialise the network."""
        return next(iter(self.valid_set))
