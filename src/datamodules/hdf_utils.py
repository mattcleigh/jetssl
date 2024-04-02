import logging
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class HDFRead:
    key: str
    dtype: type
    s_: slice = slice(None)  # noqa: RUF009


def make_slice(*args) -> slice:
    """Create slices from arguments (easier for hydra)."""
    if isinstance(args[0], Iterable):
        return tuple(starmap(slice, args))
    return slice(*args)


def combine_slices(a, b) -> slice:
    """Combine a list of slices into a single multidim slice."""
    if b == slice(None) or b is None:
        return a
    if not isinstance(b, slice):
        b = make_slice(*b)
    c = tuple(a) if isinstance(a, Iterable) else (a,)
    if isinstance(b, Iterable):
        c += tuple(b)
    else:
        c += (b,)
    return c


def load_h5_into_dict(
    file_list: list,
    data_types: list[HDFRead],
    n_samples: int | list | None = None,
    disable: bool = True,
    concatenate: bool = True,
) -> dict:
    """Load data from an h5 file into a dictionary of numpy arrays."""
    # Make sure the n_samples is a list
    if isinstance(n_samples, int) or n_samples is None:
        n_samples = [n_samples] * file_list

    # Create the dict from the requested data types
    data = {d.key: [] for d in data_types}

    # Cycle through all the files
    for file, nj in tqdm(
        zip(file_list, n_samples, strict=False),
        total=len(file_list),
        desc=f"loading {len(file_list)} files into memory",
        disable=disable,
    ):
        with h5py.File(file, mode="r") as ifile:
            for d in data_types:
                # Create a combined slice that works for lower dimensions
                sl = combine_slices(np.s_[:nj], d.s_)

                # Index the data
                sample = ifile[d.key][sl].astype(d.dtype)

                # If we are concatenating, keep as array for final stack
                if concatenate:
                    data[d.key].append(sample)

                # Otherwise convert to a list of arrays
                else:
                    data[d.key] += list(sample)

    # Stack everything together
    if concatenate:
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0)
    return data


def get_file_list(
    processes: list, path: Path, n_files: int | list | None = None
) -> list:
    """Load the list of files for each process. If n_files is not specified, will always
    makes sure that the number of files for each process is balanced.

    Parameters
    ----------
    processes : list
        List of processes to load files for.
    path : Path
        Path to the directory containing the files.
    n_files : int | list | None, optional
        Number of files to load for each process.
        If a single integer is provided, it will be used for all processes.
        If a list is provided, it must have the same length as `processes`.
        If `None`, the number of files will be automatically balanced to limit.
        Default is `None`.

    Returns
    -------
    files_per_proc
        List of lists, where each sublist contains the file paths for one process.
    """
    n_proc = len(processes)

    # Make a list of all possible files
    files_per_proc = [list(path.glob(p + "_*.h5")) for p in processes]
    for x in files_per_proc:
        x.sort(key=lambda f: int("".join(filter(str.isdigit, str(f)))))

    # Autobalance the dataset
    if n_files is None:
        log.info("Checking file list to ensure balanced dataset")
        n_files = 999999999
        for i in range(n_proc):
            proc_files = len(files_per_proc[i])
            n_files = min(n_files, proc_files)

    # Turn into a list for generality
    if isinstance(n_files, int):
        n_files = n_proc * [n_files]

    # Crop the list to length and couple with the number of jets to load
    for i in range(n_proc):
        files_per_proc[i] = files_per_proc[i][: n_files[i]]
    return files_per_proc
