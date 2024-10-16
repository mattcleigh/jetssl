import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/",
        help="The path to the JetClass files",
    )
    return parser.parse_args()


def main() -> None:
    """Combine all jetclass files into a single HDF5 file."""
    # Get the arguments
    args = get_args()

    # Get the top level folders (train, val, test)
    subsets = [x for x in Path(args.data_path).iterdir() if x.is_dir()]
    # Cycle through each subset
    for subset in subsets:
        # Skip the train set
        if "train" in subset.name:
            continue

        print(f"Processing {subset.name}")

        # Create the target file
        target_file = Path(args.data_path) / f"{subset.name}_combined.h5"
        h5fw = h5py.File(target_file, mode="w")
        row = 0  # Counter for current location

        # Get a list of all files in the subset and sort
        files = list(subset.glob("*.h5"))

        # Get the name of the keys from the first file
        with h5py.File(files[0], "r") as h5fr:
            buffer = {k: [] for k in h5fr}

        # Get a list of common numbers in the file names
        # This way we ensure each buffer has one file of each type
        common_nums = np.unique([int(x.stem.split("_")[-1]) for x in files])
        for num in tqdm(common_nums):
            # Reset the buffer
            for k in buffer:
                buffer[k] = []

            # Cycle through each file
            sublist = [x for x in files if int(x.stem.split("_")[-1]) == num]
            for h5name in tqdm(sublist, leave=False):
                with h5py.File(h5name, "r") as h5fr:
                    for k in buffer:  # noqa: PLC0206
                        buffer[k].append(h5fr[k][:])

            # Shuffle each list in the buffer
            len_buff = sum(len(v) for v in buffer[k])
            order = np.random.default_rng().permutation(len_buff)
            for k in buffer:
                buffer[k] = np.concatenate(buffer[k], axis=0)[order]

            # Write the buffer to the target file
            for k, v in tqdm(buffer.items(), leave=False):
                # Create the dataset if it doesn't exist
                if row == 0:
                    h5fw.create_dataset(
                        k,
                        dtype=v.dtype,
                        shape=v.shape,
                        chunks=(1000, *v.shape[1:]),
                        maxshape=(None, *v.shape[1:]),
                    )

                # Resize the target table if it is too small
                if row + len_buff > len(h5fw[k]):
                    h5fw[k].resize((row + len_buff, *v.shape[1:]))

                # Save the data
                h5fw[k][row : row + len_buff] = v
            row += len_buff

        # Close the file
        h5fw.close()


if __name__ == "__main__":
    main()
