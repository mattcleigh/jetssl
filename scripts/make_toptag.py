import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mltools.mltools.utils import signed_angle_diff
from src.setup.root_utils import csts_to_jet, pxpypz_to_ptetaphi


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/datasets/TopTagging/",
        help="The path to the TopTag files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/srv/fast/share/rodem/TopTagging/",
        help="The path to save the converted files",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the JetClass root files to a more usable HDF format.

    The output features are the following:
    Independant continuous (7 dimensional vector)
    - 0: pt
    - 1: deta
    - 2: dphi
    - 3: d0val
    - 4: d0err
    - 5: dzval
    - 6: dzerr
    Independant categorical (single int representing following classes)
    - 0: isPhoton
    - 1: isHadron_Neg
    - 2: isHadron_Neutral
    - 3: isHadron_Pos
    - 4: isElectron_Neg
    - 5: isElectron_Pos
    - 6: isMuon_Neg
    - 7: isMuon_Pos
    """
    # Get the arguments
    args = get_args()

    # Make the destination folder
    Path(args.dest_path).mkdir(parents=True, exist_ok=True)

    # Loop over the files
    for file in ["val.h5", "test.h5", "train.h5"]:
        print(f"Processing {file}")

        # Load the data from the file
        df = pd.read_hdf(Path(args.source_path) / file, key="table")

        # Pull out the class labels
        labels = df.is_signal_new.to_numpy()

        # Select the constituent columns columns
        col_names = ["PX", "PY", "PZ"]
        cst_cols = [f"{var}_{i}" for i in range(200) for var in col_names]
        csts = np.reshape(df[cst_cols].to_numpy().astype("f"), (-1, 200, 3))
        mask = np.any(csts != 0, axis=-1)

        # Calculate the overall jet kinematics from the constituents
        jets = csts_to_jet(csts, mask)

        # Convert both sets of values to spherical
        csts[..., :3] = pxpypz_to_ptetaphi(csts)
        jets[..., :3] = pxpypz_to_ptetaphi(jets)

        # Calculate the relative eta and phi
        csts[..., 1] -= jets[..., 1:2]
        csts[..., 2] = signed_angle_diff(csts[..., 2], jets[..., 2:3])

        # Pad the missing impact parameter information with zeros
        csts = np.concatenate([csts, np.zeros((*csts.shape[:-1], 4))], axis=-1)

        # Create an empty array for the missing ids
        csts_id = np.zeros(csts.shape[:-1], int)

        # The jet features need the number of constituents
        num_csts = np.sum(mask, axis=-1, keepdims=True)
        jets = np.concatenate([jets, num_csts], axis=-1)
        jets = np.nan_to_num(jets, nan=0)

        # Save the data to an HDF file
        dest_file = Path(args.dest_path) / file.replace(".h5", "_set.h5")
        with h5py.File(dest_file, "w") as f:
            f.create_dataset("csts", data=csts)
            f.create_dataset("csts_id", data=csts_id)
            f.create_dataset("jets", data=jets)
            f.create_dataset("labels", data=labels)
            f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    main()
