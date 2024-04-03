import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np

from mltools.mltools.utils import signed_angle_diff
from src.setup.root_utils import (
    common_particle_class,
    lifetime_signing,
    read_shlomi_file,
)

# The branches needed from the shlomi root files


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Shlomi root files to a more usable format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/datasets/shlomi/",
        help="The path to the Shlomi files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/srv/fast/share/rodem/shlomi/",
        help="The path to save the converted files",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the Shlomi root files to a more usable HDF format.

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

    # Make sure the destination path exists
    Path(args.dest_path).mkdir(parents=True, exist_ok=True)

    # Get all of the root files in the source path
    root_files = list(Path(args.source_path).iterdir())

    # Loop over the root files
    for file in root_files:
        # Read the root file
        jets, tracks, labels, vertices = read_shlomi_file(file)

        # The labels are 0, 4, 5, change them to 0, 1, 2
        labels = np.where(labels == 4, 1, labels)
        labels = np.where(labels == 5, 2, labels)
        labels = np.squeeze(labels)

        # We need to split the tracks into the different groups of information
        csts = tracks[..., :7]
        charge = tracks[..., 7]
        pdgid = tracks[..., 8]
        vtx_id = tracks[..., 9].astype("l")

        # Switch to lifetime signing convention for the impact parameters
        d0, z0 = lifetime_signing(
            d0=csts[..., 3],
            z0=csts[..., 5],
            tracks=csts[..., :3],
            jets=jets[..., :3],
        )
        csts[..., 3] = d0
        csts[..., 5] = z0

        # The csts eta and phi must be centered on the jet
        csts[..., 1] -= jets[:, 1:2]
        csts[..., 2] = signed_angle_diff(csts[..., 2], jets[:, 2:3])

        # Convert the particle class information to the common format
        csts_id = common_particle_class(charge, pdgid)

        # Get a mask based on track pt
        mask = csts[..., 0] > 0

        # Save the data to an HDF file
        dest_file = (
            str(file).replace(args.source_path, args.dest_path).replace(".root", ".h5")
        )
        with h5py.File(dest_file, "w") as f:
            f.create_dataset("csts", data=csts)
            f.create_dataset("csts_id", data=csts_id)
            f.create_dataset("jets", data=jets)
            f.create_dataset("labels", data=labels)
            f.create_dataset("vtx_id", data=vtx_id)
            f.create_dataset("vtx_pos", data=vertices)
            f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    main()
