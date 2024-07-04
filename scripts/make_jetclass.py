import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.setup.root_utils import (
    common_particle_class,
    lifetime_signing,
    read_jetclass_file,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/JetClass/Pythia/",
        help="The path to the JetClass files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/",
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

    # Make sure the destination path exists
    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get all of the root files in the source path
    source_path = Path(args.source_path)
    subfolders = [x for x in source_path.iterdir() if x.is_dir()]

    # Loop over the subfolders
    for subfolder in subfolders:
        print(f"Processing {subfolder.name}")

        # Copy the subfolder to the destination path
        dest_folder = dest_path / subfolder.name

        # Make the folder
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        files = list(subfolder.glob("*.root"))

        # Sort the files based the number in the name
        files = sorted(files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        # Loop through the files in the subfolder and load the information
        for file in tqdm(files):
            # Define the destination file
            dest_file = dest_path / file.name.replace(".root", ".h5")

            # Skip if the file already exists
            # if Path(dest_file).exists():
            # continue

            # Load the data from the file
            jets, csts, labels = read_jetclass_file(file)

            # Get the pt from the px and py
            pt = np.linalg.norm(csts[..., :2], axis=-1, keepdims=True)

            # Split the csts into the different groups of information
            sel_csts = np.concatenate([pt, csts[..., 2:8]], axis=-1)

            # Switch to lifetime signing convention for the impact parameters
            d0, z0 = lifetime_signing(
                d0=sel_csts[..., 3],
                z0=sel_csts[..., 5],
                tracks=sel_csts[..., :3],
                jets=jets[..., :3],
                is_centered=True,
            )
            sel_csts[..., 3] = d0
            sel_csts[..., 5] = z0

            # Clip eta and phi to the actual jet radius
            sel_csts[..., 1:3] = np.clip(sel_csts[..., 1:3], -0.8, 0.8)

            # Convert the particle class information to the common format
            csts_id = common_particle_class(
                charge=csts[..., -6],
                isPhoton=csts[..., -5].astype(bool),
                isHadron=csts[..., -4].astype(bool) | csts[..., -3].astype(bool),
                isElectron=csts[..., -2].astype(bool),
                isMuon=csts[..., -1].astype(bool),
            )

            # The jet features need the number of constituents
            mask = sel_csts[..., 0] > 0
            num_csts = np.sum(mask, axis=-1, keepdims=True)
            jets = np.concatenate([jets, num_csts], axis=-1)

            # Save the data to an HDF file
            with h5py.File(dest_file, "w") as f:
                f.create_dataset("csts", data=sel_csts)
                f.create_dataset("csts_id", data=csts_id)
                f.create_dataset("jets", data=jets)
                f.create_dataset("labels", data=labels)
                f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    main()
