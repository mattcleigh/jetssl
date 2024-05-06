import awkward as ak
import numpy as np
import uproot
import vector
from awkward.highlevel import Array


def lifetime_signing(
    d0: np.ndarray,
    z0: np.ndarray,
    tracks: np.ndarray,
    jets: np.ndarray,
    is_centered: bool = False,
) -> np.ndarray:
    """Change to the lifetime signing convention for the impact parameters.

    See the DIPS paper for details:
    https://cds.cern.ch/record/2718948/files/ATL-PHYS-PUB-2020-014.pdf
    """
    # Break the tracks and jets into their components
    trk_pt = tracks[..., 0]
    trk_eta = tracks[..., 1]
    trk_phi = tracks[..., 2]
    jet_pt = jets[..., 0:1]  # Slice allows for broadcasting
    jet_eta = jets[..., 1:2]
    jet_phi = jets[..., 2:3]

    # If the track coordinates are centered this must be undone
    if is_centered:
        trk_eta = trk_eta + jet_eta
        trk_phi = trk_phi + jet_phi

    # Create the vectors needed for the cross product
    trk = vector.array({"pt": trk_pt, "eta": trk_eta, "phi": trk_phi})
    jet = vector.array({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi})
    dm = vector.array({
        "x": d0 * np.cos(trk_phi - np.pi / 2),
        "y": d0 * np.sin(trk_phi - np.pi / 2),
        "z": z0,
    })

    # Apply the signs using the formula from Rachel's code
    signed_d0 = np.abs(d0) * np.sign((jet.cross(trk)).dot(trk.cross(-dm)))
    signed_z0 = np.abs(z0) * np.sign(z0 * (jet_eta - trk_eta))

    return signed_d0, signed_z0


def get_treename(file: str) -> str:
    with uproot.open(file) as f:
        # Get all the trees in the file
        treenames = {
            k.split(";")[0]
            for k, v in f.items()
            if getattr(v, "classname", "") == "TTree"
        }
    if len(treenames) == 1:
        treename = treenames.pop()
    else:
        raise RuntimeError(
            "Need to specify `treename` as more than one tree is found in file:",
            f"{file}: {treenames}",
        )
    return treename


def read_shlomi_file(
    filepath: str,
    num_particles: int = 15,
    track_features: list | None = None,
    jet_features: list | None = None,
    vertex_features: list | None = None,
) -> tuple:
    """Read a Shlomi root file and returns the jets, tracks, labels, and vertices."""
    # If the feature list is none then use the default features
    if track_features is None:
        track_features = [
            "trk_pt",
            "trk_eta",
            "trk_phi",
            "trk_d0",
            "trk_d0err",
            "trk_z0",
            "trk_z0err",
            "trk_charge",
            "trk_pdg_id",
            "trk_vtx_index",
        ]
    if jet_features is None:
        jet_features = [
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_M",
            "n_trks",
        ]
    if vertex_features is None:
        vertex_features = [
            "true_vtx_x",
            "true_vtx_y",
            "true_vtx_z",
            "true_vtx_L3D",
        ]

    with uproot.open(filepath) as f:
        tree = f["tree"]
        jets = tree.arrays(expressions=jet_features, library="pd")
        labels = tree.arrays(expressions="jet_flav", library="pd")
        tracks = tree.arrays(expressions=track_features, library="ak")
        vertices = tree.arrays(expressions=vertex_features, library="ak")

    # Convert to padded numpy arrays
    tracks = ak_to_numpy_padded(tracks, max_len=num_particles)
    vertices = ak_to_numpy_padded(vertices)
    jets = jets.to_numpy().astype("f")
    labels = labels.to_numpy().astype("l")

    return jets, tracks, labels, vertices


def read_jetclass_file(
    filepath: str,
    num_particles: int = 128,
    particle_features: list | None = None,
    jet_features: list | None = None,
    jet_labels: list | None = None,
    treename: str | None = None,
) -> tuple:
    """Read a JetClass root file and returns the jets, particles, and labels."""
    # If the feature list is none then use the default features
    if particle_features is None:
        particle_features = [
            "part_px",
            "part_py",
            "part_deta",
            "part_dphi",
            "part_d0val",
            "part_d0err",
            "part_dzval",
            "part_dzerr",
            "part_charge",
            "part_isPhoton",
            "part_isNeutralHadron",
            "part_isChargedHadron",
            "part_isElectron",
            "part_isMuon",
        ]
    if jet_features is None:
        jet_features = [
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_sdmass",
        ]
    if jet_labels is None:
        jet_labels = [
            "label_QCD",
            "label_Tbl",
            "label_Tbqq",
            "label_Wqq",
            "label_Zqq",
            "label_Hbb",
            "label_Hcc",
            "label_Hgg",
            "label_H4q",
            "label_Hqql",
        ]

    # If the treename is not given, get it from the file
    treename = treename or get_treename(filepath)

    # Open the file and load all features
    with uproot.open(filepath) as f:
        tree = f[treename]
        jets = tree.arrays(expressions=jet_features, library="pd")
        labels = tree.arrays(expressions=jet_labels, library="pd")
        csts = tree.arrays(expressions=particle_features, library="ak")

    # Convert to numpy arrays
    csts = ak_to_numpy_padded(csts, max_len=num_particles)
    jets = jets.to_numpy().astype("f")
    labels = labels.to_numpy().astype("l").argmax(axis=1)  # Undo one-hot encoding

    return jets, csts, labels


def ak_to_numpy_padded(
    arr: Array, max_len: int | None = None, features: list | None = None
) -> np.ndarray:
    if features is None:  # If no features are given, use all of them
        features = arr.fields
    if max_len is None:  # If no max_len is given, use the maximum length of the array
        max_len = int(ak.max(ak.num(arr[arr.fields[0]])))
    arr = ak.fill_none(ak.pad_none(arr, max_len, clip=True), 0)
    arr = [ak.to_numpy(arr[f]).astype("f").data for f in features]
    arr = np.stack(arr, axis=-1)
    return np.nan_to_num(arr, nan=0.0)


def common_particle_class(
    charge: np.ndarray,
    pdgid: np.ndarray | None = None,
    isPhoton: np.ndarray | None = None,
    isHadron: np.ndarray | None = None,
    isElectron: np.ndarray | None = None,
    isMuon: np.ndarray | None = None,
) -> None:
    """Converts a collection of particle IDs into the common particle class labels.

    - 0: part_isPhoton
    - 1: part_isHadron_Neg
    - 2: part_isHadron_Neutral
    - 3: part_isHadron_Pos
    - 4: part_isElectron_Neg
    - 5: part_isElectron_Pos
    - 6: part_isMuon_Neg
    - 7: part_isMuon_Pos

    This allows us to be completely consistent with particle classes from different
    datasets in a way that removes redundancy (like having both isPhoton and charge).
    Also in a real dataset we could not know types of hadrons.
    """
    # If the pdgid is given, use it to determine the particle type
    if pdgid is not None:
        abs_pdgid = np.abs(pdgid)
        isPhoton = abs_pdgid == 22
        isElectron = abs_pdgid == 11
        isMuon = abs_pdgid == 13
        isHadron = abs_pdgid > 100  # Placeholder now but it works with Shlomi and JC

    # Create new class labels based on the particle type and charge
    label = np.zeros(charge.shape, int)
    label[isPhoton] = 0
    label[isHadron & (charge == -1)] = 1
    label[isHadron & (charge == 0)] = 2
    label[isHadron & (charge == 1)] = 3
    label[isElectron & (charge == -1)] = 4
    label[isElectron & (charge == 1)] = 5
    label[isMuon & (charge == -1)] = 6
    label[isMuon & (charge == 1)] = 7
    return label


def csts_to_jet(csts: np.ndarray, mask: np.ndarray) -> tuple:
    """Calculate high level jet variables using only the constituents."""
    # Split the csts into the different groups of information
    cst_px = csts[..., 0] * mask
    cst_py = csts[..., 1] * mask
    cst_pz = csts[..., 2] * mask
    cst_e = np.sqrt(cst_px**2 + cst_py**2 + cst_pz**2)

    # Calculate the total jet kinematics
    jet_px = cst_px.sum(axis=-1)
    jet_py = cst_py.sum(axis=-1)
    jet_pz = cst_pz.sum(axis=-1)
    jet_e = cst_e.sum(axis=-1)

    # Calculate the total jet mass
    jet_m = np.sqrt(np.maximum(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0))

    return np.vstack([jet_px, jet_py, jet_pz, jet_m]).T


def pxpypz_to_ptetaphi(kinematics: np.ndarray) -> np.ndarray:
    """Convert from cartesian to ATLAS co-ordinates."""
    # Split the kinematics into the different components
    px = kinematics[..., 0:1]
    py = kinematics[..., 1:2]
    pz = kinematics[..., 2:3]

    pt = np.sqrt(px**2 + py**2)
    mtm = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(np.clip(pz / (mtm + 1e-8), -0.9999, 0.9999))
    phi = np.arctan2(py, px)

    return np.concatenate([pt, eta, phi], axis=-1)
