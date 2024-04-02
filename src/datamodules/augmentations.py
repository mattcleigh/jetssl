"""Defines a set of jet augmentations that can be applied to the jet constituents.

All constituent arrays are assumed to be in the format [pt, (del)eta, (del)phi] Pt
should be in GeV for the smearing and merge operations!!!
"""

import random

import numpy as np

EPS = 1e-8


class Augmentation:
    """Base class for all augmentations."""

    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def __call__(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        """Check if the augmentation should be applied."""
        if self.prob < 1 and self.prob > random.random():
            return csts, mask
        return self.apply(csts, mask)

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        """Apply the augmentation to the jet."""
        raise NotImplementedError


class AugmentationSequence(Augmentation):
    def __init__(self, aug_list: list[Augmentation]) -> None:
        super().__init__(prob=1.0)
        self.aug_list = aug_list

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        for aug in self.aug_list:
            csts, mask = aug(csts, mask)
        return csts, mask


class Crop(Augmentation):
    """Drop the last 'amount' of constituents in the jet.

    Will not drop constituents passed the minimum allowed level
    """

    def __init__(
        self, prob: float = 1.0, amount: int = 32, min_allowed: int = 10
    ) -> None:
        super().__init__(prob)
        self.amount = amount
        self.min_allowed = min_allowed

    def __call__(self, csts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.prob < 1 and self.prob > random.random():
            return csts, mask

        csts = csts.copy()
        mask = mask.copy()

        # Check how many we can drop
        n_csts = np.sum(mask)
        allowed_to_drop = min(n_csts - self.min_allowed, self.amount)

        # Generate randomly the number of nodes to kill if allowed to drop
        if allowed_to_drop > 0:
            drop_num = np.random.default_rng().randint(allowed_to_drop + 1)

            # Drop them from the mask and the node features
            mask[n_csts - drop_num : n_csts] = False
            csts[n_csts - drop_num : n_csts] = 0

        return csts, mask


class Smear(Augmentation):
    """Add noise to the constituents eta and phi to simulates soft emmisions.

    Noise is gaussian with mean = 0 and deviation = strength/pT
    The default strength for the blur is set to 100 MeV
    The default minimum smearing value is 500 MeV
    - https://arxiv.org/pdf/2108.04253.pdf
    """

    def __init__(
        self, prob: float = 1.0, strength: float = 0.1, pt_min: float = 0.5
    ) -> None:
        super().__init__(prob)
        self.strength = strength
        self.pt_min = pt_min

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Create a copy of the arrays to prevent inplace changes
        csts = csts.copy()

        # Get the smearing strength inversly proportional to the pt
        pt = np.clip(csts[mask, 0:1], a_min=self.pt_min, a_max=None)
        strength = self.strength / pt

        # Get noise and add it to the eta phi
        smear = np.random.default_rng().standard_normal(pt.shape[0], 2) * strength
        csts[mask, 1:] += smear

        return csts, mask


class CollinearSplit(Augmentation):
    """Split some of the constituents into a pair ontop of each other.

    Will not split to exceed the padding limit.
    """

    def __init__(
        self, prob: float = 1, max_splits: int = 50, min_pt_spit: float = 0.5
    ) -> None:
        super().__init__(prob)
        self.max_splits = max_splits
        self.min_pt_spit = min_pt_spit

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        # Make a copy of the arrays to prevent inplace changes
        csts = csts.copy()
        mask = mask.copy()

        # Has to be zero paded!
        csts[~mask] = 0

        # See how many constituents can be split
        rng = np.random.default_rng()
        n_csts = np.sum(mask)
        n_splittable = np.sum(csts[:, 0] > self.min_pt_spit)
        n_to_split = min([self.max_splits, csts.shape[0] - n_csts, n_splittable])
        n_to_split = rng.randint(n_to_split + 1)

        # If splitting will take place
        if n_to_split > 0:
            # Randomly choose the indexes of those to split
            idx_to_split = rng.choice(n_splittable, n_to_split, replace=False)

            # Their new indecies are added to the end of the array
            new_idxes = np.arange(n_to_split) + n_csts

            # Generate the splitting momentum fractions from uniform [0.25, 0.75]
            frc_of_splits = rng.random(n_to_split) / 2 + 0.25

            # Add new particles on the end of the array with the same values
            csts[new_idxes] = csts[idx_to_split].copy()
            csts[new_idxes, 0] *= frc_of_splits  # Reduce the pt

            # Subtract the pt fraction from the original locations
            csts[idx_to_split, 0] *= 1 - frc_of_splits

            # Sort the constituents with respect to pt again
            csts = csts[csts[:, 0].argsort()[::-1]]

        # Update the mask to reflect the new additions
        mask = csts[:, 0] > 0

        return csts, mask


class Merge(Augmentation):
    """Randomly merge soft constituents if they are close enough together."""

    def __init__(
        self,
        prob: float = 1.0,
        max_pt: float = 5.0,
        max_del_r: float = 0.05,
        min_allowed: int = 16,
    ) -> None:
        """Parameters
        ----------
        prob : float
            The probability of applying the augmentation
        max_del_r : float
            Maximum seperation to consider merging
        max_pt : float
            The maximum pt to consider merging
        min_allowed : int
            Will not do any merging that may reduce the constituent count past
            this number
        """
        super().__init__(prob)
        self.max_pt = max_pt
        self.max_del_r = max_del_r
        self.min_allowed = min_allowed

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        # Create a copy of the arrays to prevent inplace changes
        csts = csts.copy()
        mask = mask.copy()

        # Count the number of constituents and how many we can drop
        num_csts = mask.sum()
        allowed_to_merge = num_csts - self.min_allowed
        if allowed_to_merge <= 0:
            return csts, mask
        num_to_merge = np.random.default_rng().randint(allowed_to_merge + 1)

        # Break up the constituents into their respective coordinates
        pt = csts[..., 0]
        eta = csts[..., 1]
        phi = csts[..., 2]
        eta_phi = csts[..., 1:3]  # Quick access to the eta and phi

        # Create a sum pt matrix and check where it is below the max
        sum_pt = np.expand_dims(pt, 0) + np.expand_dims(pt, 1)
        pos_merges = sum_pt < self.max_pt
        pos_merges = np.triu(pos_merges)  # Make it upper triangular

        # Make the diagonal and masked portions not possible to merge
        np.fill_diagonal(pos_merges, False)
        pos_merges[~mask, :] = False
        pos_merges[:, ~mask] = False

        # Return if no possible merges
        if not np.any(pos_merges):
            return csts, mask

        # Create a distance matrix for all constituents
        del_r = np.expand_dims(eta_phi, 0) - np.expand_dims(eta_phi, 1)
        del_r = np.linalg.norm(del_r, axis=-1)

        # Include into the possible merges those close enough
        pos_merges = pos_merges & (del_r < self.max_del_r)

        # Return if no possible merges
        if not np.any(pos_merges):
            return csts, mask

        # Make an index array for the possible merges and select only top candidates
        idx_merges = np.argwhere(pos_merges)
        np.random.default_rng().shuffle(idx_merges)

        # Loop through all candidates and merge them using the 4 momenta
        previous = []  # For checking if we have already used a constituent
        i_idx = []  # Idxes for the results of the combine
        j_idx = []  # Idxes for the nodes to delete in the combine
        for i, j in idx_merges:
            if i not in previous and j not in previous:
                previous += [i, j]
                i_idx.append(i)
                j_idx.append(j)
                if len(i_idx) >= num_to_merge:
                    break

        # Combined Momentum
        px = pt[i_idx] * np.cos(phi[i_idx]) + pt[j_idx] * np.cos(phi[j_idx])
        py = pt[i_idx] * np.sin(phi[i_idx]) + pt[j_idx] * np.sin(phi[j_idx])
        pz = pt[i_idx] * np.sinh(eta[i_idx]) + pt[j_idx] * np.sinh(eta[j_idx])

        # Replace element i
        mtm = np.sqrt(px**2 + py**2 + pz**2)
        pt[i_idx] = np.sqrt(px**2 + py**2)
        eta[i_idx] = np.arctanh(np.clip(pz / (mtm + EPS), -1 + EPS, 1 - EPS))
        phi[i_idx] = np.arctan2(py, px)

        # Kill element j
        csts[j_idx] = 0
        mask[j_idx] = False

        # Sort the constituents with respect to pt again
        csts = csts[csts[:, 0].argsort()[::-1]]
        mask = csts[:, 0] > 0

        return csts, mask


class RandomRotate(Augmentation):
    """Rotate all constituents about their eta and phi coordinates."""

    def __init__(self, prob: float = 1.0) -> None:
        super().__init__(prob)

    def apply(self, csts: np.ndarray, mask: np.ndarray) -> tuple:
        # Create a copy of the arrays to prevent inplace changes
        csts = csts.copy()

        # Define the rotation matrix
        angle = np.random.default_rng().random() * 2 * np.pi
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])

        # Apply to variables wrt the jet axis
        eta_phi = csts[mask, 1:3]

        # Modify the constituent with the new variables
        csts[mask, 1:3] = rot_matrix.dot(eta_phi.T).T

        return csts, mask
