import numpy as np


def random_pt_masking(
    csts: np.ndarray,
    mask: np.ndarray,
    mask_fraction: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total pt."""
    # Create the random number generator
    rng = np.random.default_rng(seed)

    # Calculate the total pt and the pt fraction
    n_csts = mask.sum()
    pt = csts[mask, 0]
    drop_pt = mask_fraction * pt.sum()

    # Create a random order to cycle through the constituents
    order = rng.permutation(n_csts)

    # Go through each constituent
    running_pt = 0
    drop_mask = np.zeros(n_csts, dtype=bool)
    for i in order:
        # Check if the proposed addition would exceed the drop pt
        if running_pt + pt[i] > drop_pt:
            continue
        # If not, add it to the mask
        running_pt += pt[i]
        drop_mask[i] = True

    # Unravel the drop mask to the full mask
    null_mask = np.zeros_like(mask)
    null_mask[mask] = drop_mask

    return null_mask


def random_masking(
    csts: np.ndarray,
    mask: np.ndarray,
    mask_fraction: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    # Create the random number generator, the number to drop and the mask
    rng = np.random.default_rng(seed)
    max_drop = np.floor(mask_fraction * mask.sum()).astype(int)
    null_mask = np.full_like(mask, False)

    # Exit now if we are not dropping any nodes
    if max_drop == 0:
        return null_mask

    # Generate a random number per node, the lowest frac will be killed
    rand = rng.uniform(size=len(mask))
    rand[~mask] = 9999  # Padded nodes shouldnt be dropped

    # Get the indices for which to sort the random values
    drop_idx = np.argsort(rand)[:max_drop]

    # Create the null mask by dropping the nodes
    null_mask[drop_idx] = True
    return null_mask


def knn_masking(
    csts: np.ndarray,
    mask: np.ndarray,
    pos_dims: list | tuple | slice | None = None,
    k_min: int = 5,
    k_max: int = 20,
    mask_fraction: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Iteratively remove nodes and its knns."""
    # Default slice
    if pos_dims is None:
        pos_dims = slice(1, 3)

    # Create the random number generator, the number to drop and the mask
    rng = np.random.default_rng(seed)
    max_drop = np.rint(mask_fraction * mask.sum()).astype(int)
    null_mask = np.full_like(mask, False)

    # Extract the positions from the constituents
    positions = csts[:, pos_dims].copy()  # Copy to prevent inplace changes
    positions[~mask] = 99999  # Make the masked ones really far away
    idxes = np.arange(len(mask))
    probs = mask / mask.sum()  # No chance of selecting masked tokens

    # Build the patches iteratively, stop at 50 for safety
    for _ in range(50):
        # Randomly select a REAL constituent and k for dropping
        idx = rng.choice(idxes, p=probs)
        pos = positions[idx]

        # Work out how many nodes to drop
        d_max = min(k_max, max_drop - null_mask.sum())
        if d_max <= k_min:
            break
        k_drop = rng.integers(k_min, d_max)

        # Find the closest k neighbours to this node
        distances = np.linalg.norm(pos - positions, axis=-1)
        drop_idx = np.argsort(distances)[:k_drop]

        # Add the dropped nodes to the mask and check stopping criteria
        null_mask[drop_idx] = True
        null_mask[~mask] = False
        if null_mask.sum() >= max_drop:
            break

    return null_mask


def radius_masking(
    csts: np.ndarray,
    mask: np.ndarray,
    pos_dims: list | tuple | slice | None = None,
    r_min: int = 0.1,
    r_max: int = 0.3,
    mask_fraction: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Iteratively remove nodes and those within a radius."""
    # Default slice
    if pos_dims is None:
        pos_dims = slice(1, 3)

    # Create the random number generator, the number to drop and the mask
    rng = np.random.default_rng(seed)
    max_drop = np.rint(mask_fraction * mask.sum()).astype(int)
    null_mask = np.full_like(mask, False)

    # Extract the positions from the constituents
    positions = csts[:, pos_dims].copy()  # Copy to prevent inplace changes
    positions[~mask] = 99999  # Make the masked ones really far away
    idxes = np.arange(len(mask))
    probs = mask / mask.sum()  # No chance of selecting masked tokens

    # Build the patches iteratively, stop at 50 for safety
    for _ in range(50):
        # Randomly select a REAL constituent and r for dropping
        idx = rng.choice(idxes, p=probs)
        pos = positions[idx]
        r_drop = rng.uniform(r_min, r_max)

        # Drop all nodes within the radius
        distances = np.linalg.norm(pos - positions, axis=-1)
        null_mask[distances < r_drop] = True

        # Check stopping criteria
        null_mask[~mask] = False
        if null_mask.sum() >= max_drop:
            break

    return null_mask
