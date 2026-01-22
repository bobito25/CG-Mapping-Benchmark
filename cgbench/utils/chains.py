"""
Chain/trajectory splitting and filtering utilities.
"""

import numpy as np
from jax import numpy as jnp
import jax


def get_line_locations(t_eq, t_tot, n_chains, print_every=0.5):
    """
    Compute chain boundary indices from equilibration time and total time.

    Parameters
    ----------
    t_eq : float
        Equilibration time
    t_tot : float
        Total simulation time
    n_chains : int
        Number of chains
    print_every : float
        Output interval

    Returns
    -------
    np.ndarray
        1D array of frame indices marking chain starts.
    """
    steps = int((t_tot - t_eq) / print_every)
    arr = np.arange(0, steps * n_chains, steps)
    return arr[1:]


def compute_line_locations(config: dict[str, float]) -> np.ndarray:
    """
    Compute chain boundary indices from simulation configuration.

    Based on total simulation time, equilibration time, output interval,
    and number of chains, returns the indices where each chain restarts.

    Parameters
    ----------
    config : dict[str, float]
        Simulation parameters including:
        - 't_total': total time (float)
        - 't_eq': equilibration time (float)
        - 'print_every': output interval (float, default 0.5)
        - 'n_chains': number of chains (int, default 1)

    Returns
    -------
    np.ndarray
        1D integer array of frame indices marking chain starts.
    """
    t_total = config["t_total"]
    t_eq = config["t_eq"]
    print_every = config.get("print_every", 0.5)
    n_chains = config.get("n_chains", 1)
    steps = int((t_total - t_eq) / print_every)
    arr = np.arange(0, steps * n_chains, steps)
    return arr[1:]


def split_into_chains(data: np.ndarray, line_locations: list[int]) -> np.ndarray:
    """
    Split array data into separate chains using boundary indices.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n, ...) to split along first axis.
    line_locations : list[int]
        Indices at which to split the array.

    Returns
    -------
    np.ndarray
        Array of shape (n_chains, segment_length, ...) after splitting.
    """
    segments: list[np.ndarray] = []
    start = 0
    for loc in line_locations:
        segments.append(data[start:loc])
        start = loc
    segments.append(data[start:])
    return np.array(segments)


def setup_distance_filter_fn(ref_means, disp_fn, delta=0.05):
    """
    Creates a distance filter function to validate and filter trajectory frames
    based on distance constraints between specified atom pairs.

    Parameters
    ----------
    ref_means : list of float
        Reference mean distances for each atom pair.
    disp_fn : callable
        Function to compute displacement vectors.
    delta : float, optional
        Tolerance for distance constraints. Defaults to 0.05 nm.

    Returns
    -------
    callable
        A function that filters trajectory frames based on distance constraints.
    """
    from cgbench.utils.geometry import compute_atom_distance

    def apply_distance_filter(traj, indices, verbose=True, print_every=0.5):
        """
        Filter multi-chain trajectory frames based on distance constraints.

        Args:
            traj: JAX array of shape (n_chains, n_frames, n_atoms, 3).
            indices: List of atom index pairs to form distances.
            verbose: Print shape information.

        Returns:
            traj: Shape (n_chains, n_frames, n_atoms, 3) with NaN padding after violations.
            combined_mask: Boolean mask of shape (n_chains, n_frames).
        """
        assert all(len(pair) == 2 for pair in indices), "indices elements must be pairs."

        if verbose:
            print("Input shape:", traj.shape)

        def filter_single_chain(chain_traj):
            chain_mask = jnp.ones(chain_traj.shape[0], dtype=bool)

            for (i, j), mean_dist in zip(indices, ref_means):
                distances = compute_atom_distance(chain_traj, i, j, disp_fn)

                distance_mask = (distances > (mean_dist - delta)) & (
                    distances < (mean_dist + delta)
                )

                chain_mask = chain_mask & distance_mask

            cumulative_mask = jnp.cumprod(chain_mask, axis=0).astype(bool)

            masked_chain_traj = jnp.where(
                cumulative_mask[:, None, None], chain_traj, jnp.nan
            )

            return masked_chain_traj, cumulative_mask

        mapped_fn = jax.vmap(filter_single_chain)
        filtered_traj, final_mask = mapped_fn(traj)

        if verbose:
            valid_counts = jnp.sum(final_mask, axis=1)
            print(f">> Processed shape: {filtered_traj.shape}")
            avg_length = jnp.mean(valid_counts) * print_every
            std_length = jnp.std(valid_counts) * print_every
            print(f">> Valid frames per chain ns {avg_length:.1f} ± {std_length:.1f}")

            chains_with_invalid_frames = jnp.sum(jnp.any(~final_mask, axis=1))
            print(
                f">> Chains with at least one invalid frame: {chains_with_invalid_frames}/{traj.shape[0]}"
            )

        return filtered_traj, final_mask

    return apply_distance_filter


def mark_nan(traj: np.ndarray, aux: np.ndarray, threshold: float = 5, verbose=False) -> np.ndarray:
    """
    Mark trajectory data as NaN from the point where auxiliary data exceeds threshold.

    For each chain, if any auxiliary value exceeds the threshold, all trajectory data
    from that point onward in that chain is set to NaN.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (n_chains, n_frames, n_atoms, 3)
    aux : np.ndarray
        Auxiliary data array of shape (n_chains, n_frames)
    threshold : float
        Threshold value for auxiliary data

    Returns
    -------
    np.ndarray
        Modified trajectory with NaN values where aux exceeded threshold
    """
    traj_marked = traj.copy()

    for chain_idx in range(aux.shape[0]):
        exceeded = aux[chain_idx] > threshold

        if np.any(exceeded):
            if verbose:
                first_exceed_frame = np.argmax(exceeded)
                print(
                    f"Chain {chain_idx}: Aux exceeded threshold at frame {first_exceed_frame}"
                )
            first_exceed_frame = np.argmax(exceeded)

            traj_marked[chain_idx, first_exceed_frame:] = np.nan

    return traj_marked


def calculate_stability(traj_marked: np.ndarray, print_every=0.5):
    """
    Calculate stability metrics from a trajectory with NaN values.

    Computes the mean and standard deviation of the number of valid frames
    (before NaN occurs) for each chain in the trajectory.

    Parameters
    ----------
    traj_marked : np.ndarray
        Trajectory array of shape (n_chains, n_frames, n_atoms, 3)
        with NaN values marking unstable regions
    print_every : float
        Time interval between frames

    Returns
    -------
    tuple
        (mean_length, std_length) - Mean and std of frames before NaN occurs
    """
    n_chains, n_frames = traj_marked.shape[0], traj_marked.shape[1]
    valid_lengths = []

    for chain_idx in range(n_chains):
        # Check if any NaN exists in this chain (check first coordinate of first atom)
        chain_data = traj_marked[chain_idx, :, 0, 0]
        nan_mask = np.isnan(chain_data)

        if np.any(nan_mask):
            # Find first NaN occurrence
            first_nan_frame = np.argmax(nan_mask)
            valid_lengths.append(first_nan_frame * print_every)
        else:
            # Chain remained stable throughout
            valid_lengths.append(n_frames * print_every)

    mean_length = np.mean(valid_lengths)
    std_length = np.std(valid_lengths)

    return mean_length, std_length


def compute_bond_metrics(traj: np.ndarray, disp_fn) -> tuple:
    """
    Compute bond metrics using JAX vmap for displacement calculations.

    Computes nearest neighbor distances, bond switching events, and
    mutual nearest neighbor pairs for each frame.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (n_frames, n_particles, 3)
    disp_fn : callable
        Displacement function for computing distances

    Returns
    -------
    tuple
        (cumulative_fraction, nearest_neighbor_distances, bonds_per_frame)
        - cumulative_fraction: Fraction of particles that have switched neighbors
        - nearest_neighbor_distances: NN distances per particle
        - bonds_per_frame: List of bond pairs per frame
    """
    n_frames = traj.shape[0]
    n_particles = traj.shape[1]

    # Convert to JAX array for vmap operations
    traj_jax = jnp.array(traj)

    # Vectorize displacement function over particles (j dimension)
    disp_vmap_j = jax.vmap(disp_fn, in_axes=(None, 0))
    # Vectorize over frames
    disp_vmap_frames = jax.vmap(disp_vmap_j, in_axes=(0, 0))

    # Compute all pairwise displacements and distances
    dist_matrix = np.full((n_frames, n_particles, n_particles), np.inf)

    # Loop over i particles, but vectorize over frames and j particles
    for i in range(n_particles):
        # For particle i, compute distances to all other particles across all frames
        Ra = traj_jax[:, i, :]  # [n_frames, 3]
        Rb = traj_jax  # [n_frames, n_particles, 3]

        # Compute displacements for all j particles at once, across all frames
        disp = disp_vmap_frames(Ra, Rb)  # [n_frames, n_particles, 3]
        distances = jnp.linalg.norm(disp, axis=-1)  # [n_frames, n_particles]

        dist_matrix[:, i, :] = np.array(distances)

    # Set diagonal to infinity
    dist_matrix[:, np.arange(n_particles), np.arange(n_particles)] = np.inf

    # Find nearest neighbors
    nn_idx = np.argmin(dist_matrix, axis=-1)  # [n_frames, n_particles]
    nn_dist = np.take_along_axis(
        dist_matrix, nn_idx[..., np.newaxis], axis=-1
    ).squeeze(-1)

    # Detect switches
    switches = np.diff(nn_idx, axis=0, prepend=nn_idx[0:1]) != 0
    ever_switched = np.cumsum(switches, axis=0) > 0
    cumulative_fraction = np.sum(ever_switched, axis=1) / float(n_particles)
    cumulative_fraction[0] = 0.0

    # Find mutual nearest neighbors (bonds)
    mutual_nn = (
        nn_idx[np.arange(n_frames)[:, None], nn_idx] == np.arange(n_particles)
    )

    bonds_per_frame = []
    for f in range(n_frames):
        i_idx = np.where(mutual_nn[f])[0]
        j_idx = nn_idx[f, i_idx]
        valid = i_idx < j_idx
        pairs = np.column_stack([i_idx[valid], j_idx[valid]])
        bonds_per_frame.append(pairs.tolist())

    nearest_neighbor_distances = nn_dist.T

    return cumulative_fraction, nearest_neighbor_distances, bonds_per_frame
