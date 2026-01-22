"""
File I/O utilities for trajectory loading, XYZ writing, and output directory preparation.
"""

import os
import io
import pickle as pkl
import numpy as np
from jax import numpy as jnp
from concurrent.futures import ProcessPoolExecutor


def prepare_output_dir(traj_path: str) -> str:
    """
    Create an output directory named 'plots' next to a trajectory file.

    Ensures that a directory called 'plots' exists alongside the given
    trajectory file path. If it does not exist, it is created.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory file.

    Returns
    -------
    str
        Path to the 'plots' directory where outputs will be saved.
    """
    outdir = os.path.join(os.path.dirname(traj_path), "plots")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_trajectory(traj_path: str) -> tuple[jnp.ndarray, dict]:
    """
    Load trajectory coordinates and auxiliary state from pickle files.

    Opens 'trajectory.pkl' and 'traj_state_aux.pkl' in the same directory as
    the provided path, and returns the trajectory as a JAX array along with
    auxiliary simulation data.

    Parameters
    ----------
    traj_path : str
        Path to one of the trajectory pickle files.

    Returns
    -------
    tuple[jnp.ndarray, dict]
        traj : JAX array of shape (n_frames, n_particles, 3)
            Simulation trajectory coordinates.
        aux : dict
            Auxiliary state information (energy, temperature, etc.).
    """
    base = os.path.dirname(traj_path)
    traj = pkl.load(open(os.path.join(base, "trajectory.pkl"), "rb"))
    aux = pkl.load(open(os.path.join(base, "traj_state_aux.pkl"), "rb"))
    return jnp.array(traj), aux


def _format_xyz_frame(args):
    """Worker: format one frame to an XYZ string."""
    frame_idx, positions_frame, species_col = args
    n_atoms = positions_frame.shape[0]
    buf = io.StringIO()
    buf.write(f"{n_atoms}\nFrame {frame_idx + 1}\n")
    data = np.c_[species_col, positions_frame]
    np.savetxt(buf, data, fmt="%s %.6f %.6f %.6f")
    return buf.getvalue()


def save_xyz_frames_parallel(
    positions,
    species_list,
    filename,
    workers=None,
    chunksize=8,
    buffer_bytes=1_048_576,
):
    """
    Parallel XYZ writer.
    - Parallelizes CPU-bound text formatting per frame with processes.
    - Preserves frame order in the output file.
    - Streams results to disk to avoid large memory spikes.

    positions: (n_frames, n_atoms, 3) float array
    species_list: list[str] length n_atoms
    """
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions.shape
    if len(species_list) != n_atoms:
        raise ValueError(
            f"Species list length ({len(species_list)}) must match number of atoms ({n_atoms})"
        )

    # Cache species column once; small and cheap to pickle
    species_col = np.asarray(species_list, dtype=object).reshape(-1, 1)

    # Small datasets don't benefit from process spin-up
    if workers == 1 or n_frames < 4:
        with open(filename, "w", buffering=buffer_bytes) as f:
            for frame_idx in range(n_frames):
                f.write(
                    _format_xyz_frame((frame_idx, positions[frame_idx], species_col))
                )
        return

    # Parallel formatting
    with open(filename, "w", buffering=buffer_bytes) as f, ProcessPoolExecutor(
        max_workers=workers
    ) as ex:
        iterable = ((i, positions[i], species_col) for i in range(n_frames))
        for frame_str in ex.map(_format_xyz_frame, iterable, chunksize=chunksize):
            f.write(frame_str)


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")

    if fractional:
        box = dataset["box"][0, 0, 0]
        dataset["R"] = dataset["R"] / box
    else:
        dataset["R"] = dataset["R"] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset["box"] = scale_R * dataset["box"]
    dataset["F"] *= scale_F

    return dataset
