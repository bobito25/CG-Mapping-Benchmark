"""
Geometry calculation utilities (dihedrals, angles, distances).
"""

import numpy as np

from jax import numpy as jnp
from jax import vmap
import jax
from typing import Callable
from jax_md_mod import custom_quantity
from jax_md import space


def periodic_displacement(
    box: np.ndarray, fractional: bool = False
) -> tuple[callable, None]:
    """
    Create a periodic displacement function for simulating boundary conditions.

    Uses JAX MD's periodic_general to produce a function that calculates
    displacement vectors under periodic boundary conditions for a given box.

    Parameters
    ----------
    box : np.ndarray
        Array or matrix defining the simulation box.
    fractional : bool, optional
        Whether input coordinates are in fractional units, by default False.

    Returns
    -------
    tuple[callable, None]
        A function to compute periodic displacements and a placeholder None.
    """
    return space.periodic_general(box=box, fractional_coordinates=fractional)


def init_dihedral_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute dihedral angles from trajectory positions.

    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Four atom indices defining the dihedral angle

    Returns
    -------
    Callable
        Function that takes positions and returns dihedral angles
    """
    idcs = jnp.array(idcs)

    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_dihedrals = jax.vmap(
            custom_quantity.dihedral_displacement, (0, None, None)
        )
        dihedral_angles = batched_dihedrals(positions, displacement_fn, idcs)
        return dihedral_angles.T

    return postprocess_fn


def init_angle_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute bond angles from trajectory positions.

    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Three atom indices defining the bond angle

    Returns
    -------
    Callable
        Function that takes positions and returns bond angles
    """
    idcs = jnp.array(idcs)

    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_angles = jax.vmap(custom_quantity.angular_displacement, (0, None, None))
        angles = batched_angles(positions, displacement_fn, idcs)
        return angles.T

    return postprocess_fn


def compute_atom_distance(
    coords: jnp.ndarray,
    idx1: int,
    idx2: int,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute the PBC-aware distance between two atoms over all trajectory frames.

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    displacement_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Function that computes PBC-corrected displacement between two position vectors

    Returns
    -------
    jnp.ndarray
        Array of scalar distances with shape (n_frames,)
    """
    r1 = coords[:, idx1, :]
    r2 = coords[:, idx2, :]

    disp = vmap(displacement_fn)(r1, r2)
    distances = jnp.linalg.norm(disp, axis=-1)

    return distances


def calculate_dihedral(
    p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate dihedral angle between four points in degrees.

    Parameters
    ----------
    p0 : jnp.ndarray
        Position vector of the first atom with shape (3,)
    p1 : jnp.ndarray
        Position vector of the second atom with shape (3,)
    p2 : jnp.ndarray
        Position vector of the third atom with shape (3,)
    p3 : jnp.ndarray
        Position vector of the fourth atom with shape (3,)

    Returns
    -------
    jnp.ndarray
        Dihedral angle in degrees (scalar)
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= jnp.linalg.norm(b1)

    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)

    return jnp.degrees(jnp.arctan2(y, x))


def calc_mse_dihedrals(
    phi_ref: jnp.ndarray,
    psi_ref: jnp.ndarray,
    phi_sim: jnp.ndarray,
    psi_sim: jnp.ndarray,
    nbins: int = 60,
) -> float:
    """
    Calculate mean squared error between reference and simulation dihedral angle distributions.

    Parameters
    ----------
    phi_ref : jnp.ndarray
        Reference phi dihedral angles in degrees
    psi_ref : jnp.ndarray
        Reference psi dihedral angles in degrees
    phi_sim : jnp.ndarray
        Simulation phi dihedral angles in degrees
    psi_sim : jnp.ndarray
        Simulation psi dihedral angles in degrees
    nbins : int, optional
        Number of bins for 2D histogram (default: 60)

    Returns
    -------
    float
        Mean squared error between the two 2D density histograms
    """
    phi_ref_rad, psi_ref_rad = jnp.deg2rad(phi_ref), jnp.deg2rad(psi_ref)
    phi_sim_rad, psi_sim_rad = jnp.deg2rad(phi_sim), jnp.deg2rad(psi_sim)

    h_ref, _, _ = np.histogram2d(phi_ref_rad, psi_ref_rad, bins=nbins, density=True)
    h_sim, _, _ = np.histogram2d(phi_sim_rad, psi_sim_rad, bins=nbins, density=True)

    mse = np.mean((h_ref - h_sim) ** 2)
    print("MSE of the phi-psi dihedral density histogram:", mse)
    return mse
