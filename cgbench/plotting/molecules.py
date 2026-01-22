"""
High-level molecule visualization routines.
"""

import os
import json
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from chemtrain import quantity

from cgbench.utils.io import load_trajectory, prepare_output_dir
from cgbench.utils.geometry import (
    init_dihedral_fn,
    init_angle_fn,
    compute_atom_distance,
    periodic_displacement,
)
from cgbench.utils.chains import compute_line_locations, split_into_chains
from cgbench.plotting.timeseries import (
    plot_energy_and_kT,
    plot_time_series,
    plot_dist_series,
    plot_dihedrals,
)
from cgbench.plotting.structural import plot_ramachandran
from cgbench.plotting.distributions import plot_1d_dihedral


def plot_hexane_angle(
    angle_indices_all: list[tuple[int, int, int]],
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    outpath: str,
    disp_fn: callable,
) -> None:
    """
    Plot KDE of bond angles across all hexane molecules.

    Calculates angle values for each frame and molecule, then produces two
    density plots: full range [0, π] and zoomed [1.6, π].
    """
    angle_fn = init_angle_fn(disp_fn, angle_indices_all)
    angles_ref = angle_fn(ref_coords)
    angles_traj = angle_fn(traj_coords)
    ref_flat = np.radians(np.concatenate(angles_ref))
    traj_flat = np.radians(np.concatenate(angles_traj))
    ref_clean = ref_flat[np.isfinite(ref_flat)]
    traj_clean = traj_flat[np.isfinite(traj_flat)]

    # Full-range KDE
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        xs = np.linspace(traj_clean.min(), traj_clean.max(), 1000)
        ax1.plot(xs, kde_t(xs), label="Trajectory KDE")
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        xsr = np.linspace(
            min(ref_clean.min(), traj_clean.min()),
            max(ref_clean.max(), traj_clean.max()),
            1000,
        )
        ax1.plot(xsr, kde_r(xsr), "--", label="Reference KDE")
    ax1.set_xlim(0, np.pi)
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Bond Angle KDE: Trajectory vs Reference (Full Range)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(outpath, "bond_angles_density.png"), dpi=300)
    plt.close(fig1)

    # Zoomed KDE
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        ax2.plot(xs, kde_t(xs), label="Trajectory KDE")
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        ax2.plot(xsr, kde_r(xsr), "--", label="Reference KDE")
    ax2.set_xlim(1.6, np.pi)
    ax2.set_xlabel("Angle (radians)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Bond Angle KDE: Trajectory vs Reference (Zoomed: 1.6 to π)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, "bond_angles_density_zoomed.png"), dpi=300)
    plt.close(fig2)


def plot_hex_dihedral(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    dihedral_indices_all: list[tuple[int, int, int, int]],
    outpath: str,
) -> None:
    """
    Plot dihedral angle distributions for all hexane CG dihedrals.

    Computes dihedral angles for every molecule and frame, then overlays
    reference vs simulation histograms on a single panel.
    """
    hex_fn = init_dihedral_fn(disp_fn, dihedral_indices_all)
    CG_angles = np.concatenate(hex_fn(traj_coords))
    AT_angles = np.concatenate(hex_fn(ref_coords))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_1d_dihedral(ax, [AT_angles, CG_angles], ["AT", "Simulation"], bins=60, degrees=True)
    ax.set_title("Dihedral angle (all molecules)")
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "dihedral_angle.png"), dpi=300)
    plt.close(fig)


def plot_bond_angle_correlation(
    ref_coords, traj_coords, angle_idcs, bond_idcs, disp_fn, outpath
):
    """Plot 2D histogram of bond angles vs bond distances."""
    hex_angle_fn = init_angle_fn(disp_fn, angle_idcs)
    angles_ref = hex_angle_fn(ref_coords)
    angles_traj = hex_angle_fn(traj_coords)

    dists_ref = [compute_atom_distance(ref_coords, a, b, disp_fn) for a, b in bond_idcs]
    dists_traj = [
        compute_atom_distance(traj_coords, a, b, disp_fn) for a, b in bond_idcs
    ]

    angles_ref_flat = np.radians(np.concatenate(angles_ref))
    angles_traj_flat = np.radians(np.concatenate(angles_traj))
    dists_ref_flat = np.concatenate(dists_ref)
    dists_traj_flat = np.concatenate(dists_traj)

    # determine how many bonds per angle
    n_angles = len(angle_idcs)
    n_distances = len(bond_idcs)
    if n_angles == 0 or (n_distances % n_angles) != 0:
        raise ValueError(
            f"Expected number of distances ({n_distances}) to be a multiple of number of angles ({n_angles})"
        )
    repeat_factor = n_distances // n_angles

    # repeat angles to align with distances
    angles_ref_rep = np.repeat(angles_ref_flat, repeat_factor)
    angles_traj_rep = np.repeat(angles_traj_flat, repeat_factor)

    # drop any pairs where either is NaN
    mask_ref = np.isfinite(angles_ref_rep) & np.isfinite(dists_ref_flat)
    mask_traj = np.isfinite(angles_traj_rep) & np.isfinite(dists_traj_flat)

    angles_ref_final = angles_ref_rep[mask_ref]
    dists_ref_final = dists_ref_flat[mask_ref]
    angles_traj_final = angles_traj_rep[mask_traj]
    dists_traj_final = dists_traj_flat[mask_traj]

    # make 2D histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hist_ref, xedges_ref, yedges_ref = np.histogram2d(
        angles_ref_final, dists_ref_final, bins=50, density=True
    )
    hist_traj, xedges_traj, yedges_traj = np.histogram2d(
        angles_traj_final, dists_traj_final, bins=50, density=True
    )
    # Plot reference histogram
    extent_ref = [xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]]
    im1 = ax1.imshow(
        hist_ref.T, origin="lower", extent=extent_ref, aspect="auto", cmap="plasma"
    )
    ax1.set_xlabel("Bond Angle (radians)")
    ax1.set_ylabel("Bond Distance (nm)")
    ax1.set_title("Reference: Bond vs Angle")

    extent_traj = [xedges_traj[0], xedges_traj[-1], yedges_traj[0], yedges_traj[-1]]
    im2 = ax2.imshow(
        hist_traj.T, origin="lower", extent=extent_traj, aspect="auto", cmap="plasma"
    )
    ax2.set_xlabel("Bond Angle (radians)")
    ax2.set_ylabel("Bond Distance (nm)")
    ax2.set_title("Trajectory: Bond vs Angle")
    plt.colorbar(im2, ax=ax2, label="Density")

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "bond_angle_correlation_heatmap.png"), dpi=300)
    plt.close(fig)


def vis_ala2(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize alanine dipeptide trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 8, 14]
        psi_indices = [6, 8, 14, 16]
        pairs = [(4, 6), (6, 8), (8, 14)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "heavyOnly": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "heavyOnlyMap2": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreSingle": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
            "coreBetaMap2": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
            "coreBetaSingle": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def vis_hexane(
    traj_path,
    config,
    type="AT",
    name="Simulation",
    dataset=None,
    cg_map="six-site",
    nmol=100,
):
    """Visualize hexane trajectory."""
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    config = json.load(
        open(os.path.join(os.path.dirname(traj_path), "traj_config.json"), "r")
    )
    line_locs = compute_line_locations(config)

    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    # Initialize variables
    cg_dihedral_idcs = None
    CG_angle_idcs = None

    # mapping
    if type == "AT":
        sites_per_mol = 20
        at_dihedral_idcs = [4, 7, 10, 13]
        CC_pairs = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)]
        CH_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (4, 5),
            (4, 6),
            (7, 8),
            (7, 9),
            (10, 11),
            (10, 12),
            (13, 14),
            (13, 15),
            (16, 17),
            (16, 18),
            (16, 19),
        ]
        ref_coords = np.concatenate(
            [
                dataset.dataset_X["training"]["R"],
                dataset.dataset_X["validation"]["R"],
                dataset.dataset_X["testing"]["R"],
            ],
            axis=0,
        )

    else:
        definitions = {
            "two-site": ([(0, 1)], 2, None, None),
            "two-site-Map2": ([(0, 1)], 2, None, None),
            "three-site": ([(0, 1), (1, 2)], 3, [(0, 1, 2)], None),
            "three-site-Map1": ([(0, 1), (1, 2)], 3, [(0, 1, 2)], None),
            "four-site": ([(0, 1), (1, 2), (2, 3)], 4, None, [0, 1, 2, 3]),
            "six-site": ([(1, 2), (2, 3), (3, 4)], 6, None, [1, 2, 3, 4]),
            "six-site-Map2": ([(1, 2), (2, 3), (3, 4)], 6, None, [1, 2, 3, 4]),
        }

        if cg_map not in definitions:
            raise ValueError(
                f"Unknown cg_map: {cg_map}. Available options: {list(definitions.keys())}"
            )

        CC_pairs, sites_per_mol, CG_angle_idcs, cg_dihedral_idcs = definitions[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )

    actual_nmol = config.get("nmol", nmol)
    plot_energy_and_kT(aux, line_locs, outpath)

    if "epot" in aux:
        epot = aux["epot"]
        if np.any(epot > 1000):
            first_explosion = np.where(epot > 1000)[0][0]
            traj_coords = traj_coords[:first_explosion]
            aux = {
                k: v[:first_explosion]
                for k, v in aux.items()
                if isinstance(v, (np.ndarray, list))
            }
            print(f"Energy exceeded 10^4 at frame {first_explosion}, truncating trajectory.")

    CC_all = []
    Dihedrals_idcs_all = []
    Angles_idcs_all = []

    for m in range(actual_nmol):
        offset = m * sites_per_mol
        CC_all.extend([(a + offset, b + offset) for a, b in CC_pairs])

        if cg_dihedral_idcs is not None:
            Dihedrals_idcs_all.extend(
                [
                    (a + offset, b + offset, c + offset, d + offset)
                    for a, b, c, d in [cg_dihedral_idcs]
                ]
            )

        if CG_angle_idcs is not None:
            Angles_idcs_all.extend(
                [(a + offset, b + offset, c + offset) for a, b, c in CG_angle_idcs]
            )

    if type == "AT":
        CH_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            CH_all.extend([(a + offset, b + offset) for a, b in CH_pairs])

        fig_ch, ax_ch = plt.subplots(figsize=(10, 5))
        for a, b in CH_all:
            d = compute_atom_distance(traj_coords, a, b, disp_fn)
            ax_ch.plot(d, alpha=0.1)
        ax_ch.set_title("AT CH distances (all molecules)")
        ax_ch.set_xlabel("Time step")
        ax_ch.set_ylabel("Distance")
        plt.tight_layout()
        fig_ch.savefig(os.path.join(outpath, "AT_CH_distances_all.png"), dpi=300)
        plt.close(fig_ch)

        Dihedral_AT_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            Dihedral_AT_all.extend(
                [
                    (a + offset, b + offset, c + offset, d + offset)
                    for a, b, c, d in [at_dihedral_idcs]
                ]
            )

        plot_hex_dihedral(ref_coords, traj_coords, disp_fn, Dihedral_AT_all, outpath)

    elif cg_map == "three-site":
        plot_hexane_angle(Angles_idcs_all, ref_coords, traj_coords, outpath, disp_fn)
        plot_bond_angle_correlation(
            ref_coords, traj_coords, Angles_idcs_all, CC_all, disp_fn, outpath
        )

    elif "six-site" in cg_map or "four-site" in cg_map:
        if cg_dihedral_idcs is not None:
            plot_hex_dihedral(
                ref_coords, traj_coords, disp_fn, Dihedrals_idcs_all, outpath
            )


def vis_ala15(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize ALA15 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    plot_energy_and_kT(aux, line_locs, outpath)

    if type == "AT":
        raise NotImplementedError("AT visualization for ALA15 is not implemented yet.")
    else:
        maps = {
            "CA": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "CA-Map2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "CA-Map3": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "CA-Map4": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([3, 4, 5, 6], [4, 5, 6, 7], [(4, 5), (5, 6), (6, 7)]),
            "coreBetaMap2": ([4, 5, 6, 8], [5, 6, 8, 9], [(4, 5), (5, 6), (6, 8)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )

    if len(phi_indices) > 0:
        ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
        AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
        Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

        plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
        plot_ramachandran(
            AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath
        )

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)


def vis_pro2(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize PRO2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "heavyOnly": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "heavyOnlyMap2": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 3, 4], [1, 3, 4, 5], [(0, 1), (1, 3), (3, 4)]),
            "coreBetaMap2": ([0, 1, 3, 4], [1, 3, 4, 5], [(0, 1), (1, 3), (3, 4)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def vis_gly2(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize GLY2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 8, 11]
        psi_indices = [6, 8, 11, 13]
        pairs = [(4, 6), (6, 8), (8, 11)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "heavyOnly": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "heavyOnlyMap2": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def vis_thr2(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize THR2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "heavyOnly": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "heavyOnlyMap2": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 3)]),
            "coreBetaMap2": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate(
            [
                dataset.cg_dataset_U["training"]["R"],
                dataset.cg_dataset_U["validation"]["R"],
                dataset.cg_dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)
