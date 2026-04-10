#!/usr/bin/env python3
"""
Plot radial distribution functions (RDF) from trajectory.pkl files for liquid hexane models.

Usage:
    python plot_rdf.py trajectory.pkl [options]

Example:
    python plot_rdf.py trajectory.pkl --output rdf_plot.png --sites 2 --rm-exploded
"""

import argparse
import re
import numpy as np
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cgbench.utils.structural import calculate_rdf
from cgbench.utils.chains import get_line_locations, split_into_chains, mark_nan, calculate_stability
from cgbench.plotting.style import setup_plot_style


# Hexane box length in nm
BOX_LENGTH = 2.79573


def get_bead_types(sites_per_mol: int) -> list:
    """Return bead types based on the number of sites per molecule."""
    if sites_per_mol == 2:
        return ['A', 'A']
    elif sites_per_mol == 3:
        return ['A', 'B', 'A']
    elif sites_per_mol == 4:
        return ['A', 'B', 'B', 'A']
    else:
        raise ValueError(f"Unsupported sites_per_mol: {sites_per_mol}")


def parse_path_for_params(path: str) -> dict:
    """
    Parse trajectory path to extract simulation parameters.
    
    Example path pattern:
    traj_mol=hexane_dt=2_teq=0_t=1000_nmol=100_nchains=50_mode=sampling_seed=22/trajectory.pkl
    """
    params = {
        'n_chains': 1,
        't_eq': 0,
        't_total': 1000,
    }
    
    # Parse nchains
    match = re.search(r'nchains=(\d+)', path)
    if match:
        params['n_chains'] = int(match.group(1))
    
    # Parse t_eq
    match = re.search(r'teq=(\d+)', path)
    if match:
        params['t_eq'] = int(match.group(1))
    
    # Parse t_total
    match = re.search(r'_t=(\d+)_', path)
    if match:
        params['t_total'] = int(match.group(1))
    
    # Parse sites from map= (e.g., map=two-site, map=three-site, map=four-site)
    match = re.search(r'map=(two|three|four)-site', path)
    if match:
        site_mapping = {'two': 2, 'three': 3, 'four': 4}
        params['sites_per_mol'] = site_mapping[match.group(1)]
    
    return params


def load_trajectory(path: str, box_length: float = BOX_LENGTH) -> np.ndarray:
    """Load and scale trajectory from a .pkl file."""
    traj = np.load(path, allow_pickle=True)
    
    # Trajectories are stored in fractional coordinates, scale to real coordinates
    traj = traj * box_length
    
    return traj


def load_aux_data(traj_path: str) -> np.ndarray | None:
    """Load auxiliary data (kT values) from traj_state_aux.pkl."""
    aux_path = traj_path.replace('trajectory.pkl', 'traj_state_aux.pkl')
    if os.path.exists(aux_path):
        aux_data = np.load(aux_path, allow_pickle=True)['kT']
        return aux_data
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot RDF from a liquid hexane trajectory.pkl file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "trajectory", 
        type=str,
        help="Path to trajectory.pkl file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output PNG filename. If not specified, uses trajectory name with .png extension"
    )
    parser.add_argument(
        "--sites", "-s",
        type=int,
        default=None,
        choices=[2, 3, 4],
        help="Number of CG sites per hexane molecule. Auto-detected from path if not specified."
    )
    parser.add_argument(
        "--box-length", "-b",
        type=float,
        default=BOX_LENGTH,
        help="Simulation box length in nm"
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.01,
        help="Bin width for RDF histogram"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output image"
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Output interval in ps (for chain splitting)"
    )
    parser.add_argument(
        "--rm-exploded",
        action="store_true",
        help="Remove exploded simulations using 5 kT threshold from traj_state_aux.pkl"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="kT threshold for detecting exploded simulations (used with --rm-exploded)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")
    
    # Parse parameters from path
    path_params = parse_path_for_params(args.trajectory)
    
    # Determine sites_per_mol
    if args.sites is not None:
        sites_per_mol = args.sites
    elif 'sites_per_mol' in path_params:
        sites_per_mol = path_params['sites_per_mol']
    else:
        sites_per_mol = 2
        print(f"Warning: Could not detect sites_per_mol from path, defaulting to {sites_per_mol}")
    
    n_chains = path_params['n_chains']
    t_eq = path_params['t_eq']
    t_total = path_params['t_total']
    
    print(f"Detected parameters from path: n_chains={n_chains}, t_eq={t_eq}, t_total={t_total}, sites={sites_per_mol}")
    
    # Set up output path
    if args.output is None:
        traj_path = Path(args.trajectory)
        output_path = traj_path.parent / f"{traj_path.stem}_rdf.png"
    else:
        output_path = Path(args.output)
    
    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}")
    traj = load_trajectory(args.trajectory, args.box_length)
    print(f"Trajectory shape: {traj.shape}")
    
    # Check if multi-chain simulation
    is_multi_chain = n_chains > 1
    
    # Split into chains if multi-chain simulation
    if is_multi_chain:
        print(f"Splitting trajectory into {n_chains} chains...")
        line_locs = get_line_locations(t_eq, t_total, n_chains, print_every=args.print_every)
        traj_chains = split_into_chains(traj, line_locs)
        print(f"Split trajectory shape: {traj_chains.shape}")
        
        # Remove exploded simulations if requested
        if args.rm_exploded:
            print(f"\nRemoving exploded simulations (threshold={args.threshold} kT)...")
            aux_data = load_aux_data(args.trajectory)
            
            if aux_data is not None:
                aux_chains = split_into_chains(aux_data, line_locs)
                print(f"Aux data shape: {aux_chains.shape}")
                
                # Mark NaN for exploded frames
                traj_chains = mark_nan(traj_chains, aux_chains, threshold=args.threshold, verbose=True)
                
                # Calculate stability
                mean_length, std_length = calculate_stability(traj_chains, print_every=args.print_every)
                print(f"Mean stable length: {mean_length:.1f} ± {std_length:.1f} ps")
                
                # Filter out chains with any NaN values
                valid_chain_mask = np.array([~np.isnan(chain).any() for chain in traj_chains])
                n_valid = np.sum(valid_chain_mask)
                n_total = len(traj_chains)
                
                print(f"Valid chains kept: {n_valid} / {n_total}")
                
                if n_valid == 0:
                    print("ERROR: No valid chains remain after NaN filtering. Cannot plot RDF.")
                    return
                
                # Keep only valid chains
                traj_chains = np.array([chain for chain, ok in zip(traj_chains, valid_chain_mask) if ok])
                print(f"Filtered trajectory shape: {traj_chains.shape}")
            else:
                print(f"Warning: traj_state_aux.pkl not found, skipping explosion detection")
        
        # Create list of trajectories (one per chain)
        trajectories = [traj_chains[i] for i in range(traj_chains.shape[0])]
    else:
        trajectories = [traj]
    
    # Get bead types for the given sites_per_mol
    bead_types = get_bead_types(sites_per_mol)
    print(f"\nUsing {sites_per_mol} sites per molecule with bead types: {bead_types}")
    
    # Calculate RDF
    print("Calculating RDF...")
    rdf_data, bead_combinations = calculate_rdf(
        trajectories,
        bead_types,
        sites_per_mol=sites_per_mol,
        box_length=args.box_length,
        dr=args.dr,
        pair_batch_size=10_000,
        frame_batch_size=1_000
    )
    
    # Set up plot style
    setup_plot_style()
    
    # Create figure with subplots for each bead combination
    n_combos = len(bead_combinations)
    fig, axes = plt.subplots(1, n_combos, figsize=(5 * n_combos, 4))
    
    if n_combos == 1:
        axes = [axes]
    
    r_max = args.box_length / 2
    
    for ax, bead_combo in zip(axes, bead_combinations):
        if bead_combo not in rdf_data:
            continue
        
        type1, type2 = bead_combo
        combo_label = f"{type1}-{type2}"
        
        if is_multi_chain and len(trajectories) > 1:
            # Compute mean and std across chains for multi-chain simulations
            all_r = []
            all_g = []
            for traj_idx in rdf_data[bead_combo]:
                r_vals, g_vals = rdf_data[bead_combo][traj_idx]
                all_r.append(r_vals)
                all_g.append(g_vals)
            
            all_g = np.array(all_g)
            r_vals = all_r[0]  # r values should be the same for all chains
            
            g_mean = np.mean(all_g, axis=0)
            g_std = np.std(all_g, axis=0)
            
            ax.plot(r_vals, g_mean, linewidth=1.5, label=f'Mean (n={len(trajectories)})')
            ax.fill_between(r_vals, g_mean - g_std, g_mean + g_std, alpha=0.3, label='±1 std')
        else:
            r_vals, g_vals = rdf_data[bead_combo][0]
            ax.plot(r_vals, g_vals, linewidth=1.5)
        
        ax.set_xlabel("r (nm)")
        ax.set_ylabel(f"g$_{{{combo_label}}}$(r)")
        ax.set_xlim(0.3, r_max)
        ax.tick_params(direction='in')
        ax.set_title(f"RDF: {combo_label}")
        
        if is_multi_chain and len(trajectories) > 1:
            ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved RDF plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
