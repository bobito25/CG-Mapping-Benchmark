import MDAnalysis as mda
import numpy as np
import os
import argparse
import sys

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
# Default mapping of atom names/types to atomic numbers
ATOM_MAP = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 
    'CL': 17, 'NA': 11, 'K': 19, 'CA': 20, 'MG': 12
}

# Unit Conversion Factors
ANGSTROM_TO_NM = 0.1
FORCE_CONVERSION = 10.0  # kJ/mol/A to kJ/mol/nm

def get_atomic_number(symbol):
    """Returns atomic number based on atom name or type."""
    # Clean symbol: remove numbers and uppercase
    clean_sym = ''.join([i for i in symbol if not i.isdigit()]).upper()
    
    # Try exact match (e.g., "CA" for Alpha Carbon or Calcium)
    if symbol.upper() in ATOM_MAP:
        return ATOM_MAP[symbol.upper()]
    
    # Try first letter match (e.g., "N" for Nitrogen)
    first_letter = clean_sym[0]
    if first_letter in ATOM_MAP:
        return ATOM_MAP[first_letter]
    
    raise ValueError(f"Unknown atomic symbol: {symbol}")

def parse_arguments():
    """Handles command line arguments."""
    parser = argparse.ArgumentParser(description="Convert MD trajectories (GROMACS) to .npz dataset for ML.")
    
    # File Paths
    parser.add_argument("-s", "--topology", required=True, help="Topology file (e.g., .gro, .pdb)")
    parser.add_argument("-f", "--trajectory", required=True, help="Coordinate trajectory (e.g., .xtc, .trr)")
    parser.add_argument("-force", "--forces", help="Force trajectory (usually .trr). If not provided, assumes -f contains forces.")
    parser.add_argument("-o", "--output", default="dataset.npz", help="Output filename (default: dataset.npz)")
    
    # Selection and Box
    parser.add_argument("-sel", "--selection", default="not (resname SOL NA CL K CA)", 
                        help="MDAnalysis selection string (default: protein/ligand, no solvent/ions)")
    parser.add_argument("-box", "--box_size", type=float, default=8.27717, 
                        help="Box size in nm (assumes cubic box if single value provided)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Initialize Universes
    try:
        print(f"--- Loading System ---")
        u_coords = mda.Universe(args.topology, args.trajectory)
        
        # If separate force file is provided, use it; otherwise use the main trajectory
        force_file = args.forces if args.forces else args.trajectory
        u_forces = mda.Universe(args.topology, force_file)

        # 2. Apply Selections
        prot_coords = u_coords.select_atoms(args.selection)
        prot_forces = u_forces.select_atoms(args.selection)

        n_frames = min(len(u_coords.trajectory), len(u_forces.trajectory))
        n_atoms = len(prot_coords)

        if len(prot_coords) != len(prot_forces):
            print(f"Error: Atom count mismatch! Coords: {len(prot_coords)}, Forces: {len(prot_forces)}")
            sys.exit(1)

        print(f"Frames: {n_frames}")
        print(f"Atoms selected: {n_atoms}")

        # 3. Extract Data
        species = np.array([get_atomic_number(a.name) for a in prot_coords])
        coords_arr = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        forces_arr = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

        print(f"\n--- Processing Frames ---")
        for i in range(n_frames):
            u_coords.trajectory[i]
            u_forces.trajectory[i]
            
            coords_arr[i] = prot_coords.positions * ANGSTROM_TO_NM
            
            if hasattr(u_forces.trajectory.ts, 'forces') and u_forces.trajectory.ts.forces is not None:
                forces_arr[i] = prot_forces.forces * FORCE_CONVERSION
            else:
                if i == 0: print("Warning: No force data found in trajectory. Filling with zeros.")
                forces_arr[i] = np.zeros_like(prot_coords.positions)

            if (i+1) % 100 == 0 or i == n_frames-1:
                print(f" Progress: {i+1}/{n_frames}", end='\r')

        # 4. Package Dataset
        dataset = {
            'R': coords_arr,
            'F': forces_arr,
            'species': np.repeat(species.reshape(1, -1), n_frames, axis=0),
            'box': np.repeat((np.identity(3) * args.box_size).reshape(1, 3, 3), n_frames, axis=0),
            'mask': np.ones((n_frames, n_atoms), dtype=bool)
        }

        # 5. Save
        np.savez_compressed(args.output, **dataset)
        print(f"\n\nSuccess! Dataset saved as: {args.output}")
        print(f"Final shape: {coords_arr.shape}")

    except Exception as e:
        print(f"\nCritical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    