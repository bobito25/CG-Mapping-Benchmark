#!/usr/bin/env python3
import argparse
import os
import sys
import json

# Setup sys.path to find mapping_viz
viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mapping_viz"))
if viz_dir not in sys.path:
    sys.path.insert(0, viz_dir)

try:
    from viz import visualize_mapping_grid
    from rdkit import Chem
except ImportError as e:
    print(f"[ERROR] Could not import required packages: {e}")
    print("[ERROR] Please make sure rdkit and other dependencies are installed in your environment.")
    sys.exit(1)

MOLECULE_SMILES = {
    "ala2": "CC(=O)N[C@@H](C)C(=O)NC",
    "hexane": "CCCCCC",
    "pro2": "CC(=O)N1CCC[C@H]1C(=O)NC",
    "thr2": "CC(=O)N[C@@H]([C@H](O)C)C(=O)NC",
    "gly2": "CC(=O)NCC(=O)NC",
    "ala15": "CC(=O)" + "N[C@@H](C)C(=O)" * 15 + "NC",
}

def main():
    parser = argparse.ArgumentParser(description="Generate CG maps visualization from cg_maps.json")
    parser.add_argument(
        "--cg-maps",
        type=str,
        required=True,
        help="Path to cg_maps.json or directory containing it"
    )
    parser.add_argument(
        "--mol",
        type=str,
        help="Molecule name (e.g. ala2, hexane, pro2, thr2, gly2, ala15). If not specified, the script tries to read it from config.json in the same directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the generated PNG. Defaults to cg_maps_over_time.png in the same directory as cg_maps.json"
    )
    parser.add_argument(
        "--num-beads",
        type=int,
        help="Number of CG beads. If not specified, inferred as max_bead_index + 1."
    )
    args = parser.parse_args()

    # Determine paths
    cg_maps_path = args.cg_maps
    if os.path.isdir(cg_maps_path):
        cg_maps_path = os.path.join(cg_maps_path, "cg_maps.json")

    if not os.path.exists(cg_maps_path):
        print(f"[ERROR] File not found: {cg_maps_path}")
        sys.exit(1)

    # Read cg_maps.json
    try:
        with open(cg_maps_path, "r") as f:
            saved_cg_maps = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {cg_maps_path}: {e}")
        sys.exit(1)

    if not saved_cg_maps:
        print("[ERROR] cg_maps.json is empty.")
        sys.exit(1)

    # In JSON, keys are strings, convert to integer epochs
    try:
        saved_cg_maps = {int(k): v for k, v in saved_cg_maps.items()}
    except ValueError as e:
        print(f"[ERROR] Epoch keys in cg_maps.json must be integers: {e}")
        sys.exit(1)

    sorted_epochs = sorted(saved_cg_maps.keys())
    mappings = [saved_cg_maps[ep] for ep in sorted_epochs]

    # Infer mol_name if not specified
    mol_name = args.mol
    if not mol_name:
        config_path = os.path.join(os.path.dirname(cg_maps_path), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                mol_name = config.get("mol")
                print(f"[INFO] Inferred molecule '{mol_name}' from {config_path}")
            except Exception as e:
                print(f"[WARNING] Failed to read {config_path}: {e}")
        else:
            print(f"[WARNING] config.json not found in {os.path.dirname(cg_maps_path)}")

    if not mol_name:
        print("[ERROR] Molecule name could not be inferred. Please specify it using --mol.")
        sys.exit(1)

    # Get SMILES
    mol_name_lower = mol_name.lower()
    smiles = MOLECULE_SMILES.get(mol_name_lower)
    if not smiles:
        print(f"[ERROR] SMILES not found for molecule '{mol_name}'. Available molecules: {list(MOLECULE_SMILES.keys())}")
        sys.exit(1)

    # Construct molecule
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f"[ERROR] Failed to build molecule structure from SMILES: {e}")
        sys.exit(1)

    # Renumber if it's ala2 (matching the renumbering in viz.py/run_mace_training.py)
    if mol_name_lower == "ala2":
        permutation = [0,10,11,12,1,2,3,13,4,14,5,15,16,17,6,7,8,18,9,19,20,21]
        try:
            mol = Chem.RenumberAtoms(mol, permutation)
            print(f"[INFO] Renumbered atoms for {mol_name} using permutation.")
        except Exception as e:
            print(f"[WARNING] Failed to renumber atoms for {mol_name}: {e}")

    # Determine num_cg_beads
    if args.num_beads is not None:
        num_cg_beads = args.num_beads
    else:
        max_bead = 0
        for mapping in mappings:
            valid_indices = [idx for idx in mapping if idx >= 0]
            if valid_indices:
                max_bead = max(max_bead, max(valid_indices))
        num_cg_beads = max_bead + 1
    print(f"[INFO] Number of CG beads: {num_cg_beads}")

    # Set up arguments for visualize_mapping_grid
    legends = [f"Epoch {ep}" for ep in sorted_epochs]
    epoch_species = [list(range(num_cg_beads)) for _ in sorted_epochs]

    # Output path
    output_image_path = args.output
    if not output_image_path:
        output_image_path = os.path.join(os.path.dirname(cg_maps_path), "cg_maps_over_time.png")

    # Generate visualization
    try:
        print(f"[INFO] Generating visualization and saving to: {output_image_path}")
        visualize_mapping_grid(mol, mappings, legends, epoch_species, output_image_path)
        print("[INFO] Successfully generated CG maps visualization!")
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
