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
    from viz import visualize_mapping, visualize_mapping_grid, get_molecule_with_node_ordering
except ImportError as e:
    print(f"[ERROR] Could not import required viz package: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate CG maps visualization from cg_maps.json or mapping array")
    parser.add_argument(
        "--cg-maps",
        type=str,
        required=True,
        help="Path to cg_maps.json or directory containing it"
    )
    parser.add_argument(
        "--mol",
        type=str,
        help="Molecule name (e.g. ala2, hexane, pro2, thr2, gly2, ala15). If not specified, tries to read it from config.json in the same directory."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the generated PNG."
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

    # Convert epoch keys to integers if possible, else keep string keys
    try:
        saved_cg_maps = {int(k): v for k, v in saved_cg_maps.items()}
        sorted_epochs = sorted(saved_cg_maps.keys())
    except ValueError:
        sorted_epochs = list(saved_cg_maps.keys())

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

    if not mol_name:
        print("[ERROR] Molecule name could not be inferred. Please specify it using --mol.")
        sys.exit(1)

    # Get molecule structure with canonical node ordering
    try:
        mol = get_molecule_with_node_ordering(mol_name)
    except Exception as e:
        print(f"[ERROR] Failed to prepare molecule '{mol_name}': {e}")
        sys.exit(1)

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

    # Resolve species list accurately across static, custom, and learned maps
    species_list = None

    # 1. Read config.json if available
    config = {}
    config_path = os.path.join(os.path.dirname(cg_maps_path), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to read {config_path}: {e}")

    # 2. Check if cg_species is stored directly in config.json
    if "cg_species" in config and isinstance(config["cg_species"], (list, tuple)):
        species_list = list(config["cg_species"])
        print(f"[INFO] Using cg_species from config.json: {species_list}")
    elif config.get("unique_cg_species", False):
        species_list = list(range(1, num_cg_beads + 1))
        print(f"[INFO] Using unique_cg_species 1..{num_cg_beads}")

    # 3. Otherwise try to infer species from cgbench mapping definitions
    if species_list is None:
        try:
            scripts_dir = os.path.abspath(os.path.dirname(__file__))
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            root_dir = os.path.abspath(os.path.join(scripts_dir, "../.."))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)

            from cgbench.core import mapping as cg_mapping
            from cgbench.core.mapping import register_custom_map

            mol_name_map = {
                "ala2": "Ala2_Map",
                "ala15": "Ala15_Map",
                "hexane": "Hexane_Map",
                "pro2": "Pro2_Map",
                "thr2": "Thr2_Map",
                "gly2": "Gly2_Map",
            }
            if mol_name.lower() in mol_name_map:
                map_cls_name = mol_name_map[mol_name.lower()]
                map_cls = getattr(cg_mapping, map_cls_name, None)
                if map_cls is not None:
                    map_inst = map_cls()
                    cg_map_name = config.get("CG_map", "hmerged")
                    if cg_map_name == "learned":
                        cg_map_name = "hmerged"

                    # If custom CG map, register custom map to get inherited species
                    if (cg_map_name == "custom" or len(mappings) == 1) and mappings:
                        register_custom_map(map_inst, mappings[0])
                        cg_map_name = "custom"

                    if cg_map_name in map_inst.get_available_maps():
                        _, inferred_species, _, _ = map_inst.get_map(cg_map_name)
                        species_list = list(inferred_species)
                        print(f"[INFO] Inferred species for map '{cg_map_name}': {species_list}")
        except Exception as e:
            print(f"[WARNING] Could not infer species from map definition: {e}")

    # 4. Fallback to 1..num_cg_beads if all else fails
    if species_list is None:
        species_list = list(range(1, num_cg_beads + 1))
        print(f"[INFO] Falling back to default species 1..{num_cg_beads}")

    epoch_species = [species_list for _ in sorted_epochs]

    # Output path
    output_image_path = args.output
    if not output_image_path:
        out_dir = os.path.dirname(cg_maps_path)
        if len(mappings) > 1:
            output_image_path = os.path.join(out_dir, "cg_maps_over_time.png")
        else:
            output_image_path = os.path.join(out_dir, "cg_map.png")

    # Generate visualization
    try:
        print(f"[INFO] Generating visualization and saving to: {output_image_path}")
        if len(mappings) == 1:
            visualize_mapping(mol, mappings[0], output_image_path, species=epoch_species[0], legend=f"CG Map ({mol_name})")
        else:
            legends = [f"Epoch {ep}" for ep in sorted_epochs]
            visualize_mapping_grid(mol, mappings, legends, epoch_species, output_image_path)
        print("[INFO] Successfully generated CG map visualization!")
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
