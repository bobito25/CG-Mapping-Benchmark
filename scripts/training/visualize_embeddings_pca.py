#!/usr/bin/env python
"""
Standalone CLI tool to visualize learned atom embeddings as vectors and overlay combined CG bead embeddings in PCA space.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cgbench.plotting.training import plot_atom_and_cg_embeddings_pca


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned atom embeddings as vectors and combined CG bead embeddings via PCA."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to training output directory or atom_embeddings.json file.",
    )
    parser.add_argument(
        "--cg_maps",
        type=str,
        default=None,
        help="Optional path to cg_maps.json (default: looked up in the same folder).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to visualize (default: final epoch).",
    )
    parser.add_argument(
        "--mol",
        type=str,
        default=None,
        help="Molecule name (e.g. 'ala2', 'hexane'). If not specified, inferred from config.json or defaults to 'ala2'.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory to save plots (default: same directory as atom_embeddings.json).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="atom_and_cg_embeddings_pca",
        help="Prefix for saved plots (default: 'atom_and_cg_embeddings_pca').",
    )

    args = parser.parse_args()

    # Determine paths
    if os.path.isdir(args.path):
        out_dir = args.path
        atom_emb_path = os.path.join(args.path, "atom_embeddings.json")
        cg_maps_path = args.cg_maps or os.path.join(args.path, "cg_maps.json")
    else:
        atom_emb_path = args.path
        out_dir = os.path.dirname(args.path) or "."
        cg_maps_path = args.cg_maps or os.path.join(out_dir, "cg_maps.json")

    if args.out:
        out_dir = args.out

    if not os.path.exists(atom_emb_path):
        print(f"[ERROR] atom_embeddings.json not found at {atom_emb_path}")
        sys.exit(1)

    if not os.path.exists(cg_maps_path):
        print(f"[ERROR] cg_maps.json not found at {cg_maps_path}")
        sys.exit(1)

    # Incur molecule name from config.json if available
    mol_name = args.mol
    if mol_name is None:
        config_path = os.path.join(out_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                mol_name = cfg.get("mol", "ala2")
            except Exception:
                mol_name = "ala2"
        else:
            mol_name = "ala2"

    print(f"[INFO] Loading atom embeddings from {atom_emb_path}")
    with open(atom_emb_path, "r") as f:
        atom_embeddings = json.load(f)

    print(f"[INFO] Loading CG maps from {cg_maps_path}")
    with open(cg_maps_path, "r") as f:
        cg_maps = json.load(f)

    print(f"[INFO] Generating PCA vector visualizations for {mol_name.capitalize()} (Epoch: {args.epoch or 'final'})...")
    plot_atom_and_cg_embeddings_pca(
        saved_atom_embeddings=atom_embeddings,
        saved_cg_maps=cg_maps,
        out_dir=out_dir,
        mol_name=mol_name,
        epoch=args.epoch,
        save_prefix=args.prefix,
    )
    print(f"[INFO] Done! Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
