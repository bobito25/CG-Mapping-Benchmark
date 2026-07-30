#!/usr/bin/env python3
"""
Script to plot training and validation losses for a given set of output directories
in outputs/MLP_train into a single comparison plot.
"""

import argparse
import glob
import os
import sys

# Ensure repository root is on sys.path for cgbench imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cgbench.plotting.training import plot_multi_loss_comparison


def resolve_directories(input_dirs: list[str], base_dir: str = "outputs/MLP_train") -> list[str]:
    """
    Resolve given input directory paths, relative names, or glob patterns
    into absolute/valid directory paths.
    """
    resolved = []
    for entry in input_dirs:
        # Check if entry is a glob pattern or path
        matches = sorted(glob.glob(entry))
        if not matches:
            # Try combining with base_dir
            joined = os.path.join(base_dir, entry)
            matches = sorted(glob.glob(joined))
            if not matches and os.path.exists(joined):
                matches = [joined]

        if not matches:
            # Check if entry exists directly
            if os.path.exists(entry):
                matches = [entry]

        if not matches:
            print(f"[WARNING] Could not find matching directory for: '{entry}'")
            continue

        for m in matches:
            if os.path.isdir(m) and m not in resolved:
                resolved.append(m)

    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation losses for output directories in outputs/MLP_train into one plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="Output directory names, paths, or glob patterns (e.g. 'Ala2_map=AT*' or 'outputs/MLP_train/Ala2_map=core*')",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="outputs/MLP_train",
        help="Base directory containing training outputs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outputs/MLP_train/loss_comparison.png",
        help="Output image path for saved plot",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "train", "val"],
        default="both",
        help="Which losses to display: 'both', 'train', or 'val'",
    )
    parser.add_argument(
        "--split_subplots",
        "--split",
        "--side_by_side",
        "--side-by-side",
        dest="split_subplots",
        action="store_true",
        help="Plot Train and Validation losses in separate side-by-side subplots",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Custom labels for legend corresponding to each resolved directory",
    )
    parser.add_argument(
        "--linear_scale",
        action="store_true",
        help="Use linear scale for y-axis instead of logarithmic (semilogy)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Loss Comparison",
        help="Custom title for the figure",
    )

    args = parser.parse_args()

    resolved_dirs = resolve_directories(args.dirs, base_dir=args.base_dir)
    if not resolved_dirs:
        print("[ERROR] No valid output directories resolved. Exiting.")
        sys.exit(1)

    print(f"[INFO] Plotting loss comparison for {len(resolved_dirs)} directory(ies):")
    for d in resolved_dirs:
        print(f"  - {d}")

    plot_multi_loss_comparison(
        dir_paths=resolved_dirs,
        labels=args.labels,
        out_path=args.output,
        mode=args.mode,
        split_subplots=args.split_subplots,
        log_scale=not args.linear_scale,
        title=args.title,
    )


if __name__ == "__main__":
    main()
