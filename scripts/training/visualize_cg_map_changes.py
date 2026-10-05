import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cg_map_changes(data: dict, output_path: str):
    """
    Plots the frequency and dynamics of hard assignment CG map changes between consecutive steps/batches.
    
    Args:
        data: Dict containing 'transitions' and optional 'change_events'
        output_path: Path to save the output plot
    """
    transitions = data.get("transitions", data)
    epochs = transitions.get("epoch", [])
    if not epochs:
        print("[WARNING] No epoch data found in CG map transitions.")
        return

    change_events = data.get("change_events", [])
    epoch_changes = transitions.get("epoch_changes", [])
    epoch_steps = transitions.get("epoch_steps", [])
    epoch_change_rate = [r * 100.0 for r in transitions.get("epoch_change_rate", [])]
    cumulative_changes = transitions.get("cumulative_changes", [])
    cumulative_flips = transitions.get("cumulative_atom_flips", [])
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Subplot 1: Map Changes per Epoch (Count and Rate %)
    ax1 = axes[0]
    color_bar = '#1f77b4'
    color_rate = '#d62728'

    ax1.bar(epochs, epoch_changes, color=color_bar, alpha=0.6, width=0.8, label="Step Changes Count")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Changed Step Transitions (Count)", color=color_bar, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, epoch_change_rate, color=color_rate, marker='o', markersize=3, linewidth=1.8, label="Change Rate (%)")
    ax1_twin.set_ylabel("Step Transitions Changed (%)", color=color_rate, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color_rate)
    ax1.set_title("CG Map Changes per Epoch", fontsize=13, fontweight='bold')

    # Subplot 2: Cumulative Step Changes & Atom Flips
    ax2 = axes[1]
    if cumulative_changes:
        ax2.plot(epochs, cumulative_changes, color='#2ca02c', linewidth=2.2, label="Cumulative Map Changes")
    if cumulative_flips:
        ax2.plot(epochs, cumulative_flips, color='#9467bd', linestyle='--', linewidth=2.0, label="Cumulative Atom Flips")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cumulative Count", fontsize=12)
    ax2.set_title("Cumulative Map Stability", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10, loc='best')

    # Subplot 3: Step-Level Flip Events / Magnitude
    ax3 = axes[2]
    if change_events:
        steps = [ev["step"] for ev in change_events]
        flips = [ev.get("changed_atoms", 1) for ev in change_events]
        scatter = ax3.scatter(steps, flips, c=flips, cmap='plasma', alpha=0.8, edgecolors='k', linewidth=0.5, s=35)
        cbar = fig.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label("Atoms Flipped", fontsize=10)
        ax3.set_xlabel("Global Step", fontsize=12)
        ax3.set_ylabel("Number of Flipped Atoms", fontsize=12)
        ax3.set_title(f"Step-to-Step Flip Events (Total: {len(change_events)})", fontsize=13, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "No CG Map Changes Detected\n(Map remained constant across all steps)",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=12, color='gray', fontstyle='italic')
        ax3.set_title("Step-to-Step Flip Events", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Global Step", fontsize=12)
        ax3.set_ylabel("Number of Flipped Atoms", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved CG map changes visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize CG map changes between steps/batches over time.")
    parser.add_argument("changes_json", type=str, help="Path to cg_map_changes.json")
    parser.add_argument("--out", type=str, default=None, help="Path to save output plot (default: cg_map_changes_viz.png in same folder)")
    args = parser.parse_args()

    if not os.path.exists(args.changes_json):
        print(f"Error: {args.changes_json} does not exist.")
        return

    with open(args.changes_json, 'r') as f:
        data = json.load(f)

    save_path = args.out if args.out else os.path.join(os.path.dirname(args.changes_json), "cg_map_changes_viz.png")
    plot_cg_map_changes(data, save_path)

if __name__ == "__main__":
    main()
