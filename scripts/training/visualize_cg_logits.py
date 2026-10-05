import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cg_logits(data: dict, output_path: str):
    """
    Plots CG pre-softmax logit magnitudes, margins, and heatmaps.
    
    Args:
        data: Dict containing 'epoch', 'norm', 'max_abs', 'min', 'max', 'mean',
              'avg_margin', 'max_margin', and optionally 'initial_logits', 'final_logits'.
        output_path: Path to save the plot.
    """
    history = data.get("history", data)
    epochs = history.get("epoch", [])
    if not epochs:
        print("[WARNING] No epoch data found in logits history.")
        return

    has_initial = "initial_logits" in data and data["initial_logits"] is not None
    has_final = "final_logits" in data and data["final_logits"] is not None
    
    if has_initial and has_final:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Norm and Bounds over Epochs
    axes[0].plot(epochs, history["norm"], color='#1f77b4', linewidth=2, label="Frobenius Norm (Logits)")
    axes[0].plot(epochs, history["mean"], color='#2ca02c', linestyle='--', linewidth=1.5, label="Mean Logit")
    axes[0].fill_between(epochs, history["min"], history["max"], color='#1f77b4', alpha=0.15, label="[Min, Max] Range")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Logit Magnitude", fontsize=12)
    axes[0].set_title("Pre-Softmax Logits Magnitude & Range", fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10, loc='best')

    # Subplot 2: Max |Logit| and Margins vs Gumbel Noise Scale
    axes[1].plot(epochs, history["max_abs"], color='#d62728', linewidth=2, label="Max |Logit|")
    if "avg_margin" in history:
        axes[1].plot(epochs, history["avg_margin"], color='#ff7f0e', linewidth=2, label="Avg Top-1 Margin")
    if "max_margin" in history:
        axes[1].plot(epochs, history["max_margin"], color='#9467bd', linestyle=':', linewidth=1.5, label="Max Top-1 Margin")
    
    # Reference band for standard Gumbel noise scale (std dev ≈ 1.28, typical fluctuations [-2, 4])
    axes[1].axhspan(0, 3.0, color='gray', alpha=0.2, label="Gumbel Noise Regime (≤ 3.0)")
    axes[1].axhline(10.0, color='black', linestyle='--', alpha=0.7, label="Saturation Threshold (≈ 10.0)")

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Logit Value / Margin", fontsize=12)
    axes[1].set_title("Logit Separation vs. Gumbel Noise Scale", fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9, loc='best')

    # Subplot 3 & 4: Initial and Final Logits Heatmap
    if has_initial and has_final:
        init_arr = np.array(data["initial_logits"])
        final_arr = np.array(data["final_logits"])

        # Heatmap 1: Initial
        im1 = axes[2].imshow(init_arr, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[2].set_xlabel("CG Bead Index", fontsize=11)
        axes[2].set_ylabel("Atom Index", fontsize=11)
        axes[2].set_title(f"Initial Logits (Epoch {epochs[0]})", fontsize=13, fontweight='bold')
        fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

        # Heatmap 2: Final
        im2 = axes[3].imshow(final_arr, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[3].set_xlabel("CG Bead Index", fontsize=11)
        axes[3].set_ylabel("Atom Index", fontsize=11)
        axes[3].set_title(f"Final Logits (Epoch {epochs[-1]})", fontsize=13, fontweight='bold')
        fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved CG logits visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize CG mapping pre-softmax logits over time.")
    parser.add_argument("logits_json", type=str, help="Path to cg_logits.json")
    parser.add_argument("--out", type=str, default=None, help="Path to save output plot (default: cg_logits_viz.png in same folder)")
    args = parser.parse_args()

    if not os.path.exists(args.logits_json):
        print(f"Error: {args.logits_json} does not exist.")
        return

    with open(args.logits_json, 'r') as f:
        data = json.load(f)

    save_path = args.out if args.out else os.path.join(os.path.dirname(args.logits_json), "cg_logits_viz.png")
    plot_cg_logits(data, save_path)

if __name__ == "__main__":
    main()
