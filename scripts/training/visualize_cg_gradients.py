import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize CG mapping gradients over time.")
    parser.add_argument("grads_json", type=str, help="Path to cg_gradients.json")
    parser.add_argument("--out", type=str, default=None, help="Path to save output plot (default: same folder as json)")
    args = parser.parse_args()

    if not os.path.exists(args.grads_json):
        print(f"Error: {args.grads_json} does not exist.")
        return

    with open(args.grads_json, 'r') as f:
        grads = json.load(f)

    if not grads:
        print("Warning: Gradients file is empty.")
        return

    # Sort epochs
    epochs = sorted([int(k) for k in grads.keys()])
    
    norms_kernel = []
    norms_bias = []
    
    first_epoch = epochs[0]
    final_epoch = epochs[-1]
    
    first_kernel_grad = np.array(grads[str(first_epoch)]['kernel'])
    final_kernel_grad = np.array(grads[str(final_epoch)]['kernel'])

    for ep in epochs:
        k_grad = np.array(grads[str(ep)]['kernel'])
        b_grad = np.array(grads[str(ep)]['bias'])
        
        # Compute Frobenius norm
        norms_kernel.append(np.linalg.norm(k_grad))
        norms_bias.append(np.linalg.norm(b_grad))

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Gradient Norms over Epochs
    axes[0].plot(epochs, norms_kernel, marker='o', label='Kernel Grad Norm (Frobenius)', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, norms_bias, marker='s', label='Bias Grad Norm (Frobenius)', color='#ff7f0e', linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Frobenius Norm", fontsize=12)
    axes[0].set_title("Mapping Parameter Gradient Norms", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # Subplot 2: Initial Kernel Gradient Heatmap
    max_val1 = max(float(np.max(np.abs(first_kernel_grad))), 1e-8)
    im1 = axes[1].imshow(first_kernel_grad, aspect='auto', cmap='coolwarm', interpolation='nearest',
                         vmin=-max_val1, vmax=max_val1)
    axes[1].set_xlabel("CG Bead Index", fontsize=12)
    axes[1].set_ylabel("Atom Index", fontsize=12)
    axes[1].set_title(f"Kernel Gradients at Epoch {first_epoch}", fontsize=14, fontweight='bold')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Subplot 3: Final Kernel Gradient Heatmap
    max_val2 = max(float(np.max(np.abs(final_kernel_grad))), 1e-8)
    im2 = axes[2].imshow(final_kernel_grad, aspect='auto', cmap='coolwarm', interpolation='nearest',
                         vmin=-max_val2, vmax=max_val2)
    axes[2].set_xlabel("CG Bead Index", fontsize=12)
    axes[2].set_ylabel("Atom Index", fontsize=12)
    axes[2].set_title(f"Kernel Gradients at Epoch {final_epoch}", fontsize=14, fontweight='bold')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Determine save path
    if args.out:
        save_path = args.out
    else:
        save_path = os.path.join(os.path.dirname(args.grads_json), "cg_gradients_viz.png")
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved gradient visualization to {save_path}")

    # Create the grid plot of all epochs
    num_epochs = len(epochs)
    cols = min(5, num_epochs)
    import math
    rows = math.ceil(num_epochs / cols)
    
    fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows), squeeze=False)
    
    for idx, ep in enumerate(epochs):
        r = idx // cols
        c = idx % cols
        ax = axes_grid[r, c]
        
        kernel_grad = np.array(grads[str(ep)]['kernel'])
        max_val = max(float(np.max(np.abs(kernel_grad))), 1e-8)
        
        im = ax.imshow(kernel_grad, aspect='auto', cmap='coolwarm', interpolation='nearest',
                       vmin=-max_val, vmax=max_val)
        ax.set_title(f"Epoch {ep}", fontsize=11, fontweight='bold')
        ax.set_xlabel("CG Bead", fontsize=8)
        ax.set_ylabel("Atom", fontsize=8)
        fig_grid.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Hide any unused subplots
    for idx in range(num_epochs, rows * cols):
        r = idx // cols
        c = idx % cols
        axes_grid[r, c].axis('off')
        
    plt.tight_layout()
    
    if args.out:
        grid_save_path = args.out.replace("cg_gradients_viz.png", "cg_gradients_grid.png")
        if grid_save_path == args.out:
            base, ext = os.path.splitext(args.out)
            grid_save_path = f"{base}_grid{ext}"
    else:
        grid_save_path = os.path.join(os.path.dirname(args.grads_json), "cg_gradients_grid.png")
        
    fig_grid.savefig(grid_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig_grid)
    print(f"[INFO] Saved all epochs gradient grid to {grid_save_path}")

if __name__ == "__main__":
    main()
