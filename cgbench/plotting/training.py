"""
Training diagnostics plotting (predictions, convergence, distances).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler


def plot_predictions(
    predictions: dict, reference_data: dict, out_dir: str, name: str
) -> None:
    """
    Plot force predictions vs reference data with scatter plot and compute MAE.

    Parameters
    ----------
    predictions : dict
        Dictionary containing predicted values with 'F' key for forces
    reference_data : dict
        Dictionary containing reference values with 'F' key for forces
    out_dir : str
        Output directory to save the figure
    name : str
        Name for the output file
    """
    # Simplifies comparison: convert units
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), layout="constrained")
    fig.suptitle("Predictions")

    # Reshape forces and scale units
    pred_F = predictions["F"].reshape(-1, 3) / scale_energy * scale_pos
    ref_F = reference_data["F"].reshape(-1, 3) / scale_energy * scale_pos

    # Ensure pred_F has same number of entries as ref_F by dropping extra entries
    if len(pred_F) > len(ref_F):
        pred_F = pred_F[: len(ref_F)]
    elif len(ref_F) > len(pred_F):
        ref_F = ref_F[: len(pred_F)]

    # Verify shapes match
    assert (
        pred_F.shape == ref_F.shape
    ), f"Shape mismatch: pred_F {pred_F.shape}, ref_F {ref_F.shape}"

    # Compute MAE
    mae = np.mean(np.abs(pred_F - ref_F))
    ax.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")

    # 45-degree reference line
    ax.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 5, 1, 5)), linewidth=1)

    # Scatter plot
    ax.set_prop_cycle(cycler(color=plt.get_cmap("tab20c").colors))
    ax.scatter(ref_F.ravel(), pred_F.ravel(), s=5, edgecolors="none", alpha=0.2)

    ax.set_xlabel("Ref. F [eV/A]")
    ax.set_ylabel("Pred. F [eV/A]")
    ax.legend().remove()  # no legend needed

    # Save figure
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight", dpi=1200)


def plot_convergence(trainer, out_dir: str) -> None:
    """
    Plot training and validation loss convergence.

    Parameters
    ----------
    trainer : object
        Trainer object with train_losses and val_losses attributes
    out_dir : str
        Output directory to save the figure
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(f"{out_dir}/convergence.pdf", bbox_inches="tight")


def plot_atom_distance(
    ax: Axes,
    distances: np.ndarray | list[np.ndarray],
    labels: list[str] | None = None,
    bins: int = 60,
    xlabel: str = "Distance",
    ylabel: str = "Frequency",
) -> Axes:
    """
    Plot histogram of atom distances.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    distances : np.ndarray | list[np.ndarray]
        Distance data - single array or list of arrays for multiple models
    labels : list[str] | None, optional
        List of labels for each set of distances
    bins : int, optional
        Number of bins for the histogram
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ["#368274", "#0C7CBA", "#C92D39", "k"]
    line = ["-", "-", "-", "--"]

    if isinstance(distances, (list, tuple)) and hasattr(distances[0], "__len__"):
        n_models = len(distances)
        for i in range(n_models):
            ax.hist(
                distances[i],
                bins=bins,
                alpha=0.6,
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                histtype="step",
                linewidth=2.0,
                linestyle=line[i % len(line)],
            )
    else:
        ax.hist(
            distances,
            bins=bins,
            alpha=0.6,
            color=color[0],
            histtype="step",
            linewidth=2.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def compare_atom_distances(
    AT_distances: list[np.ndarray],
    Traj_distances: list[np.ndarray],
    dist_labels: list[str],
    outpath: str,
    name: str,
    at_label: str = "Reference",
    traj_label: str = "Simulation",
    bins: int = 60,
    at_color: str = "#368274",
    traj_color: str = "#C92D39",
    xlabel: str = "Distance",
    ylabel: str = "Normalized frequency",
) -> str:
    """
    Plot reference vs simulation atom-distance histograms side by side.

    Parameters
    ----------
    AT_distances : list[np.ndarray]
        List of 1D arrays of reference distances
    Traj_distances : list[np.ndarray]
        List of 1D arrays of simulation distances
    dist_labels : list[str]
        List of titles for each subplot
    outpath : str
        Directory to save the figure in
    name : str
        Basename for the output file
    at_label : str, optional
        Legend label for reference data
    traj_label : str, optional
        Legend label for simulation data
    bins : int, optional
        Number of bins
    at_color : str, optional
        Color for reference histograms
    traj_color : str, optional
        Color for simulation histograms
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label

    Returns
    -------
    str
        Full path to the saved figure file
    """
    n = len(dist_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)

    for i, title in enumerate(dist_labels):
        ax = axes[i] if n > 1 else axes
        # AT
        ax.hist(
            AT_distances[i],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            linestyle="-",
            color=at_color,
            label=at_label,
        )
        # Simulation
        ax.hist(
            Traj_distances[i],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            linestyle="-",
            color=traj_color,
            label=traj_label,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    plt.tight_layout()
    fname = f"{outpath}/Atom_distances_{name}_vs_Reference.png"
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    return fname
