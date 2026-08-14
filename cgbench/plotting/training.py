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


def plot_temperature_schedule(
    gumbel_temp_choice: str | float,
    epochs: int,
    out_dir: str,
    t_min: float = 0.1,
    t_max: float = 1.0,
    decay_rate: float = None,
    threephase_points: list = None,
    threephase_timings: list = None,
) -> None:
    """
    Plot the Gumbel-Softmax temperature schedule and save to out_dir.

    Parameters
    ----------
    gumbel_temp_choice : str or float
        Gumbel temperature configuration: a number for constant temp or a string
        for schedule ('exponential', 'linear', or '3phase')
    epochs : int
        Number of epochs to train
    out_dir : str
        Output directory to save the figure
    t_min : float, optional
        Minimum temperature for schedule (default: 0.1)
    t_max : float, optional
        Maximum/starting temperature for schedule (default: 1.0)
    decay_rate : float, optional
        Optional explicit exponential decay rate (default: None, solves automatically)
    threephase_points : list of float, optional
        4 temperature points for 3phase schedule (default: [1.0, 0.4, 0.3, 0.1])
    threephase_timings : list of float, optional
        2 timing fractions for middle points in 3phase schedule (default: [0.10, 0.90])
    """
    epochs_arr = np.arange(epochs)
    temperatures = []

    try:
        gumbel_temp_val = float(gumbel_temp_choice)
        temperatures = [gumbel_temp_val] * epochs
    except (ValueError, TypeError):
        t_start = t_max
        gumbel_temp_choice_str = str(gumbel_temp_choice)
        if gumbel_temp_choice_str == "exponential":
            r_decay = decay_rate
            if r_decay is None:
                r_decay = (t_min / t_start) ** (1.0 / (epochs - 1)) if epochs > 1 else 1.0
            for epoch in epochs_arr:
                temperatures.append(max(t_min, t_start * (r_decay ** epoch)))
        elif gumbel_temp_choice_str == "linear":
            for epoch in epochs_arr:
                temperatures.append(max(t_min, t_start - (t_start - t_min) * (epoch / (epochs - 1)) if epochs > 1 else 0.0))
        elif gumbel_temp_choice_str == "3phase":
            t0, t1, t2, t3 = threephase_points if threephase_points is not None else [1.0, 0.4, 0.3, 0.1]
            f1, f2 = threephase_timings if threephase_timings is not None else [0.10, 0.90]
            if epochs <= 1:
                temperatures = [t0] * epochs
            else:
                e1 = f1 * (epochs - 1)
                e2 = f2 * (epochs - 1)
                e3 = epochs - 1
                for epoch in epochs_arr:
                    if epoch <= e1:
                        val = t0 + (t1 - t0) * (epoch / e1) if e1 > 0 else t1
                    elif epoch <= e2:
                        val = t1 + (t2 - t1) * ((epoch - e1) / (e2 - e1)) if e2 > e1 else t2
                    else:
                        val = t2 + (t3 - t2) * ((epoch - e2) / (e3 - e2)) if e3 > e2 else t3
                    temperatures.append(val)
        else:
            raise ValueError(f"Unknown Gumbel temperature schedule: {gumbel_temp_choice}")

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), layout="constrained")
    ax.plot(epochs_arr, temperatures, marker="o", markersize=4, color="#C92D39", linewidth=2.0)
    ax.set_title("Gumbel-Softmax Temperature Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Temperature")
    ax.grid(True, linestyle="--", alpha=0.6)

    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/gumbel_temperature_schedule.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def plot_atom_embeddings_grid(saved_atom_embeddings: dict, out_dir: str) -> None:
    """
    Plot a visualization grid of learned atom type embeddings over time/epochs.

    Parameters
    ----------
    saved_atom_embeddings : dict
        Dictionary mapping epoch (int) to dict of {atom_type_species (int or str): embedding_vector (list or ndarray)}
    out_dir : str
        Output directory to save the figure
    """
    if not saved_atom_embeddings:
        return

    epochs = sorted([int(e) for e in saved_atom_embeddings.keys()])
    num_epochs = len(epochs)
    if num_epochs == 0:
        return

    first_epoch_data = saved_atom_embeddings[epochs[0]]
    species_keys = sorted([int(k) for k in first_epoch_data.keys()])
    num_atom_types = len(species_keys)

    species_labels = [
        ATOMIC_NUMBER_TO_SYMBOL.get(s, f"Species_{s}") for s in species_keys
    ]

    emb_dim = len(first_epoch_data[species_keys[0]])
    data_matrix = np.zeros((num_epochs, num_atom_types, emb_dim))

    for i, ep in enumerate(epochs):
        ep_dict = saved_atom_embeddings[ep]
        for j, spec in enumerate(species_keys):
            vec = ep_dict.get(spec) if spec in ep_dict else ep_dict.get(str(spec))
            if vec is not None:
                data_matrix[i, j, :] = np.array(vec)

    cols = min(4, num_epochs)
    rows = int(np.ceil(num_epochs / cols))
    grid_cols = max(3, cols)

    fig = plt.figure(figsize=(4 * grid_cols, 3.5 * (rows + 1)), layout="constrained")
    fig.suptitle("Learned Atom-Type Embeddings Over Training Epochs", fontsize=16, fontweight="bold")

    gs = fig.add_gridspec(rows + 1, grid_cols)

    vmin, vmax = data_matrix.min(), data_matrix.max()
    im = None

    for i, ep in enumerate(epochs):
        r, c = i // cols, i % cols
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(
            data_matrix[i],
            aspect="auto",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Epoch {ep}", fontsize=11)
        ax.set_yticks(np.arange(num_atom_types))
        ax.set_yticklabels(species_labels)
        ax.set_xlabel("Embedding Dim")
        if c == 0:
            ax.set_ylabel("Atom Type")

    if im is not None:
        cbar = fig.colorbar(im, ax=fig.axes[:num_epochs], shrink=0.8, location="right")
        cbar.set_label("Embedding Value")

    flat_matrix = data_matrix.reshape(-1, emb_dim)
    ax_pca = fig.add_subplot(gs[rows, 0])
    ax_norm = fig.add_subplot(gs[rows, 1])
    ax_cossim = fig.add_subplot(gs[rows, 2])

    colors = plt.cm.Set1(np.linspace(0, 1, max(1, num_atom_types)))

    if emb_dim >= 2:
        try:
            mean_vec = flat_matrix.mean(axis=0)
            centered = flat_matrix - mean_vec
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vh[:2].T
            proj = proj.reshape(num_epochs, num_atom_types, 2)

            for j in range(num_atom_types):
                traj = proj[:, j, :]
                ax_pca.plot(traj[:, 0], traj[:, 1], "o-", color=colors[j], label=species_labels[j], alpha=0.8)
                ax_pca.scatter(traj[0, 0], traj[0, 1], color=colors[j], s=80, marker="s", edgecolors="black")
                ax_pca.scatter(traj[-1, 0], traj[-1, 1], color=colors[j], s=120, marker="*", edgecolors="black")

            ax_pca.set_title("Atom Embedding Trajectory (2D PCA)")
            ax_pca.set_xlabel("PC 1")
            ax_pca.set_ylabel("PC 2")
            ax_pca.legend()
        except Exception:
            ax_pca.text(0.5, 0.5, "PCA plot unavailable", ha="center", va="center")
    else:
        ax_pca.text(0.5, 0.5, "Embedding dim < 2", ha="center", va="center")

    norms = np.linalg.norm(data_matrix, axis=-1)
    for j in range(num_atom_types):
        ax_norm.plot(epochs, norms[:, j], "o-", color=colors[j], label=species_labels[j])

    ax_norm.set_title("Embedding L2-Norm Evolution")
    ax_norm.set_xlabel("Epoch")
    ax_norm.set_ylabel("||Embedding||_2")
    ax_norm.legend()

    # Compute Cosine Similarity relative to Epoch 0
    norm_expanded = np.maximum(norms[:, :, None], 1e-8)
    unit_matrix = data_matrix / norm_expanded  # shape: (num_epochs, num_atom_types, emb_dim)
    cos_sim_ep0 = np.sum(unit_matrix * unit_matrix[0:1, :, :], axis=-1)  # shape: (num_epochs, num_atom_types)

    for j in range(num_atom_types):
        ax_cossim.plot(epochs, cos_sim_ep0[:, j], "o-", color=colors[j], label=species_labels[j])

    ax_cossim.set_title("Cosine Sim to Epoch 0 (Direction Shift)")
    ax_cossim.set_xlabel("Epoch")
    ax_cossim.set_ylabel("CosSim(v_t, v_0)")
    ax_cossim.set_ylim(-1.05, 1.05)
    ax_cossim.grid(True, linestyle="--", alpha=0.5)
    ax_cossim.legend()

    import os
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/atom_embeddings_grid.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved atom embeddings grid visualization to {out_file}")

    # Generate normalized unit-vector heatmaps grid (Directional heatmaps)
    try:
        fig_norm = plt.figure(figsize=(4 * cols + 2, 3.5 * rows), layout="constrained")
        fig_norm.suptitle("Normalized Atom Embeddings (Directional Unit Vectors)", fontsize=16, fontweight="bold")
        gs_norm = fig_norm.add_gridspec(rows, cols)
        im_norm = None

        for i, ep in enumerate(epochs):
            r, c = i // cols, i % cols
            ax_n = fig_norm.add_subplot(gs_norm[r, c])
            im_norm = ax_n.imshow(
                unit_matrix[i],
                aspect="auto",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
            )
            ax_n.set_title(f"Epoch {ep}", fontsize=11)
            ax_n.set_yticks(np.arange(num_atom_types))
            ax_n.set_yticklabels(species_labels)
            ax_n.set_xlabel("Embedding Dim")
            if c == 0:
                ax_n.set_ylabel("Atom Type")

        if im_norm is not None:
            cbar_n = fig_norm.colorbar(im_norm, ax=fig_norm.axes[:num_epochs], shrink=0.8, location="right")
            cbar_n.set_label("Normalized Feature Value")

        out_norm_file = f"{out_dir}/atom_embeddings_normalized_grid.png"
        fig_norm.savefig(out_norm_file, bbox_inches="tight", dpi=300)
        plt.close(fig_norm)
        print(f"[INFO] Saved normalized atom embeddings grid visualization to {out_norm_file}")
    except Exception as e:
        print(f"[WARNING] Could not save normalized atom embeddings grid: {e}")


def load_losses_from_dir(dir_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load train and val losses from a directory by checking trainer.pkl first,
    falling back to parsing force_matching.log if necessary.

    Parameters
    ----------
    dir_path : str
        Path to the output directory.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (train_losses, val_losses) as numpy arrays or None if not found.
    """
    import os
    import pickle
    import re

    train_losses = None
    val_losses = None

    pkl_path = os.path.join(dir_path, "trainer.pkl")
    log_path = os.path.join(dir_path, "force_matching.log")

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            tr_matches = re.findall(r"Average train loss:\s*([0-9\.eE+-]+)", content)
            vl_matches = re.findall(r"Average val loss:\s*([0-9\.eE+-]+)", content)
            if tr_matches:
                train_losses = [float(x) for x in tr_matches]
            if vl_matches:
                val_losses = [float(x) for x in vl_matches]
        except Exception:
            pass

    if train_losses is None and os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    train_losses = data.get("train_losses")
                    val_losses = data.get("val_losses")
                elif hasattr(data, "train_losses"):
                    train_losses = data.train_losses
                    val_losses = getattr(data, "val_losses", None)
        except Exception:
            pass

    if train_losses is not None:
        train_losses = np.asarray(train_losses, dtype=float)
    if val_losses is not None:
        val_losses = np.asarray(val_losses, dtype=float)

    return train_losses, val_losses


def plot_multi_loss_comparison(
    dir_paths: list[str],
    labels: list[str] | None = None,
    out_path: str = "loss_comparison.png",
    mode: str = "both",
    split_subplots: bool = False,
    log_scale: bool = True,
    title: str = "Loss Comparison",
) -> str:
    """
    Plot losses from multiple output directories into a single plot.

    Parameters
    ----------
    dir_paths : list[str]
        List of directory paths to plot.
    labels : list[str] | None, optional
        Custom labels for each directory. If None, derived from folder names.
    out_path : str, optional
        Output file path to save figure.
    mode : str, optional
        'both', 'train', or 'val'
    split_subplots : bool, optional
        If True and mode is 'both', plot train and val in two side-by-side subplots.
    log_scale : bool, optional
        If True, use semilogy (log scale for y-axis).
    title : str, optional
        Title for the figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    import os

    if not dir_paths:
        raise ValueError("No directory paths provided for loss comparison plot.")

    # Prepare labels
    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dir_paths]
    elif len(labels) != len(dir_paths):
        raise ValueError(f"Mismatch: {len(dir_paths)} directories provided, but {len(labels)} labels.")

    # Color palette cycle
    color_cycle = plt.cm.tab10.colors
    if len(dir_paths) > 10:
        color_cycle = plt.cm.tab20.colors

    # Load data for each directory
    data_list = []
    for d, lbl in zip(dir_paths, labels):
        tr, vl = load_losses_from_dir(d)
        if tr is None and vl is None:
            print(f"[WARNING] No loss data found in {d}")
        else:
            data_list.append((d, lbl, tr, vl))

    if not data_list:
        raise RuntimeError("No valid loss data could be loaded from any specified directory.")

    if mode == "both" and split_subplots:
        fig, (ax_tr, ax_vl) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plot_ax_tr = ax_tr.semilogy if log_scale else ax_tr.plot
        plot_ax_vl = ax_vl.semilogy if log_scale else ax_vl.plot

        for i, (d, lbl, tr, vl) in enumerate(data_list):
            color = color_cycle[i % len(color_cycle)]
            if tr is not None:
                epochs = np.arange(len(tr))
                plot_ax_tr(epochs, tr, label=lbl, color=color, linewidth=2.0)
            if vl is not None:
                epochs = np.arange(len(vl))
                plot_ax_vl(epochs, vl, label=lbl, color=color, linewidth=2.0)

        ax_tr.set_title("Training Loss")
        ax_tr.set_xlabel("Epoch")
        ax_tr.set_ylabel("Loss")
        ax_tr.grid(True, linestyle="--", alpha=0.5)
        ax_tr.legend(fontsize=9)

        ax_vl.set_title("Validation Loss")
        ax_vl.set_xlabel("Epoch")
        ax_vl.set_ylabel("Loss")
        ax_vl.grid(True, linestyle="--", alpha=0.5)
        ax_vl.legend(fontsize=9)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), layout="constrained")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plot_ax = ax.semilogy if log_scale else ax.plot

        for i, (d, lbl, tr, vl) in enumerate(data_list):
            color = color_cycle[i % len(color_cycle)]
            if mode in ("both", "train") and tr is not None:
                epochs = np.arange(len(tr))
                tr_label = f"{lbl} (Train)" if mode == "both" else lbl
                plot_ax(epochs, tr, label=tr_label, color=color, linestyle="-", linewidth=2.0)

            if mode in ("both", "val") and vl is not None:
                epochs = np.arange(len(vl))
                vl_label = f"{lbl} (Val)" if mode == "both" else lbl
                linestyle = "--" if mode == "both" else "-"
                plot_ax(epochs, vl, label=vl_label, color=color, linestyle=linestyle, linewidth=2.0, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved loss comparison plot to {out_path}")
    return out_path


def plot_per_bead_force_losses(per_bead_data: dict, out_dir: str) -> None:
    """
    Plot per-bead force MSE losses over training epochs and final per-bead loss bar chart with variance.

    Parameters
    ----------
    per_bead_data : dict
        Dictionary containing 'epochs', 'val_mean' (shape: [epochs, n_beads]), and 'val_var' (shape: [epochs, n_beads]).
    out_dir : str
        Output directory to save the figures.
    """
    import os
    if not per_bead_data or "val_mean" not in per_bead_data:
        return

    epochs = np.array(per_bead_data["epochs"])
    means = np.array(per_bead_data["val_mean"])  # shape: (n_epochs, n_beads)
    vars = np.array(per_bead_data["val_var"])    # shape: (n_epochs, n_beads)
    stds = np.sqrt(np.maximum(vars, 0.0))

    if means.ndim != 2 or means.shape[0] == 0:
        return

    n_epochs, n_beads = means.shape
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_beads)))

    # Figure 1: Trajectory of force loss per bead over epochs with shaded std dev band
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), layout="constrained")
    for b in range(n_beads):
        mean_b = means[:, b]
        std_b = stds[:, b]
        ax.plot(epochs, mean_b, label=f"Bead {b}", color=colors[b % len(colors)], linewidth=2.0)
        ax.fill_between(
            epochs,
            np.maximum(mean_b - std_b, 0.0),
            mean_b + std_b,
            color=colors[b % len(colors)],
            alpha=0.15,
        )

    ax.set_title("Per-Bead Force Loss Convergence (\u03bc \u00b1 \u03c3)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Force MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc="upper left")

    os.makedirs(out_dir, exist_ok=True)
    out_traj_file = os.path.join(out_dir, "per_bead_force_losses.png")
    fig.savefig(out_traj_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved per-bead force loss plot to {out_traj_file}")

    # Figure 2: Final epoch per-bead force loss bar chart with error bars
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(6.5, 4.5), layout="constrained")
    final_means = means[-1]
    final_stds = stds[-1]
    bead_indices = np.arange(n_beads)

    ax_bar.bar(
        bead_indices,
        final_means,
        yerr=final_stds,
        capsize=4,
        color=colors[:n_beads],
        edgecolor="black",
        alpha=0.85,
    )

    ax_bar.set_title(f"Final Epoch ({epochs[-1]}) Per-Bead Force Loss", fontsize=13, fontweight="bold")
    ax_bar.set_xlabel("Bead Index")
    ax_bar.set_ylabel("Force MSE Loss")
    ax_bar.set_xticks(bead_indices)
    ax_bar.set_xticklabels([f"Bead {i}" for i in bead_indices], rotation=45 if n_beads > 10 else 0)
    ax_bar.grid(True, linestyle="--", alpha=0.5, axis="y")

    out_bar_file = os.path.join(out_dir, "per_bead_force_losses_final.png")
    fig_bar.savefig(out_bar_file, bbox_inches="tight", dpi=300)
    plt.close(fig_bar)
    print(f"[INFO] Saved final per-bead force loss bar chart to {out_bar_file}")



