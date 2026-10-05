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


def plot_lr_schedule(
    lr_schedule: str,
    epochs: int,
    out_dir: str,
    init_lr: float = 0.001,
    decay_rate: float = 0.90,
    end_lr: float = 1e-5,
) -> None:
    """
    Plot the learning rate schedule over epochs and save to out_dir.

    Parameters
    ----------
    lr_schedule : str
        Schedule type: 'constant', 'linear', or 'exponential'
    epochs : int
        Number of epochs to train
    out_dir : str
        Output directory to save the figure
    init_lr : float, optional
        Initial learning rate (default: 0.001)
    decay_rate : float, optional
        Decay rate for exponential schedule (default: 0.90)
    end_lr : float, optional
        End learning rate for linear schedule (default: 1e-5)
    """
    epochs_arr = np.arange(epochs)
    schedule_str = str(lr_schedule).lower()

    if schedule_str == "constant":
        lrs = [init_lr] * epochs
    elif schedule_str == "linear":
        if epochs <= 1:
            lrs = [init_lr] * epochs
        else:
            lrs = [init_lr + (end_lr - init_lr) * (e / (epochs - 1)) for e in epochs_arr]
    elif schedule_str == "exponential":
        if epochs <= 1:
            lrs = [init_lr] * epochs
        else:
            lrs = [init_lr * (decay_rate ** (e / (epochs - 1))) for e in epochs_arr]
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}. Expected 'constant', 'linear', or 'exponential'.")

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), layout="constrained")
    ax.plot(epochs_arr, lrs, marker="o", markersize=4, color="#1f77b4", linewidth=2.0)
    ax.set_title(f"Learning Rate Schedule ({schedule_str.capitalize()})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/learning_rate_schedule.png", bbox_inches="tight", dpi=300)
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


def plot_atom_and_cg_embeddings_pca(
    saved_atom_embeddings: dict,
    saved_cg_maps: dict,
    out_dir: str,
    mol_name: str = "ala2",
    epoch: int | None = None,
    save_prefix: str = "atom_and_cg_embeddings_pca",
) -> None:
    """
    Visualize learned atom embeddings as vectors and combined CG bead embeddings
    in the same PCA projection for a given epoch (default: final epoch).

    Parameters
    ----------
    saved_atom_embeddings : dict
        Dict mapping epoch (int or str) -> {str(spec): list of float embeddings}
    saved_cg_maps : dict
        Dict mapping epoch (int or str) -> list of int bead assignments per atom
    out_dir : str
        Output directory to save the plots
    mol_name : str, optional
        Molecule name ('ala2', 'ala15', 'hexane', etc.), by default 'ala2'
    epoch : int, optional
        Epoch to visualize. If None, the final epoch is used.
    save_prefix : str, optional
        Prefix for output filenames, by default 'atom_and_cg_embeddings_pca'
    """
    import os
    from mpl_toolkits.mplot3d import Axes3D

    if not saved_atom_embeddings or not saved_cg_maps:
        print("[WARNING] Missing saved_atom_embeddings or saved_cg_maps for PCA visualization.")
        return

    # Determine epoch to plot
    common_epochs = sorted(
        list(set(int(k) for k in saved_atom_embeddings.keys()) & set(int(k) for k in saved_cg_maps.keys()))
    )
    if not common_epochs:
        print("[WARNING] No overlapping epochs between saved_atom_embeddings and saved_cg_maps.")
        return

    if epoch is None or int(epoch) not in common_epochs:
        target_epoch = common_epochs[-1]
    else:
        target_epoch = int(epoch)

    ep_atom_data = saved_atom_embeddings.get(target_epoch) or saved_atom_embeddings.get(str(target_epoch))
    ep_cg_map = saved_cg_maps.get(target_epoch) or saved_cg_maps.get(str(target_epoch))

    if ep_atom_data is None or ep_cg_map is None:
        print(f"[WARNING] Data for epoch {target_epoch} not found.")
        return

    # Extract unique atom species and embeddings
    species_order = sorted([int(k) for k in ep_atom_data.keys()])
    species_symbols = {s: ATOMIC_NUMBER_TO_SYMBOL.get(s, f"S_{s}") for s in species_order}
    species_colors_palette = {
        1: "#0096C7",  # Hydrogen: Sky Blue
        6: "#212529",  # Carbon: Charcoal / Near Black
        7: "#3A0CA3",  # Nitrogen: Deep Indigo
        8: "#D90429",  # Oxygen: Vivid Crimson Red
    }
    species_colors = {
        s: species_colors_palette.get(s, plt.cm.tab10(idx % 10))
        for idx, s in enumerate(species_order)
    }

    E_atoms = np.array([ep_atom_data[str(s)] if str(s) in ep_atom_data else ep_atom_data[s] for s in species_order])
    num_atoms = len(ep_cg_map)

    # Determine atom species sequence for the molecule
    mol_species = None
    try:
        from cgbench.core import mapping
        map_class_name = f"{mol_name.capitalize()}_Map"
        map_class = getattr(mapping, map_class_name, None)
        if map_class is not None and hasattr(map_class, "map_species"):
            raw_specs = map_class.map_species
            mol_species = [mapping.atomic_number_map_reverse.get(s, s) for s in raw_specs]
    except Exception:
        pass

    if mol_species is None or len(mol_species) != num_atoms:
        if num_atoms == 22:
            mol_species = [6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]
        elif num_atoms == 20:
            mol_species = [6, 1, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 1]
        else:
            mol_species = [species_order[0]] * num_atoms

    # Compute CG bead combined embeddings (vector sum of constituent atom embeddings)
    num_beads = max(ep_cg_map) + 1 if ep_cg_map else 0
    emb_dim = E_atoms.shape[1]
    E_beads = np.zeros((num_beads, emb_dim))
    bead_info = {}

    for b in range(num_beads):
        atoms_in_b = [i for i, b_idx in enumerate(ep_cg_map) if b_idx == b]
        specs = [mol_species[i] for i in atoms_in_b]
        spec_counts = {species_symbols[s]: specs.count(s) for s in set(specs)}
        comp_str = "".join([f"{k}{v if v > 1 else ''}" for k, v in sorted(spec_counts.items())])
        bead_info[b] = {
            "atoms": atoms_in_b,
            "species": specs,
            "comp": comp_str or "Empty",
            "is_empty": len(atoms_in_b) == 0,
        }
        for s in specs:
            if s in species_order:
                E_beads[b] += E_atoms[species_order.index(s)]

    # SVD on atom embeddings for origin-preserving PCA (preserves linear vector addition)
    u, s_vals, vt = np.linalg.svd(E_atoms, full_matrices=False)
    n_comp = min(3, len(s_vals))
    W = vt[:n_comp].T
    var_exp = (s_vals**2) / np.sum(s_vals**2)

    proj_atoms = E_atoms @ W
    proj_beads = E_beads @ W

    # Group beads with identical 2D coordinates
    grouped_beads = {}
    for b in range(num_beads):
        if bead_info[b]["is_empty"]:
            continue
        key = (round(float(proj_beads[b, 0]), 4), round(float(proj_beads[b, 1]), 4))
        if key not in grouped_beads:
            grouped_beads[key] = []
        grouped_beads[key].append(b)

    bead_colors = ["#F72585", "#7209B7", "#4CC9F0", "#F77F00", "#06D6A0", "#FB8500", "#118AB2", "#8338EC", "#2EC4B6", "#E71D36"]

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------
    # 1. STANDALONE 2D VECTOR PLOT
    # ----------------------------------------------------
    fig2d, ax2d = plt.subplots(figsize=(11, 10), dpi=300)
    ax2d.set_facecolor("#FAFAFC")

    all_x = np.concatenate([proj_atoms[:, 0], proj_beads[:, 0], [0]])
    all_y = np.concatenate([proj_atoms[:, 1], proj_beads[:, 1], [0]])
    pad_x = (all_x.max() - all_x.min()) * 0.20
    pad_y = (all_y.max() - all_y.min()) * 0.25
    ax2d.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax2d.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    ax2d.axhline(0, color="#ADB5BD", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax2d.axvline(0, color="#ADB5BD", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax2d.scatter(0, 0, color="#212529", s=60, zorder=5)
    ax2d.text(0.02, -0.04, "(0, 0)", fontsize=10, color="#6C757D", fontweight="semibold")

    # Vector Addition Guides for Composite Beads
    for b in range(num_beads):
        if bead_info[b]["is_empty"]:
            continue
        specs = bead_info[b]["species"]
        bx, by = proj_beads[b, :2]
        if len(specs) == 2 and all(s in species_order for s in specs):
            p1 = proj_atoms[species_order.index(specs[0]), :2]
            p2 = proj_atoms[species_order.index(specs[1]), :2]
            ax2d.plot([p1[0], bx], [p1[1], by], linestyle=":", color=species_colors[specs[1]], lw=1.8, alpha=0.8, zorder=2)
            ax2d.plot([p2[0], bx], [p2[1], by], linestyle=":", color=species_colors[specs[0]], lw=1.8, alpha=0.8, zorder=2)
        elif len(specs) > 2 and len(set(specs)) == 2:
            heavy = [s for s in specs if s != 1]
            if heavy and 1 in specs:
                h_s = heavy[0]
                h_coord = proj_atoms[species_order.index(1), :2]
                base_coord = proj_atoms[species_order.index(h_s), :2]
                curr = base_coord.copy()
                step_pts = [curr.copy()]
                for _ in range(specs.count(1)):
                    curr += h_coord
                    step_pts.append(curr.copy())
                step_pts = np.array(step_pts)
                ax2d.plot(step_pts[:, 0], step_pts[:, 1], linestyle="--", color=species_colors[1], lw=1.4, alpha=0.5, zorder=2)

    # Plot Atom Vectors
    for idx, s in enumerate(species_order):
        sym = species_symbols[s]
        col = species_colors[s]
        x, y = proj_atoms[idx, 0], proj_atoms[idx, 1]
        ax2d.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.7", lw=3.8, color=col, alpha=0.95),
            zorder=4,
        )
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"
        offset_x = 0.05 if x >= 0 else -0.05
        offset_y = 0.04 if y >= 0 else -0.04
        if s == 7:  # N points down
            offset_y = -0.06
            offset_x = 0.0
            va = "top"
            ha = "center"
        elif s == 1:  # H points left-down
            offset_x = -0.05
            offset_y = -0.03
            ha = "right"
            va = "top"

        ax2d.text(
            x + offset_x, y + offset_y,
            sym,
            fontsize=15, fontweight="bold", color=col,
            ha=ha, va=va, zorder=6,
        )

    # Plot Bead Vectors and Endpoints
    for i, (coord, b_list) in enumerate(grouped_beads.items()):
        bx, by = coord
        first_b = b_list[0]
        comp = bead_info[first_b]["comp"]
        b_color = bead_colors[i % len(bead_colors)]
        is_single_atom = any(np.allclose([bx, by], proj_atoms[a_idx, :2], atol=1e-3) for a_idx in range(len(species_order)))

        if not is_single_atom:
            ax2d.annotate(
                "", xy=(bx, by), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.55", lw=2.2, color=b_color, linestyle="-", alpha=0.8),
                zorder=3,
            )
            marker = "D"
            size = 130
        else:
            marker = "o"
            size = 140

        ax2d.scatter(bx, by, s=size, color=b_color, edgecolor="white", linewidth=1.8, zorder=5, marker=marker)

        # If identical to a single atom (e.g. single H), skip redundant label
        if is_single_atom:
            continue

        ha = "left" if bx >= 0 else "right"
        va = "bottom" if by >= 0 else "top"
        offset_bx = 0.05 if bx >= 0 else -0.05
        offset_by = 0.04 if by >= 0 else -0.04

        if comp == "CO":
            offset_by = 0.05
            offset_bx = 0.0
            va = "bottom"
            ha = "center"
        elif comp == "CNO":
            offset_by = -0.05
            offset_bx = -0.05
            va = "top"
            ha = "right"
        elif comp == "CHN":
            offset_bx = 0.05
            offset_by = -0.05
            va = "top"
            ha = "left"
        elif comp == "CH3":
            offset_bx = -0.05
            offset_by = 0.04
            ha = "right"
            va = "bottom"
        elif comp == "CH":
            offset_bx = 0.05
            offset_by = 0.04
            ha = "left"
            va = "bottom"

        ax2d.text(
            bx + offset_bx, by + offset_by,
            comp,
            fontsize=13, fontweight="bold", color=b_color,
            ha=ha, va=va, zorder=6,
        )

    ax2d.set_title(f"Atom & Combined CG Bead Embeddings in PCA Space (Epoch {target_epoch})", fontsize=15, fontweight="bold", pad=15)
    ax2d.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=12, fontweight="semibold")
    ax2d.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=12, fontweight="semibold")
    ax2d.grid(True, linestyle=":", color="#CED4DA", alpha=0.7)
    plt.tight_layout()

    out_file_2d = f"{out_dir}/{save_prefix}_2d.png"
    fig2d.savefig(out_file_2d, dpi=300, bbox_inches="tight")
    plt.close(fig2d)
    print(f"[INFO] Saved 2D PCA vector plot to {out_file_2d}")

    # ----------------------------------------------------
    # 2. MULTI-PANEL FIGURE (2D Vector Space + 3D + Cosine Sim)
    # ----------------------------------------------------
    fig = plt.figure(figsize=(18, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_facecolor("#FAFAFC")
    ax_main.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_main.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax_main.axhline(0, color="#ADB5BD", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax_main.axvline(0, color="#ADB5BD", linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax_main.scatter(0, 0, color="#212529", s=60, zorder=5)

    # Re-plot guides on multi-panel
    for b in range(num_beads):
        if bead_info[b]["is_empty"]:
            continue
        specs = bead_info[b]["species"]
        bx, by = proj_beads[b, :2]
        if len(specs) == 2 and all(s in species_order for s in specs):
            p1 = proj_atoms[species_order.index(specs[0]), :2]
            p2 = proj_atoms[species_order.index(specs[1]), :2]
            ax_main.plot([p1[0], bx], [p1[1], by], linestyle=":", color=species_colors[specs[1]], lw=1.8, alpha=0.8, zorder=2)
            ax_main.plot([p2[0], bx], [p2[1], by], linestyle=":", color=species_colors[specs[0]], lw=1.8, alpha=0.8, zorder=2)
        elif len(specs) > 2 and len(set(specs)) == 2:
            heavy = [s for s in specs if s != 1]
            if heavy and 1 in specs:
                h_s = heavy[0]
                h_coord = proj_atoms[species_order.index(1), :2]
                base_coord = proj_atoms[species_order.index(h_s), :2]
                curr = base_coord.copy()
                step_pts = [curr.copy()]
                for _ in range(specs.count(1)):
                    curr += h_coord
                    step_pts.append(curr.copy())
                step_pts = np.array(step_pts)
                ax_main.plot(step_pts[:, 0], step_pts[:, 1], linestyle="--", color=species_colors[1], lw=1.4, alpha=0.5, zorder=2)

    for idx, s in enumerate(species_order):
        sym = species_symbols[s]
        col = species_colors[s]
        x, y = proj_atoms[idx, 0], proj_atoms[idx, 1]
        ax_main.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.65", lw=3.5, color=col, alpha=0.95), zorder=4)
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"
        offset_x = 0.05 if x >= 0 else -0.05
        offset_y = 0.04 if y >= 0 else -0.04
        if s == 7:
            offset_y = -0.06
            offset_x = 0.0
            va = "top"
            ha = "center"
        elif s == 1:
            offset_x = -0.05
            offset_y = -0.03
            ha = "right"
            va = "top"
        ax_main.text(x + offset_x, y + offset_y, sym, fontsize=14, fontweight="bold", color=col, ha=ha, va=va, zorder=6)

    for i, (coord, b_list) in enumerate(grouped_beads.items()):
        bx, by = coord
        first_b = b_list[0]
        comp = bead_info[first_b]["comp"]
        b_color = bead_colors[i % len(bead_colors)]
        is_single_atom = any(np.allclose([bx, by], proj_atoms[a_idx, :2], atol=1e-3) for a_idx in range(len(species_order)))
        if not is_single_atom:
            ax_main.annotate("", xy=(bx, by), xytext=(0, 0), arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.5", lw=2.0, color=b_color, alpha=0.8), zorder=3)
            marker = "D"
            size = 120
        else:
            marker = "o"
            size = 130
        ax_main.scatter(bx, by, s=size, color=b_color, edgecolor="white", linewidth=1.8, zorder=5, marker=marker)

        if is_single_atom:
            continue

        ha = "left" if bx >= 0 else "right"
        va = "bottom" if by >= 0 else "top"
        offset_bx = 0.05 if bx >= 0 else -0.05
        offset_by = 0.04 if by >= 0 else -0.04
        if comp == "CO":
            offset_by = 0.05
            offset_bx = 0.0
            va = "bottom"
            ha = "center"
        elif comp == "CNO":
            offset_by = -0.05
            offset_bx = -0.05
            va = "top"
            ha = "right"
        elif comp == "CHN":
            offset_bx = 0.05
            offset_by = -0.05
            va = "top"
            ha = "left"
        elif comp == "CH3":
            offset_bx = -0.05
            offset_by = 0.04
            ha = "right"
            va = "bottom"
        elif comp == "CH":
            offset_bx = 0.05
            offset_by = 0.04
            ha = "left"
            va = "bottom"

        ax_main.text(bx + offset_bx, by + offset_by, comp, fontsize=12, fontweight="bold", color=b_color, ha=ha, va=va, zorder=6)

    ax_main.set_title("A. 2D PCA Vector Space (Origin-Preserving)", fontsize=13, fontweight="bold")
    ax_main.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11, fontweight="semibold")
    ax_main.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11, fontweight="semibold")
    ax_main.grid(True, linestyle=":", color="#CED4DA", alpha=0.7)

    # Panel B: 3D PCA Projection
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3d.view_init(elev=22, azim=45)
    ax3d.scatter([0], [0], [0], color="black", s=30)
    for idx, s in enumerate(species_order):
        sym = species_symbols[s]
        col = species_colors[s]
        p = proj_atoms[idx]
        pz = p[2] if len(p) > 2 else 0.0
        ax3d.quiver(0, 0, 0, p[0], p[1], pz, color=col, lw=2.5, arrow_length_ratio=0.15, alpha=0.9)
        ax3d.text(p[0] * 1.15, p[1] * 1.15, pz * 1.15, sym, color=col, fontsize=11, fontweight="bold")

    for i, (coord, b_list) in enumerate(grouped_beads.items()):
        first_b = b_list[0]
        comp = bead_info[first_b]["comp"]
        b_color = bead_colors[i % len(bead_colors)]
        p = proj_beads[first_b]
        pz = p[2] if len(p) > 2 else 0.0
        is_single_atom = any(np.allclose(p[:2], proj_atoms[a_idx, :2], atol=1e-3) for a_idx in range(len(species_order)))
        if not is_single_atom:
            ax3d.quiver(0, 0, 0, p[0], p[1], pz, color=b_color, lw=1.5, linestyle=":", arrow_length_ratio=0.12, alpha=0.7)
            marker = "D"
            ax3d.text(p[0] * 1.08, p[1] * 1.08, pz * 1.08, comp, fontsize=9, fontweight="bold", color=b_color)
        else:
            marker = "o"
        ax3d.scatter(p[0], p[1], pz, color=b_color, s=70, edgecolor="black", marker=marker)

    var_3d = float(np.sum(var_exp[:3])) * 100
    ax3d.set_title(f"B. 3D PCA Space ({var_3d:.1f}% Variance Captured)", fontsize=13, fontweight="bold")
    ax3d.set_xlabel("PC 1", fontsize=9)
    ax3d.set_ylabel("PC 2", fontsize=9)
    ax3d.set_zlabel("PC 3", fontsize=9)

    # Panel C: Cosine Similarity Heatmap between Atoms and Unique Beads
    ax_cos = fig.add_subplot(gs[1, 1])
    unique_bead_indices = [b_list[0] for b_list in grouped_beads.values()]
    unique_bead_labels = [bead_info[b_list[0]]["comp"] for b_list in grouped_beads.values()]

    norm_atoms = E_atoms / np.maximum(np.linalg.norm(E_atoms, axis=-1, keepdims=True), 1e-8)
    E_u_beads = E_beads[unique_bead_indices]
    norm_beads = E_u_beads / np.maximum(np.linalg.norm(E_u_beads, axis=-1, keepdims=True), 1e-8)
    cos_sim = norm_beads @ norm_atoms.T

    im = ax_cos.imshow(cos_sim, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax_cos.set_xticks(np.arange(len(species_order)))
    ax_cos.set_xticklabels([species_symbols[s] for s in species_order], fontsize=11, fontweight="bold")
    ax_cos.set_yticks(np.arange(len(unique_bead_labels)))
    ax_cos.set_yticklabels(unique_bead_labels, fontsize=10.5, fontweight="bold")
    ax_cos.set_title("C. Directional Cosine Similarity (Beads vs Atoms)", fontsize=13, fontweight="bold")

    for r in range(cos_sim.shape[0]):
        for c in range(cos_sim.shape[1]):
            val = cos_sim[r, c]
            txt_col = "white" if abs(val) > 0.55 else "black"
            ax_cos.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_col, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_cos, shrink=0.85, pad=0.03)
    cbar.set_label("Cosine Similarity", fontsize=9.5)

    fig.suptitle(f"Learned Atom Embeddings & CG Bead Vector Composition ({mol_name.capitalize()}, Epoch {target_epoch})", fontsize=16, fontweight="bold", y=0.98)
    out_multi = f"{out_dir}/{save_prefix}.png"
    fig.savefig(out_multi, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved multi-panel atom & CG bead embeddings PCA visualization to {out_multi}")


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



