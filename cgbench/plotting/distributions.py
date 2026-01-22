"""
1D histogram plotting for structural distributions (dihedrals, angles, bonds).
"""

import numpy as np
from matplotlib.axes import Axes

from .style import (
    color_ref,
    color_mace,
    color_fm,
    colors_extended,
    tick_font_size,
    axis_label_font_size,
    legend_font_size,
    line_width,
)


def plot_1d_dihedral(
    ax: Axes,
    angles: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = "$\\phi$ (deg)",
    plot_legend: bool = True,
    ylabel: bool = True,
    tick_bin: float = 90,
    mode: str = "single",
    n_std: int = 1,
) -> Axes:
    """
    Plot 1D dihedral angle distributions with support for single or multiple chains.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to plot on
    angles : list[np.ndarray]
        List of angle arrays. Each can be:
        - 1D array (n_frames) for single chain
        - 2D array (n_chains, n_frames) for multiple chains
    labels : list[str]
        Labels for each dataset
    bins : int
        Number of histogram bins
    degrees : bool
        If True, angles are in degrees; if False, in radians
    xlabel : str
        X-axis label
    plot_legend : bool
        Whether to show legend
    ylabel : bool
        Whether to show y-axis label
    tick_bin : float
        Spacing for x-axis ticks
    mode : str
        'single' for original behavior, 'multi' for multi-chain with std
    n_std : int
        Number of standard deviations for fill_between in multi mode

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    n_models = len(angles)

    for i in range(n_models):
        ang = angles[i]

        # -----------------------------------
        # mode="single" (original behavior)
        # -----------------------------------
        if mode == "single":
            if degrees:
                # Convert angles to [-180, 180] range
                angles_conv = ((ang + 180) % 360) - 180
                hist_range = [-180, 180]
            else:
                # Convert angles to [-π, π] range
                angles_conv = ((ang + np.pi) % (2 * np.pi)) - np.pi
                hist_range = [-np.pi, np.pi]

            hist, x_bins = np.histogram(
                angles_conv, bins=bins, density=True, range=hist_range
            )
            width = x_bins[1] - x_bins[0]
            bin_center = x_bins[:-1] + width / 2

            ax.plot(
                bin_center,
                hist,
                label=labels[i],
                color=color[i % len(color)],
                linewidth=line_width,
            )

        # -----------------------------------
        # mode="multi" (single or multi-chain)
        # -----------------------------------
        elif mode == "multi":
            # Set histogram range based on units
            if degrees:
                hist_range = [-180, 180]
            else:
                hist_range = [-np.pi, np.pi]

            # Handle single chain (1D array)
            if ang.ndim == 1:
                if degrees:
                    angles_conv = ((ang + 180) % 360) - 180
                else:
                    angles_conv = ((ang + np.pi) % (2 * np.pi)) - np.pi

                hist, x_bins = np.histogram(
                    angles_conv, bins=bins, density=True, range=hist_range
                )
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                ax.plot(
                    bin_center,
                    hist,
                    color=color[i % len(color)],
                    linewidth=line_width,
                    label=labels[i],
                )

            # Handle multiple chains (2D array)
            elif ang.ndim == 2:
                n_chains = ang.shape[0]

                # Create bin edges once
                _, x_bins = np.histogram([], bins=bins, range=hist_range)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                chain_hists = []
                for c in range(n_chains):
                    if degrees:
                        angles_conv = ((ang[c] + 180) % 360) - 180
                    else:
                        angles_conv = ((ang[c] + np.pi) % (2 * np.pi)) - np.pi

                    hist_c, _ = np.histogram(
                        angles_conv, bins=x_bins, density=True, range=hist_range
                    )
                    chain_hists.append(hist_c)

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                col = color[i % len(color)]

                # Plot mean curve
                ax.plot(
                    bin_center,
                    hist_mean,
                    color=col,
                    linewidth=line_width,
                    label=labels[i],
                )

                # Plot ± n_std fill
                ax.fill_between(
                    bin_center,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.4,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    # Decorations
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    if ylabel:
        ax.set_ylabel("Density", fontsize=axis_label_font_size)

    # Set x-axis ticks
    if degrees:
        ax.set_xticks(np.arange(-180, 181, tick_bin))
        ax.set_xlim(-180, 180)
    else:
        ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, tick_bin))
        ax.set_xlim(-np.pi, np.pi)

    ax.tick_params(direction="in", labelsize=tick_font_size)
    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)

    ax.set_ylim(bottom=0)

    return ax


def plot_1d_dihedral_mean_std(
    ax: Axes,
    angles: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = "$\\phi$ in deg",
    ylabel: bool = True,
    color: str = "blue",
) -> Axes:
    """
    Plot 1D histogram with mean and standard deviation as shaded area.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[np.ndarray]
        List of angle arrays, one for each model/dataset
    labels : list[str]
        Labels for each angle dataset
    bins : int
        Number of histogram bins
    degrees : bool
        Whether angles are in degrees
    xlabel : str
        Label for x-axis
    ylabel : bool
        Whether to add y-axis label
    color : str
        Color for the plot

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    n_models = len(angles)
    for i in range(n_models):
        data = np.array(angles[i])
        if degrees:
            data_conv = data
            hist_range = [-180, 180]
        else:
            data_conv = np.rad2deg(data)
            hist_range = [-np.pi, np.pi]

        # Compute histogram for each sample, then mean/std over samples
        if data_conv.ndim == 2:
            hists = []
            for sample in data_conv:
                hist, x_bins = np.histogram(
                    sample, bins=bins, density=True, range=hist_range
                )
                hists.append(hist)
            hists = np.stack(hists)
            hist_mean = hists.mean(axis=0)
            hist_std = hists.std(axis=0)
        else:
            hist_mean, x_bins = np.histogram(
                data_conv, bins=bins, density=True, range=hist_range
            )
            hist_std = np.zeros_like(hist_mean)

        width = x_bins[1] - x_bins[0]
        bin_center = x_bins[:-1] + width / 2

        ax.plot(bin_center, hist_mean, label=labels[i], color=color, linewidth=2.0)
        ax.fill_between(
            bin_center,
            hist_mean - hist_std,
            hist_mean + hist_std,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_1d_angle(
    ax: Axes,
    angles: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    xlabel: str = "$\\Theta$ (deg)",
    plot_legend: bool = True,
    ylabel: bool = True,
    degrees: bool = True,
    mode: str = "single",
    n_std: int = 1,
) -> Axes:
    """
    Plot 1D bond angle distributions.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[np.ndarray]
        List of angle arrays
    labels : list[str]
        Labels for each dataset
    bins : int
        Number of histogram bins
    xlabel : str
        X-axis label
    plot_legend : bool
        Whether to show legend
    ylabel : bool
        Whether to show y-axis label
    degrees : bool
        If True, convert radians to degrees
    mode : str
        'single' or 'multi' for multi-chain with std
    n_std : int
        Number of standard deviations for fill_between

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    n_models = len(angles)

    for i in range(n_models):
        ang = angles[i]

        if mode == "single":
            if degrees:
                ang = np.rad2deg(ang)

            hist_range = [ang.min(), ang.max()]
            hist, x_bins = np.histogram(ang, bins=bins, range=hist_range)
            hist = hist / np.sum(hist)

            width = x_bins[1] - x_bins[0]
            bin_center = x_bins[:-1] + width / 2

            ax.plot(
                bin_center,
                hist,
                color=color[i % len(color)],
                linewidth=line_width,
                label=labels[i],
            )

        elif mode == "multi":
            if degrees:
                ang = np.rad2deg(ang)

            # Handle single chain (1D array)
            if ang.ndim == 1:
                hist_range = [ang.min(), ang.max()]
                hist, x_bins = np.histogram(ang, bins=bins, range=hist_range)
                hist = hist / np.sum(hist)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                ax.plot(
                    bin_center,
                    hist,
                    color=color[i % len(color)],
                    linewidth=line_width,
                    label=labels[i],
                )

            # Handle multiple chains (2D array)
            elif ang.ndim == 2:
                n_chains = ang.shape[0]

                hist_range = [ang.min(), ang.max()]
                _, x_bins = np.histogram(ang[0], bins=bins, range=hist_range)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                chain_hists = []
                for c in range(n_chains):
                    hist_c, _ = np.histogram(ang[c], bins=x_bins, range=hist_range)
                    hist_c = hist_c / np.sum(hist_c)
                    chain_hists.append(hist_c)

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                col = color[i % len(color)]

                ax.plot(
                    bin_center,
                    hist_mean,
                    color=col,
                    linewidth=line_width,
                    label=labels[i],
                )
                ax.fill_between(
                    bin_center,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.4,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    ax.tick_params(direction="in", labelsize=tick_font_size)
    if ylabel:
        ax.set_ylabel("Density", fontsize=axis_label_font_size)
    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)
    ax.set_ylim(bottom=0)

    return ax


def plot_1d_bond(
    ax: Axes,
    bonds: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    xlabel: str = "b (nm)",
    ylabel: bool = True,
    plot_legend: bool = True,
    tick_bin: float = 0.01,
    mode: str = "single",
    n_std: int = 1,
) -> Axes:
    """
    Plot 1D bond length distributions.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    bonds : list[np.ndarray]
        List of bond length arrays
    labels : list[str]
        Labels for each dataset
    bins : int
        Number of histogram bins
    xlabel : str
        X-axis label
    ylabel : bool
        Whether to show y-axis label
    plot_legend : bool
        Whether to show legend
    tick_bin : float
        Spacing for x-axis ticks
    mode : str
        'single' or 'multi' for multi-chain with std
    n_std : int
        Number of standard deviations for fill_between

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    colors = [
        color_ref,
        color_mace,
        color_fm,
        "#FFB347",
        "#7851A9",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#F4A261",
    ]

    # Shared bin edges across ALL datasets
    gmin = min(a.min() for a in bonds)
    gmax = max(a.max() for a in bonds)

    if np.isclose(gmin, gmax):
        gmin -= 1e-6
        gmax += 1e-6

    bin_edges = np.linspace(gmin, gmax, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for i, (bond_data, lab) in enumerate(zip(bonds, labels)):
        col = colors[i % len(colors)]

        print(f"Processing bond data for '{lab}': {bond_data.shape}")

        if mode == "single":
            counts, _ = np.histogram(bond_data, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                continue

            frac = counts / total
            ax.plot(bin_centers, frac, label=lab, color=col, linewidth=line_width)

        elif mode == "multi":
            if bond_data.ndim == 1:
                counts, _ = np.histogram(bond_data, bins=bin_edges)
                total = counts.sum()
                if total == 0:
                    continue

                frac = counts / total
                ax.plot(bin_centers, frac, label=lab, color=col, linewidth=line_width)

            elif bond_data.ndim == 2:
                n_chains = bond_data.shape[0]

                chain_hists = []
                for c in range(n_chains):
                    bc = bond_data[c]
                    bc = bc[np.isfinite(bc)]
                    if bc.size == 0:
                        continue

                    counts_c, _ = np.histogram(bc, bins=bin_edges)
                    total_c = counts_c.sum()
                    if total_c == 0:
                        continue

                    chain_hists.append(counts_c / total_c)

                if len(chain_hists) == 0:
                    continue

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                print(
                    f"Plotting bond distribution for '{lab}': {n_chains} chains, "
                    f"mean={hist_mean.sum()}, std sum={hist_std.sum()}"
                )

                ax.plot(
                    bin_centers, hist_mean, color=col, linewidth=line_width, label=lab
                )

                ax.fill_between(
                    bin_centers,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.5,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)

    if ylabel:
        ax.set_ylabel("Density", fontsize=axis_label_font_size)

    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)

    ax.tick_params(direction="in", labelsize=tick_font_size)
    ax.set_ylim(bottom=0)

    # tick spacing
    if tick_bin and tick_bin > 0:
        start = np.floor(gmin / tick_bin) * tick_bin
        stop = np.ceil(gmax / tick_bin) * tick_bin
        ticks = np.arange(start, stop + tick_bin * 0.5, tick_bin)
        ax.set_xticks(ticks)

    return ax
