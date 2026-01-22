"""
Shared styling constants and setup utilities for plotting.
"""

import matplotlib.pyplot as plt

# Color palette
color_ref = "#332288"
color_mace = "#88CCEE"
color_fm = "#44AA99"
colors = ["#88CCEE", "#44AA99", "#DDCC77", "#CC6677"]

# Extended color palette for multiple datasets
colors_extended = [
    "#332288",  # color_ref
    "#88CCEE",  # color_mace
    "#44AA99",  # color_fm
    "#DDCC77",
    "#CC6677",
    "#66CC99",
    "#FF6B6B",
    "#4A90E2",
    "#50514F",
    "#F4A261",
]

# Font sizes
tick_font_size = 16
axis_label_font_size = 16
legend_font_size = 16

# Line width
line_width = 3


def setup_plot_style():
    """
    Set up matplotlib rcParams with consistent styling.
    """
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = tick_font_size
    plt.rcParams["axes.labelsize"] = axis_label_font_size
    plt.rcParams["axes.titlesize"] = axis_label_font_size
    plt.rcParams["xtick.labelsize"] = tick_font_size
    plt.rcParams["ytick.labelsize"] = tick_font_size
    plt.rcParams["legend.fontsize"] = legend_font_size
    plt.rcParams["lines.linewidth"] = line_width
