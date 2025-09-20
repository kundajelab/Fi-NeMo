"""Visualization module for generating plots and reports for Fi-NeMo results.

This module provides functions for:
- Plotting motif contribution weight matrices (CWMs) as sequence logos
- Generating distribution plots for hit statistics
- Creating co-occurrence heatmaps
- Producing HTML reports with interactive visualizations
- Plotting confusion matrices and performance metrics
"""

import os
import importlib.resources
from typing import List, Optional, Dict, Any, Tuple, Union, Mapping, Iterable

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from jinja2 import Template
import polars as pl
from jaxtyping import Float, Int

from . import templates


def abbreviate_motif_name(name: str) -> str:
    """Convert TF-MoDISco motif names to abbreviated format.

    Converts full TF-MoDISco pattern names to shorter, more readable format
    for display in plots and reports.

    Parameters
    ----------
    name : str
        Full motif name (e.g., 'pos_patterns.pattern_0').

    Returns
    -------
    str
        Abbreviated name (e.g., '+/0') or original name if parsing fails.

    Examples
    --------
    >>> abbreviate_motif_name('pos_patterns.pattern_0')
    '+/0'
    >>> abbreviate_motif_name('neg_patterns.pattern_1')
    '-/1'
    >>> abbreviate_motif_name('invalid_name')
    'invalid_name'
    """
    try:
        group, motif = name.split(".")
        if group == "pos_patterns":
            group_short = "+"
        elif group == "neg_patterns":
            group_short = "-"
        else:
            raise Exception
        motif_num = motif.split("_")[1]
        return f"{group_short}/{motif_num}"
    except Exception:
        return name


def plot_hit_stat_distributions(
    hits_df: pl.LazyFrame, motif_names: List[str], plot_dir: str
) -> None:
    """Plot distributions of hit statistics for each motif.

    Creates separate histogram plots for coefficient, similarity, and importance
    score distributions for each motif. Saves plots in both PNG (high-res) and
    SVG (vector) formats.

    Parameters
    ----------
    hits_df : pl.LazyFrame
        Lazy DataFrame containing hit data with required columns:
        - motif_name : str, name of the motif
        - hit_coefficient_global : float, global coefficient values
        - hit_similarity : float, similarity scores to motif CWM
        - hit_importance : float, importance scores from attribution
    motif_names : List[str]
        List of motif names to generate plots for. Motifs not present
        in hits_df will result in empty histograms.
    plot_dir : str
        Directory path where plots will be saved. Creates subdirectory
        'motif_stat_distributions' if it doesn't exist.

    Notes
    -----
    For each motif, creates three separate plots:
    - {motif_name}_coefficients.{png,svg} : coefficient distribution
    - {motif_name}_similarities.{png,svg} : similarity distribution
    - {motif_name}_importances.{png,svg} : importance distribution
    """
    hits_df_collected = hits_df.collect()
    hits_by_motif = hits_df_collected.partition_by("motif_name", as_dict=True)
    dummy_df = hits_df_collected.clear()

    motifs_dir = os.path.join(plot_dir, "motif_stat_distributions")
    os.makedirs(motifs_dir, exist_ok=True)
    for m in motif_names:
        hits = hits_by_motif.get((m,), dummy_df)
        coefficients = hits.get_column("hit_coefficient_global").to_numpy()
        similarities = hits.get_column("hit_similarity").to_numpy()
        importances = hits.get_column("hit_importance").to_numpy()

        fig, ax = plt.subplots(figsize=(5, 2))

        # Plot coefficient distribution
        try:
            ax.hist(coefficients, bins=50, density=True)
        except ValueError:
            ax.hist(coefficients, bins=1, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_coefficients.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_coefficients.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        # Plot similarity distribution
        try:
            ax.hist(similarities, bins=50, density=True)
        except ValueError:
            ax.hist(similarities, bins=1, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_similarities.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_similarities.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        # Plot importance distribution
        try:
            ax.hist(importances, bins=50, density=True)
        except ValueError:
            ax.hist(importances, bins=1, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_importances.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_importances.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)


def plot_hit_peak_distributions(
    occ_df: pl.DataFrame, motif_names: List[str], plot_dir: str
) -> None:
    """Plot distribution of hits per peak for each motif.

    Creates bar plots showing the frequency distribution of hit counts per peak
    for each motif, plus an overall distribution of total hits per peak.

    Parameters
    ----------
    occ_df : pl.DataFrame
        DataFrame containing motif occurrence counts per peak. Expected to have:
        - One column per motif name with integer hit counts
        - 'total' column with sum of all motif hits per peak
        - Each row represents a peak/genomic region
    motif_names : List[str]
        List of motif names corresponding to columns in occ_df.
    plot_dir : str
        Directory to save plots. Creates 'motif_hit_distributions' subdirectory.

    Notes
    -----
    Generates the following plots:
    - Individual motif hit distributions: {motif_name}.{png,svg}
    - Overall hit distribution: total_hit_distribution.{png,svg}

    Bar plots show frequency (proportion) on y-axis and hit count on x-axis.
    """
    motifs_dir = os.path.join(plot_dir, "motif_hit_distributions")
    os.makedirs(motifs_dir, exist_ok=True)

    for m in motif_names:
        fig, ax = plt.subplots(figsize=(5, 2))

        unique, counts = np.unique(occ_df.get_column(m), return_counts=True)
        freq = counts / counts.sum()
        num_bins = np.amax(unique, initial=0) + 1
        x = np.arange(num_bins)
        y = np.zeros(num_bins)
        y[unique] = freq
        ax.bar(x, y)

        output_path_png = os.path.join(motifs_dir, f"{m}.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}.svg")
        plt.savefig(output_path_svg)

        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))

    unique, counts = np.unique(occ_df.get_column("total"), return_counts=True)
    freq = counts / counts.sum()
    num_bins = np.amax(unique, initial=0) + 1
    x = np.arange(num_bins)
    y = np.zeros(num_bins)
    y[unique] = freq
    ax.bar(x, y)

    ax.set_xlabel("Total hits per region")
    ax.set_ylabel("Frequency")

    output_path_png = os.path.join(plot_dir, "total_hit_distribution.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(plot_dir, "total_hit_distribution.svg")
    plt.savefig(output_path_svg, dpi=300)

    plt.close(fig)


def plot_peak_motif_indicator_heatmap(
    peak_hit_counts: Int[ndarray, "M M"], motif_names: List[str], output_dir: str
) -> None:
    """Plot co-occurrence heatmap showing motif associations across peaks.

    Creates a normalized correlation heatmap showing how frequently pairs of
    motifs co-occur within the same genomic peaks. Values are normalized by
    the geometric mean of individual motif frequencies.

    Parameters
    ----------
    peak_hit_counts : Int[ndarray, "M M"]
        Co-occurrence matrix where M = len(motif_names).
        Entry (i,j) represents the number of peaks containing both motif i and j.
        Diagonal entries represent total peaks containing each individual motif.
    motif_names : List[str]
        List of motif names for axis labels. Order must match matrix dimensions.
    output_dir : str
        Directory path where the heatmap plots will be saved.

    Notes
    -----
    Saves plots as:
    - motif_cooocurrence.png : High-resolution raster format
    - motif_cooocurrence.svg : Vector format

    The heatmap uses correlation normalization: matrix[i,j] / sqrt(matrix[i,i] * matrix[j,j])
    Colors use the 'Greens' colormap with values typically in [0, 1] range.
    """
    cov_norm = 1 / np.sqrt(np.diag(peak_hit_counts))
    matrix = peak_hit_counts * cov_norm[:, None] * cov_norm[None, :]
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    # Plot the heatmap
    cax = ax.imshow(matrix, interpolation="nearest", aspect="equal", cmap="Greens")

    # Set axes on heatmap
    ax.set_yticks(np.arange(len(motif_keys)))
    ax.set_yticklabels(motif_keys)
    ax.set_xticks(np.arange(len(motif_keys)))
    ax.set_xticklabels(motif_keys, rotation=90)
    ax.set_xlabel("Motif i")
    ax.set_ylabel("Motif j")

    ax.tick_params(axis="both", labelsize=8)

    cbar = fig.colorbar(cax, ax=ax, orientation="vertical", shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=8)

    output_path_png = os.path.join(output_dir, "motif_cooocurrence.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "motif_cooocurrence.svg")
    plt.savefig(output_path_svg, dpi=300)

    plt.close()


def plot_seqlet_confusion_heatmap(
    seqlet_confusion: Int[ndarray, "M M"], motif_names: List[str], output_dir: str
) -> None:
    """Plot confusion matrix heatmap comparing seqlets to hit calls.

    Creates a heatmap showing the overlap between TF-MoDISco seqlets and
    Fi-NeMo hit calls. Rows represent seqlet motifs, columns represent hit motifs.

    Parameters
    ----------
    seqlet_confusion : Int[ndarray, "M M"]
        Confusion matrix where M = len(motif_names).
        Entry (i,j) represents the number of seqlets of motif i that overlap
        with hits called for motif j.
    motif_names : List[str]
        List of motif names for axis labels. Order must match matrix dimensions.
    output_dir : str
        Directory path where the confusion matrix plots will be saved.

    Notes
    -----
    Saves plots as:
    - seqlet_confusion.png : High-resolution raster format
    - seqlet_confusion.svg : Vector format

    The heatmap uses 'Blues' colormap. Perfect agreement would show a diagonal
    pattern with high values along the diagonal and low off-diagonal values.
    """
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    # Plot the heatmap
    cax = ax.imshow(
        seqlet_confusion, interpolation="nearest", aspect="equal", cmap="Blues"
    )

    # Set axes on heatmap
    ax.set_yticks(np.arange(len(motif_keys)))
    ax.set_yticklabels(motif_keys)
    ax.set_xticks(np.arange(len(motif_keys)))
    ax.set_xticklabels(motif_keys, rotation=90)
    ax.set_xlabel("Hit motif")
    ax.set_ylabel("Seqlet motif")

    ax.tick_params(axis="both", labelsize=8)

    cbar = fig.colorbar(cax, ax=ax, orientation="vertical", shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=8)

    output_path_png = os.path.join(output_dir, "seqlet_confusion.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "seqlet_confusion.svg")
    plt.savefig(output_path_svg, dpi=300)

    plt.close()


class LogoGlyph(AbstractPathEffect):
    """Path effect for creating sequence logo glyphs with normalized dimensions.

    This class creates properly scaled and positioned text glyphs for sequence
    logos by normalizing character dimensions and applying appropriate transforms.

    Parameters
    ----------
    glyph : str
        Single character to render (e.g., 'A', 'C', 'G', 'T').
    ref_glyph : str, default 'E'
        Reference character used for width normalization.
    font_props : FontProperties, optional
        Font properties for the glyph rendering.
    offset : Tuple[float, float], default (0., 0.)
        Offset for glyph positioning.
    **kwargs
        Additional graphics collection parameters.
    """

    def __init__(
        self,
        glyph: str,
        ref_glyph: str = "E",
        font_props: Optional[FontProperties] = None,
        offset: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> None:
        super().__init__(offset)

        path_orig = TextPath((0, 0), glyph, size=1, prop=font_props)
        dims = path_orig.get_extents()
        ref_dims = TextPath((0, 0), ref_glyph, size=1, prop=font_props).get_extents()

        h_scale = 1 / dims.height
        ref_width = max(dims.width, ref_dims.width)
        w_scale = 1 / ref_width
        w_shift = (1 - dims.width / ref_width) / 2
        x_shift = -dims.x0
        y_shift = -dims.y0
        stretch = (
            Affine2D()
            .translate(tx=x_shift, ty=y_shift)
            .scale(sx=w_scale, sy=h_scale)
            .translate(tx=w_shift, ty=0)
        )

        self.path = stretch.transform_path(path_orig)

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace) -> Any:  # type: ignore[override]
        """Draw the glyph path using the renderer.

        Parameters
        ----------
        renderer : matplotlib renderer
            The renderer to draw with.
        gc : GraphicsContext
            Graphics context for drawing properties.
        tpath : Path
            Original text path (unused, using self.path instead).
        affine : Transform
            Affine transformation to apply.
        rgbFace : color
            Face color for the glyph.

        Returns
        -------
        Any
            Result from renderer.draw_path.
        """
        return renderer.draw_path(gc, self.path, affine, rgbFace)


def plot_logo(
    ax: Axes,
    heights: Float[ndarray, "B W"],
    glyphs: Iterable[str],
    colors: Optional[Mapping[str, Optional[str]]] = None,
    font_props: Optional[FontProperties] = None,
    shade_bounds: Optional[Tuple[int, int]] = None,
) -> None:
    """Plot sequence logo from contribution weight matrix.

    Creates a sequence logo visualization where letter heights represent
    the contribution or information content at each position. Supports
    both positive and negative contributions with proper stacking.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on.
    heights : Float[ndarray, "B W"]
        Height matrix where B = len(glyphs) and W = motif width.
        Entry (i,j) represents the height/contribution of base i at position j.
        Can contain both positive and negative values.
    glyphs : Iterable[str]
        Sequence of base characters corresponding to rows in heights matrix.
        Typically ['A', 'C', 'G', 'T'] for DNA.
    colors : Dict[str, str], optional
        Color mapping for each base. Keys should match glyphs.
        If None, all bases will use default matplotlib colors.
    font_props : FontProperties, optional
        Font properties for letter rendering. If None, uses default font.
    shade_bounds : Tuple[int, int], optional
        (start, end) position indices to shade in background.
        Useful for highlighting core motif regions.

    Notes
    -----
    Positive and negative contributions are handled separately:
    - Positive values are stacked above zero line in order of descending absolute value
    - Negative values are stacked below zero line in order of descending absolute value
    - A horizontal line is drawn at y=0 for reference

    The resulting plot has:
    - X-axis: Position in motif (0-indexed)
    - Y-axis: Contribution magnitude
    - Bar width: 0.95 (small gaps between positions)
    """
    if colors is None:
        colors = {g: None for g in glyphs}

    ax.margins(x=0, y=0)

    pos_values = np.clip(heights, 0, None)
    neg_values = np.clip(heights, None, 0)
    pos_order = np.argsort(pos_values, axis=0)
    neg_order = np.argsort(neg_values, axis=0)[::-1, :]
    pos_reorder = np.argsort(pos_order, axis=0)
    neg_reorder = np.argsort(neg_order, axis=0)
    pos_offsets = np.take_along_axis(
        np.cumsum(np.take_along_axis(pos_values, pos_order, axis=0), axis=0),
        pos_reorder,
        axis=0,
    )
    neg_offsets = np.take_along_axis(
        np.cumsum(np.take_along_axis(neg_values, neg_order, axis=0), axis=0),
        neg_reorder,
        axis=0,
    )
    bottoms = pos_offsets + neg_offsets - heights

    x = np.arange(heights.shape[1])

    for glyph, height, bottom in zip(glyphs, heights, bottoms):
        ax.bar(
            x,
            height,
            0.95,
            bottom=bottom,
            path_effects=[LogoGlyph(glyph, font_props=font_props)],
            color=colors[glyph],
        )

    if shade_bounds is not None:
        start, end = shade_bounds
        ax.axvspan(start - 0.5, end - 0.5, color="0.9", zorder=-1)

    ax.axhline(zorder=-1, linewidth=0.5, color="black")


LOGO_ALPHABET = "ACGT"
LOGO_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}
LOGO_FONT = FontProperties(weight="bold")


def plot_cwms(
    cwms: Dict[str, Dict[str, Float[ndarray, "4 W"]]],
    trim_bounds: Dict[str, Dict[str, Tuple[int, int]]],
    out_dir: str,
    alphabet: str = LOGO_ALPHABET,
    colors: Dict[str, str] = LOGO_COLORS,
    font: FontProperties = LOGO_FONT,
) -> None:
    """Plot contribution weight matrices as sequence logos.

    Creates sequence logo plots for all motifs and CWM types, with optional
    shading to highlight trimmed regions. Saves plots in both PNG and SVG formats.

    Parameters
    ----------
    cwms : Dict[str, Dict[str, Float[ndarray, "4 W"]]]
        Nested dictionary structure: {motif_name: {cwm_type: cwm_array}}.
        Each cwm_array has shape (4, W) where W is motif width.
        Rows correspond to bases in alphabet order.
    trim_bounds : Dict[str, Dict[str, Tuple[int, int]]]
        Nested dictionary: {motif_name: {cwm_type: (start, end)}}.
        Defines regions to shade in the sequence logos.
    out_dir : str
        Output directory where motif subdirectories will be created.
    alphabet : str, default LOGO_ALPHABET
        DNA alphabet string, typically 'ACGT'.
    colors : Dict[str, str], default LOGO_COLORS
        Color mapping for DNA bases. Keys should match alphabet characters.
    font : FontProperties, default LOGO_FONT
        Font properties for sequence logo rendering.

    Notes
    -----
    Directory structure created:
    ```
    out_dir/
    ├── motif1/
    │   ├── cwm_type1.png
    │   ├── cwm_type1.svg
    │   └── ...
    └── motif2/
        └── ...
    ```

    Each plot is 10x2 inches with trimmed regions shaded if specified.
    Spines (plot borders) are hidden for cleaner appearance.
    """
    for m, v in cwms.items():
        motif_dir = os.path.join(out_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            fig, ax = plt.subplots(figsize=(10, 2))

            plot_logo(
                ax,
                cwm,
                alphabet,
                colors=colors,
                font_props=font,
                shade_bounds=trim_bounds[m][cwm_type],
            )

            for name, spine in ax.spines.items():
                spine.set_visible(False)

            output_path_png = os.path.join(motif_dir, f"{cwm_type}.png")
            plt.savefig(output_path_png, dpi=100)
            output_path_svg = os.path.join(motif_dir, f"{cwm_type}.svg")
            plt.savefig(output_path_svg)

            plt.close(fig)


def plot_hit_vs_seqlet_counts(
    recall_data: Dict[str, Dict[str, Union[int, float]]], output_dir: str
) -> None:
    """Plot scatter plot comparing hit counts to seqlet counts per motif.

    Creates a log-log scatter plot showing the relationship between the number
    of hits called by Fi-NeMo and the number of seqlets identified by TF-MoDISco
    for each motif. Includes diagonal reference line and motif annotations.

    Parameters
    ----------
    recall_data : Dict[str, Dict[str, Union[int, float]]]
        Dictionary with motif names as keys and metrics dictionaries as values.
        Each metrics dictionary must contain:
        - 'num_hits_total' : int, total number of hits for the motif
        - 'num_seqlets' : int, total number of seqlets for the motif
    output_dir : str
        Directory path where the scatter plot will be saved.

    Notes
    -----
    Saves plots as:
    - hit_vs_seqlet_counts.png : High-resolution raster format
    - hit_vs_seqlet_counts.svg : Vector format

    Plot features:
    - Log-log scale on both axes
    - Diagonal reference line (y = x) as dashed line
    - Points annotated with abbreviated motif names
    """
    x = []
    y = []
    m = []
    for k, v in recall_data.items():
        x.append(v["num_hits_total"])
        y.append(v["num_seqlets"])
        m.append(k)

    lim = max(np.amax(x), np.amax(y))

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    ax.axline((0, 0), (lim, lim), color="0.3", linewidth=0.7, linestyle=(0, (5, 5)))
    ax.scatter(x, y, s=5)
    for i, txt in enumerate(m):
        short = abbreviate_motif_name(txt)
        ax.annotate(short, (x[i], y[i]), fontsize=8, weight="bold")

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_xlabel("Hits per motif")
    ax.set_ylabel("Seqlets per motif")

    output_path_png = os.path.join(output_dir, "hit_vs_seqlet_counts.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "hit_vs_seqlet_counts.svg")
    plt.savefig(output_path_svg)

    plt.close()


def write_report(
    report_df: pl.DataFrame,
    motif_names: List[str],
    out_path: str,
    compute_recall: bool,
    use_seqlets: bool,
) -> None:
    """Generate and write HTML report from motif analysis results.

    Creates a comprehensive HTML report with tables and visualizations
    summarizing the Fi-NeMo motif discovery and hit calling results.

    Parameters
    ----------
    report_df : pl.DataFrame
        DataFrame containing motif statistics and performance metrics.
        Expected columns depend on compute_recall and use_seqlets flags.
    motif_names : List[str]
        List of motif names to include in the report.
        Order determines presentation sequence in the report.
    out_path : str
        File path where the HTML report will be written.
        Parent directory must exist.
    compute_recall : bool
        Whether recall metrics were computed and should be included
        in the report template.
    use_seqlets : bool
        Whether TF-MoDISco seqlet data was used in the analysis
        and should be referenced in the report.

    Notes
    -----
    Uses Jinja2 templating with the report.html template from the
    templates package. The template receives:
    - report_data: Iterator of DataFrame rows as named tuples
    - motif_names: List of motif names
    - compute_recall: Boolean flag for recall metrics
    - use_seqlets: Boolean flag for seqlet usage

    Raises
    ------
    OSError
        If the output path cannot be written.
    """
    template_str = (
        importlib.resources.files(templates).joinpath("report.html").read_text()
    )
    template = Template(template_str)
    report = template.render(
        report_data=report_df.iter_rows(named=True),
        motif_names=motif_names,
        compute_recall=compute_recall,
        use_seqlets=use_seqlets,
    )
    with open(out_path, "w") as f:
        f.write(report)
