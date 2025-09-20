"""Main CLI module for the Fi-NeMo motif instance calling pipeline.

This module provides the command-line interface for all Fi-NeMo operations:
- Data preprocessing from various genomic formats
- Motif hit calling using the Fi-NeMo algorithm
- Report generation and result visualization
- Post-processing operations (hit collapsing, intersection)

The CLI supports multiple input formats including bigWig, HDF5 (ChromBPNet/BPNet),
and TF-MoDISco format.
"""

from . import data_io

import os
import argparse
import warnings
import inspect
from typing import Optional, List

import polars as pl


def extract_regions_bw(
    peaks_path: str,
    chrom_order_path: Optional[str],
    fa_path: str,
    bw_paths: List[str],
    out_path: str,
    region_width: int = 1000,
) -> None:
    """Extract genomic regions and contribution scores from bigWig and FASTA files.

    Parameters
    ----------
    peaks_path : str
        Path to ENCODE NarrowPeak format file.
    chrom_order_path : str, optional
        Path to chromosome ordering file.
    fa_path : str
        Path to genome FASTA file.
    bw_paths : List[str]
        List of bigWig file paths containing contribution scores.
    out_path : str
        Output path for NPZ file.
    region_width : int, default 1000
        Width of regions to extract around peak summits.

    Notes
    -----
    BigWig files only provide projected contribution scores.
    """
    half_width = region_width // 2

    # Load peak regions and extract sequences/contributions
    peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    sequences, contribs = data_io.load_regions_from_bw(
        peaks_df, fa_path, bw_paths, half_width
    )

    # Save processed data to NPZ format
    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_chrombpnet_h5(
    peaks_path: Optional[str],
    chrom_order_path: Optional[str],
    h5_paths: List[str],
    out_path: str,
    region_width: int = 1000,
) -> None:
    """Extract genomic regions and contribution scores from ChromBPNet HDF5 files.

    Parameters
    ----------
    peaks_path : str, optional
        Path to ENCODE NarrowPeak format file. If None, lacks absolute coordinates.
    chrom_order_path : str, optional
        Path to chromosome ordering file.
    h5_paths : List[str]
        List of ChromBPNet HDF5 file paths.
    out_path : str
        Output path for NPZ file.
    region_width : int, default 1000
        Width of regions to extract around peak summits.
    """
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_chrombpnet_h5(h5_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_bpnet_h5(
    peaks_path: Optional[str],
    chrom_order_path: Optional[str],
    h5_paths: List[str],
    out_path: str,
    region_width: int = 1000,
) -> None:
    """Extract genomic regions and contribution scores from BPNet HDF5 files.

    Parameters
    ----------
    peaks_path : str, optional
        Path to ENCODE NarrowPeak format file. If None, output lacks absolute coordinates.
    chrom_order_path : str, optional
        Path to chromosome ordering file.
    h5_paths : List[str]
        List of BPNet HDF5 file paths.
    out_path : str
        Output path for NPZ file.
    region_width : int, default 1000
        Width of regions to extract around peak summits.
    """
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_bpnet_h5(h5_paths, half_width)

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def extract_regions_modisco_fmt(
    peaks_path: Optional[str],
    chrom_order_path: Optional[str],
    shaps_paths: List[str],
    ohe_path: str,
    out_path: str,
    region_width: int = 1000,
) -> None:
    """Extract genomic regions and contribution scores from TF-MoDISco format files.

    Parameters
    ----------
    peaks_path : str, optional
        Path to ENCODE NarrowPeak format file. If None, output lacks absolute coordinates.
    chrom_order_path : str, optional
        Path to chromosome ordering file.
    shaps_paths : List[str]
        List of paths to .npy/.npz files containing SHAP/attribution scores.
    ohe_path : str
        Path to .npy/.npz file containing one-hot encoded sequences.
    out_path : str
        Output path for NPZ file.
    region_width : int, default 1000
        Width of regions to extract around peak summits.
    """
    half_width = region_width // 2

    if peaks_path is not None:
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
    else:
        peaks_df = None

    sequences, contribs = data_io.load_regions_from_modisco_fmt(
        shaps_paths, ohe_path, half_width
    )

    data_io.write_regions_npz(sequences, contribs, out_path, peaks_df=peaks_df)


def call_hits(
    regions_path: str,
    peaks_path: Optional[str],
    modisco_h5_path: str,
    chrom_order_path: Optional[str],
    motifs_include_path: Optional[str],
    motif_names_path: Optional[str],
    motif_lambdas_path: Optional[str],
    out_dir: str,
    cwm_trim_coords_path: Optional[str] = None,
    cwm_trim_thresholds_path: Optional[str] = None,
    cwm_trim_threshold_default: float = 0.3,
    lambda_default: float = 0.7,
    step_size_max: float = 3.0,
    step_size_min: float = 0.08,
    sqrt_transform: bool = False,
    convergence_tol: float = 0.0005,
    max_steps: int = 10000,
    batch_size: int = 2000,
    step_adjust: float = 0.7,
    device: Optional[str] = None,
    mode: str = "pp",
    no_post_filter: bool = False,
    compile_optimizer: bool = False,
) -> None:
    """Call motif hits using the Fi-NeMo algorithm on preprocessed genomic regions.

    This function implements the core Fi-NeMo hit calling pipeline, which identifies
    motif instances by solving a sparse reconstruction problem using proximal gradient
    descent. The algorithm represents contribution scores as weighted combinations of
    motif CWMs at specific positions.

    Parameters
    ----------
    regions_path : str
        Path to NPZ file containing preprocessed regions (sequences, contributions,
        and optional peak coordinates).
    peaks_path : str, optional
        DEPRECATED. Path to ENCODE NarrowPeak format file. Peak data should be
        included during preprocessing instead.
    modisco_h5_path : str
        Path to TF-MoDISco H5 file containing motif CWMs.
    chrom_order_path : str, optional
        DEPRECATED. Path to chromosome ordering file.
    motifs_include_path : str, optional
        Path to file listing motif names to include in analysis.
    motif_names_path : str, optional
        Path to file mapping motif IDs to custom names.
    motif_lambdas_path : str, optional
        Path to file specifying per-motif lambda values.
    out_dir : str
        Output directory for results.
    cwm_trim_coords_path : str, optional
        Path to file specifying custom motif trimming coordinates.
    cwm_trim_thresholds_path : str, optional
        Path to file specifying custom motif trimming thresholds.
    cwm_trim_threshold_default : float, default 0.3
        Default threshold for motif trimming.
    lambda_default : float, default 0.7
        Default L1 regularization weight.
    step_size_max : float, default 3.0
        Maximum optimization step size.
    step_size_min : float, default 0.08
        Minimum optimization step size.
    sqrt_transform : bool, default False
        Whether to apply signed square root transform to contributions.
    convergence_tol : float, default 0.0005
        Convergence tolerance for duality gap.
    max_steps : int, default 10000
        Maximum number of optimization steps.
    batch_size : int, default 2000
        Batch size for GPU processing.
    step_adjust : float, default 0.7
        Step size adjustment factor on divergence.
    device : str, optional
        DEPRECATED. Use CUDA_VISIBLE_DEVICES environment variable instead.
    mode : str, default "pp"
        Contribution type mode ('pp', 'ph', 'hp', 'hh') where 'p'=projected, 'h'=hypothetical.
    no_post_filter : bool, default False
        If True, skip post-hit-calling similarity filtering.
    compile_optimizer : bool, default False
        Whether to JIT-compile the optimizer for speed.

    Notes
    -----
    The Fi-NeMo algorithm solves the optimization problem:
    minimize_c: ||contribs - reconstruction(c)||²₂ + λ||c||₁
    subject to: c ≥ 0

    where c represents motif hit coefficients and reconstruction uses convolution
    with motif CWMs.
    """

    params = locals()
    import torch
    from . import hitcaller

    if device is not None:
        warnings.warn(
            "The `--device` flag is deprecated and will be removed in a future version. Please use the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU device."
        )

    sequences, contribs, peaks_df, has_peaks = data_io.load_regions_npz(regions_path)

    region_width = sequences.shape[2]
    if region_width % 2 != 0:
        raise ValueError(f"Region width of {region_width} is not divisible by 2.")

    half_width = region_width // 2
    num_regions = contribs.shape[0]

    if peaks_path is not None:
        warnings.warn(
            "Providing a peaks file to `call-hits` is deprecated, and this option will be removed in a future version. Peaks should instead be provided in the preprocessing step to be included in `regions.npz`."
        )
        peaks_df = data_io.load_peaks(peaks_path, chrom_order_path, half_width)
        has_peaks = True

    if not has_peaks:
        warnings.warn(
            "No peak region data provided. Output hits will lack absolute genomic coordinates."
        )

    if mode == "pp":
        motif_type = "cwm"
        use_hypothetical_contribs = False
    elif mode == "ph":
        motif_type = "cwm"
        use_hypothetical_contribs = True
    elif mode == "hp":
        motif_type = "hcwm"
        use_hypothetical_contribs = False
    elif mode == "hh":
        motif_type = "hcwm"
        use_hypothetical_contribs = True
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'pp', 'ph', 'hp', 'hh'."
        )

    if motifs_include_path is not None:
        motifs_include = data_io.load_txt(motifs_include_path)
    else:
        motifs_include = None

    if motif_names_path is not None:
        motif_name_map = data_io.load_mapping(motif_names_path, str)
    else:
        motif_name_map = None

    if motif_lambdas_path is not None:
        motif_lambdas = data_io.load_mapping(motif_lambdas_path, float)
    else:
        motif_lambdas = None

    if cwm_trim_coords_path is not None:
        trim_coords = data_io.load_mapping_tuple(cwm_trim_coords_path, int)
    else:
        trim_coords = None

    if cwm_trim_thresholds_path is not None:
        trim_thresholds = data_io.load_mapping(cwm_trim_thresholds_path, float)
    else:
        trim_thresholds = None

    motifs_df, cwms, trim_masks, _ = data_io.load_modisco_motifs(
        modisco_h5_path,
        trim_coords,
        trim_thresholds,
        cwm_trim_threshold_default,
        motif_type,
        motifs_include,
        motif_name_map,
        motif_lambdas,
        lambda_default,
        True,
    )
    num_motifs = cwms.shape[0]
    motif_width = cwms.shape[2]
    lambdas = motifs_df.get_column("lambda").to_numpy(writable=True)

    device_obj = torch.device(device) if device is not None else None
    hits_df, qc_df = hitcaller.fit_contribs(
        cwms,
        contribs,
        sequences,
        trim_masks,
        use_hypothetical_contribs,
        lambdas,
        step_size_max,
        step_size_min,
        sqrt_transform,
        convergence_tol,
        max_steps,
        batch_size,
        step_adjust,
        not no_post_filter,
        device_obj,
        compile_optimizer,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path_qc = os.path.join(out_dir, "peaks_qc.tsv")
    data_io.write_hits(hits_df, peaks_df, motifs_df, qc_df, out_dir, motif_width)
    data_io.write_qc(qc_df, peaks_df, out_path_qc)

    out_path_motif_df = os.path.join(out_dir, "motif_data.tsv")
    data_io.write_motifs_df(motifs_df, out_path_motif_df)

    out_path_motif_cwms = os.path.join(out_dir, "motif_cwms.npy")
    data_io.write_motif_cwms(cwms, out_path_motif_cwms)

    params |= {
        "region_width": region_width,
        "num_regions": num_regions,
        "untrimmed_motif_width": motif_width,
        "num_motifs": num_motifs,
    }

    out_path_params = os.path.join(out_dir, "parameters.json")
    data_io.write_params(params, out_path_params)


def report(
    regions_path: str,
    hits_dir: str,
    modisco_h5_path: Optional[str],
    peaks_path: Optional[str],
    motifs_include_path: Optional[str],
    motif_names_path: Optional[str],
    out_dir: str,
    modisco_region_width: int = 400,
    cwm_trim_threshold: float = 0.3,
    compute_recall: bool = True,
    use_seqlets: bool = True,
) -> None:
    """Generate comprehensive HTML report with statistics and visualizations.

    This function creates detailed analysis reports comparing Fi-NeMo hit calling
    results with TF-MoDISco seqlets, including performance metrics, distribution
    plots, and motif visualization. The report provides insights into hit calling
    quality and motif discovery accuracy.

    Parameters
    ----------
    regions_path : str
        Path to NPZ file containing the same regions used for hit calling.
    hits_dir : str
        Path to directory containing Fi-NeMo hit calling outputs.
    modisco_h5_path : str, optional
        Path to TF-MoDISco H5 file. If None, seqlet comparisons are skipped.
    peaks_path : str, optional
        DEPRECATED. Peak coordinates should be included in regions file.
    motifs_include_path : str, optional
        DEPRECATED. This information is inferred from hit calling outputs.
    motif_names_path : str, optional
        DEPRECATED. This information is inferred from hit calling outputs.
    out_dir : str
        Output directory for report files.
    modisco_region_width : int, default 400
        Width of regions used by TF-MoDISco (needed for coordinate conversion).
    cwm_trim_threshold : float, default 0.3
        DEPRECATED. This information is inferred from hit calling outputs.
    compute_recall : bool, default True
        Whether to compute recall metrics against TF-MoDISco seqlets.
    use_seqlets : bool, default True
        Whether to include seqlet-based comparisons in the report.

    Notes
    -----
    The generated report includes:
    - Hit vs seqlet count comparisons
    - Motif CWM visualizations
    - Hit statistic distributions
    - Co-occurrence heatmaps
    - Confusion matrices for overlapping motifs
    """
    from . import evaluation, visualization

    sequences, contribs, peaks_df, _ = data_io.load_regions_npz(regions_path)
    if len(contribs.shape) == 3:
        regions = contribs * sequences
    elif len(contribs.shape) == 2:
        regions = contribs[:, None, :] * sequences
    else:
        raise ValueError(f"Unexpected contribs shape: {contribs.shape}")

    half_width = regions.shape[2] // 2
    modisco_half_width = modisco_region_width // 2

    if peaks_path is not None:
        warnings.warn(
            "Providing a peaks file to `report` is deprecated, and this option will be removed in a future version. Peaks should instead be provided in the preprocessing step to be included in `regions.npz`."
        )
        peaks_df = data_io.load_peaks(peaks_path, None, half_width)

    if hits_dir.endswith(".tsv"):
        warnings.warn(
            "Passing a hits.tsv file to `finemo report` is deprecated. Please provide the directory containing the hits.tsv file instead."
        )

        hits_path = hits_dir

        hits_df = data_io.load_hits(hits_path, lazy=True)

        if motifs_include_path is not None:
            motifs_include = data_io.load_txt(motifs_include_path)
        else:
            motifs_include = None

        if motif_names_path is not None:
            motif_name_map = data_io.load_mapping(motif_names_path, str)
        else:
            motif_name_map = None

        if modisco_h5_path is not None:
            motifs_df, cwms_modisco, _, motif_names = data_io.load_modisco_motifs(
                modisco_h5_path,
                None,
                None,
                cwm_trim_threshold,
                "cwm",
                motifs_include,
                motif_name_map,
                None,
                1.0,
                True,
            )
        else:
            # When no modisco_h5_path is provided in legacy TSV mode, we can't compute motifs
            # This will cause an error later, but that's expected behavior
            raise ValueError(
                "modisco_h5_path is required when providing a hits.tsv file directly"
            )

    else:
        hits_df_path = os.path.join(hits_dir, "hits.tsv")
        hits_df = data_io.load_hits(hits_df_path, lazy=True)

        motifs_df_path = os.path.join(hits_dir, "motif_data.tsv")
        motifs_df, motif_names = data_io.load_motifs_df(motifs_df_path)

        cwms_modisco_path = os.path.join(hits_dir, "motif_cwms.npy")
        cwms_modisco = data_io.load_motif_cwms(cwms_modisco_path)

        params_path = os.path.join(hits_dir, "parameters.json")
        params = data_io.load_params(params_path)
        cwm_trim_threshold = params["cwm_trim_threshold_default"]

    if not use_seqlets:
        warnings.warn(
            "Usage of the `--no-seqlets` flag is deprecated and will be removed in a future version. Please omit the `--modisco-h5` argument instead."
        )
        seqlets_df = None
    elif modisco_h5_path is None:
        compute_recall = False
        seqlets_df = None
    else:
        seqlets_df = data_io.load_modisco_seqlets(
            modisco_h5_path,
            peaks_df,
            motifs_df,
            half_width,
            modisco_half_width,
            lazy=True,
        )

    motif_width = cwms_modisco.shape[2]

    # Convert to LazyFrame if needed and ensure motif_names is a list
    if isinstance(hits_df, pl.LazyFrame):
        hits_df_lazy: pl.LazyFrame = hits_df
    else:
        hits_df_lazy: pl.LazyFrame = hits_df.lazy()

    motif_names_list: List[str] = list(motif_names)

    occ_df, coooc = evaluation.get_motif_occurences(hits_df_lazy, motif_names_list)

    report_data, report_df, cwms, trim_bounds = evaluation.tfmodisco_comparison(
        regions,
        hits_df,
        peaks_df,
        seqlets_df,
        motifs_df,
        cwms_modisco,
        motif_names_list,
        modisco_half_width,
        motif_width,
        compute_recall,
    )

    if seqlets_df is not None:
        confusion_df, confusion_mat = evaluation.seqlet_confusion(
            hits_df, seqlets_df, peaks_df, motif_names_list, motif_width
        )
    else:
        confusion_df, confusion_mat = None, None

    os.makedirs(out_dir, exist_ok=True)

    occ_path = os.path.join(out_dir, "motif_occurrences.tsv")
    data_io.write_occ_df(occ_df, occ_path)

    data_io.write_report_data(report_df, cwms, out_dir)

    visualization.plot_hit_stat_distributions(hits_df_lazy, motif_names_list, out_dir)
    visualization.plot_hit_peak_distributions(occ_df, motif_names_list, out_dir)
    visualization.plot_peak_motif_indicator_heatmap(coooc, motif_names_list, out_dir)

    plot_dir = os.path.join(out_dir, "CWMs")
    visualization.plot_cwms(cwms, trim_bounds, plot_dir)

    if seqlets_df is not None:
        seqlets_collected = (
            seqlets_df.collect() if isinstance(seqlets_df, pl.LazyFrame) else seqlets_df
        )
        seqlets_path = os.path.join(out_dir, "seqlets.tsv")
        data_io.write_modisco_seqlets(seqlets_collected, seqlets_path)

        if confusion_df is not None and confusion_mat is not None:
            seqlet_confusion_path = os.path.join(out_dir, "seqlet_confusion.tsv")
            data_io.write_seqlet_confusion_df(confusion_df, seqlet_confusion_path)

            visualization.plot_hit_vs_seqlet_counts(report_data, out_dir)
            visualization.plot_seqlet_confusion_heatmap(
                confusion_mat, motif_names_list, out_dir
            )

    report_path = os.path.join(out_dir, "report.html")
    visualization.write_report(
        report_df, motif_names_list, report_path, compute_recall, seqlets_df is not None
    )


def collapse_hits(hits_path: str, out_path: str, overlap_frac: float = 0.2) -> None:
    """Collapse overlapping hits by selecting the best hit per overlapping group.

    This function processes a set of motif hits and identifies overlapping hits,
    keeping only the hit with the highest similarity score within each overlapping
    group. This reduces redundancy in hit calls while preserving the most confident
    predictions.

    Parameters
    ----------
    hits_path : str
        Path to input TSV file containing hit data (hits.tsv or hits_unique.tsv).
    out_path : str
        Path to output TSV file with additional 'is_primary' column.
    overlap_frac : float, default 0.2
        Minimum fractional overlap for considering hits as overlapping.
        For hits of lengths x and y, minimum overlap = overlap_frac * (x + y) / 2.

    Notes
    -----
    The algorithm uses a sweep line approach with a heap data structure to
    efficiently identify overlapping intervals and select the best hit based
    on similarity scores.
    """
    from . import postprocessing

    hits_df = data_io.load_hits(hits_path, lazy=False)
    hits_collapsed_df = postprocessing.collapse_hits(hits_df, overlap_frac)

    data_io.write_hits_processed(
        hits_collapsed_df, out_path, schema=data_io.HITS_COLLAPSED_DTYPES
    )


def intersect_hits(hits_paths: List[str], out_path: str, relaxed: bool = False) -> None:
    """Find intersection of hits across multiple Fi-NeMo runs.

    This function identifies motif hits that are consistently called across
    multiple independent runs, providing a way to assess reproducibility and
    identify high-confidence hits.

    Parameters
    ----------
    hits_paths : List[str]
        List of paths to input TSV files from different runs.
    out_path : str
        Path to output TSV file containing intersection results.
        Duplicate columns are suffixed with run index.
    relaxed : bool, default False
        If True, uses relaxed intersection criteria based only on motif names
        and untrimmed coordinates. If False, assumes consistent region definitions
        and motif trimming across runs.

    Notes
    -----
    The strict intersection mode requires consistent input regions and motif
    processing parameters across all runs. The relaxed mode is more permissive
    but may not be suitable when genomic coordinates are unavailable.
    """
    from . import postprocessing

    hits_dfs = [data_io.load_hits(hits_path, lazy=False) for hits_path in hits_paths]
    hits_df = postprocessing.intersect_hits(hits_dfs, relaxed)

    data_io.write_hits_processed(hits_df, out_path, schema=None)


def cli() -> None:
    """Command-line interface for the Fi-NeMo motif instance calling pipeline.

    This function provides the main entry point for all Fi-NeMo operations including:
    - Data preprocessing from various formats (bigWig, HDF5, TF-MoDISco)
    - Motif hit calling using the Fi-NeMo algorithm
    - Report generation and visualization
    - Post-processing operations (hit collapsing, intersection)
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="cmd")

    extract_regions_bw_parser = subparsers.add_parser(
        "extract-regions-bw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract sequences and contributions from FASTA and bigwig files.",
    )

    extract_regions_bw_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        required=True,
        help="A peak regions file in ENCODE NarrowPeak format.",
    )
    extract_regions_bw_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.",
    )
    extract_regions_bw_parser.add_argument(
        "-f",
        "--fasta",
        type=str,
        required=True,
        help="A genome FASTA file. If an .fai index file doesn't exist in the same directory, it will be created.",
    )
    extract_regions_bw_parser.add_argument(
        "-b",
        "--bigwigs",
        type=str,
        required=True,
        nargs="+",
        help="One or more bigwig files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.",
    )

    extract_regions_bw_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .npz file.",
    )

    extract_regions_bw_parser.add_argument(
        "-w",
        "--region-width",
        type=int,
        default=inspect.signature(extract_regions_bw)
        .parameters["region_width"]
        .default,
        help="The width of the input region centered around each peak summit.",
    )

    extract_chrombpnet_regions_h5_parser = subparsers.add_parser(
        "extract-regions-chrombpnet-h5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract sequences and contributions from ChromBPNet contributions H5 files.",
    )

    extract_chrombpnet_regions_h5_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.",
    )
    extract_chrombpnet_regions_h5_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.",
    )

    extract_chrombpnet_regions_h5_parser.add_argument(
        "-c",
        "--h5s",
        type=str,
        required=True,
        nargs="+",
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.",
    )

    extract_chrombpnet_regions_h5_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .npz file.",
    )

    extract_chrombpnet_regions_h5_parser.add_argument(
        "-w",
        "--region-width",
        type=int,
        default=inspect.signature(extract_regions_chrombpnet_h5)
        .parameters["region_width"]
        .default,
        help="The width of the input region centered around each peak summit.",
    )

    extract_regions_h5_parser = subparsers.add_parser(
        "extract-regions-h5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract sequences and contributions from ChromBPNet contributions H5 files. DEPRECATED: Use `extract-regions-chrombpnet-h5` instead.",
    )

    extract_regions_h5_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.",
    )
    extract_regions_h5_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.",
    )

    extract_regions_h5_parser.add_argument(
        "-c",
        "--h5s",
        type=str,
        required=True,
        nargs="+",
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.",
    )

    extract_regions_h5_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .npz file.",
    )

    extract_regions_h5_parser.add_argument(
        "-w",
        "--region-width",
        type=int,
        default=inspect.signature(extract_regions_chrombpnet_h5)
        .parameters["region_width"]
        .default,
        help="The width of the input region centered around each peak summit.",
    )

    extract_bpnet_regions_h5_parser = subparsers.add_parser(
        "extract-regions-bpnet-h5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract sequences and contributions from BPNet contributions H5 files.",
    )

    extract_bpnet_regions_h5_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.",
    )
    extract_bpnet_regions_h5_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.",
    )

    extract_bpnet_regions_h5_parser.add_argument(
        "-c",
        "--h5s",
        type=str,
        required=True,
        nargs="+",
        help="One or more H5 files of contribution scores, with paths delimited by whitespace. Scores are averaged across files.",
    )

    extract_bpnet_regions_h5_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .npz file.",
    )

    extract_bpnet_regions_h5_parser.add_argument(
        "-w",
        "--region-width",
        type=int,
        default=inspect.signature(extract_regions_bpnet_h5)
        .parameters["region_width"]
        .default,
        help="The width of the input region centered around each peak summit.",
    )

    extract_regions_modisco_fmt_parser = subparsers.add_parser(
        "extract-regions-modisco-fmt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract sequences and contributions from tfmodisco-lite input files.",
    )

    extract_regions_modisco_fmt_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="A peak regions file in ENCODE NarrowPeak format. If omitted, downstream outputs will lack absolute genomic coordinates.",
    )
    extract_regions_modisco_fmt_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="A tab-delimited file with chromosome names in the first column to define sort order of chromosomes. Missing chromosomes are ordered as they appear in -p/--peaks.",
    )

    extract_regions_modisco_fmt_parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        required=True,
        help="A .npy or .npz file containing one-hot encoded sequences.",
    )

    extract_regions_modisco_fmt_parser.add_argument(
        "-a",
        "--attributions",
        type=str,
        required=True,
        nargs="+",
        help="One or more .npy or .npz files of hypothetical contribution scores, with paths delimited by whitespace. Scores are averaged across files.",
    )

    extract_regions_modisco_fmt_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .npz file.",
    )

    extract_regions_modisco_fmt_parser.add_argument(
        "-w",
        "--region-width",
        type=int,
        default=inspect.signature(extract_regions_modisco_fmt)
        .parameters["region_width"]
        .default,
        help="The width of the input region centered around each peak summit.",
    )

    call_hits_parser = subparsers.add_parser(
        "call-hits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Call hits on provided sequences, contributions, and motif CWM's.",
    )

    call_hits_parser.add_argument(
        "-M",
        "--mode",
        type=str,
        default=inspect.signature(call_hits).parameters["mode"].default,
        choices={"pp", "ph", "hp", "hh"},
        help="The type of attributions to use for CWM's and input contribution scores, respectively. 'h' for hypothetical and 'p' for projected.",
    )

    call_hits_parser.add_argument(
        "-r",
        "--regions",
        type=str,
        required=True,
        help="A .npz file of input sequences, contributions, and coordinates. Can be generated using `finemo extract-regions-*` subcommands.",
    )
    call_hits_parser.add_argument(
        "-m",
        "--modisco-h5",
        type=str,
        required=True,
        help="A tfmodisco-lite output H5 file of motif patterns.",
    )

    call_hits_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.",
    )
    call_hits_parser.add_argument(
        "-C",
        "--chrom-order",
        type=str,
        default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.",
    )

    call_hits_parser.add_argument(
        "-I",
        "--motifs-include",
        type=str,
        default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column to include in hit calling. If omitted, all motifs in the modisco H5 file are used.",
    )
    call_hits_parser.add_argument(
        "-N",
        "--motif-names",
        type=str,
        default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom names in the second column. Omitted motifs default to tfmodisco names.",
    )

    call_hits_parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="The path to the output directory.",
    )

    call_hits_parser.add_argument(
        "-t",
        "--cwm-trim-threshold",
        type=float,
        default=inspect.signature(call_hits)
        .parameters["cwm_trim_threshold_default"]
        .default,
        help="The default threshold to determine motif start and end positions within the full CWMs.",
    )
    call_hits_parser.add_argument(
        "-T",
        "--cwm-trim-thresholds",
        type=str,
        default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom trim thresholds in the second column. Omitted motifs default to the `--cwm-trim-threshold` value.",
    )
    call_hits_parser.add_argument(
        "-R",
        "--cwm-trim-coords",
        type=str,
        default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and custom trim start and end coordinates in the second and third columns, respectively. Omitted motifs default to `--cwm-trim-thresholds` values.",
    )

    call_hits_parser.add_argument(
        "-l",
        "--global-lambda",
        type=float,
        default=inspect.signature(call_hits).parameters["lambda_default"].default,
        help="The default L1 regularization weight determining the sparsity of hits.",
    )
    call_hits_parser.add_argument(
        "-L",
        "--motif-lambdas",
        type=str,
        default=None,
        help="A tab-delimited file with tfmodisco motif names (e.g pos_patterns.pattern_0) in the first column and motif-specific lambdas in the second column. Omitted motifs default to the `--global-lambda` value.",
    )
    call_hits_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=None,
        help="DEPRECATED: Please use the `--lambda` argument instead.",
    )
    call_hits_parser.add_argument(
        "-A",
        "--motif-alphas",
        type=str,
        default=None,
        help="DEPRECATED: Please use the `--motif-lambdas` argument instead.",
    )

    call_hits_parser.add_argument(
        "-f",
        "--no-post-filter",
        action="store_true",
        help="Do not perform post-hit-calling filtering. By default, hits are filtered based on a minimum cosine similarity of `lambda` with the input contributions.",
    )
    call_hits_parser.add_argument(
        "-q",
        "--sqrt-transform",
        action="store_true",
        help="Apply a signed square root transform to the input contributions and CWMs before hit calling.",
    )
    call_hits_parser.add_argument(
        "-s",
        "--step-size-max",
        type=float,
        default=inspect.signature(call_hits).parameters["step_size_max"].default,
        help="The maximum optimizer step size.",
    )
    call_hits_parser.add_argument(
        "-i",
        "--step-size-min",
        type=float,
        default=inspect.signature(call_hits).parameters["step_size_min"].default,
        help="The minimum optimizer step size.",
    )
    call_hits_parser.add_argument(
        "-j",
        "--step-adjust",
        type=float,
        default=inspect.signature(call_hits).parameters["step_adjust"].default,
        help="The optimizer step size adjustment factor. If the optimizer diverges, the step size is multiplicatively adjusted by this factor",
    )
    call_hits_parser.add_argument(
        "-c",
        "--convergence-tol",
        type=float,
        default=inspect.signature(call_hits).parameters["convergence_tol"].default,
        help="The tolerance for determining convergence. The optimizer exits when the duality gap is less than the tolerance.",
    )
    call_hits_parser.add_argument(
        "-S",
        "--max-steps",
        type=int,
        default=inspect.signature(call_hits).parameters["max_steps"].default,
        help="The maximum number of optimization steps.",
    )
    call_hits_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=inspect.signature(call_hits).parameters["batch_size"].default,
        help="The batch size used for optimization.",
    )
    call_hits_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="DEPRECATED: Please use the `CUDA_VISIBLE_DEVICES` environment variable to specify the GPU device.",
    )
    call_hits_parser.add_argument(
        "-J",
        "--compile",
        action="store_true",
        help="JIT-compile the optimizer for faster performance. This may not be supported on older GPUs.",
    )

    report_parser = subparsers.add_parser(
        "report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Generate statistics and visualizations from hits and tfmodisco-lite motif data.",
    )

    report_parser.add_argument(
        "-r",
        "--regions",
        type=str,
        required=True,
        help="A .npz file containing input sequences, contributions, and coordinates. Must be the same as that used for `finemo call-hits`.",
    )
    report_parser.add_argument(
        "-H",
        "--hits",
        type=str,
        required=True,
        help="The output directory generated by the `finemo call-hits` command on the regions specified in `--regions`.",
    )
    report_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        default=None,
        help="DEPRECATED: Please provide this file to a preprocessing `finemo extract-regions-*` subcommand instead.",
    )
    report_parser.add_argument(
        "-m",
        "--modisco-h5",
        type=str,
        default=None,
        help="The tfmodisco-lite output H5 file of motif patterns. Must be the same as that used for hit calling unless `--no-recall` is set. If omitted, seqlet-derived metrics will not be computed.",
    )
    report_parser.add_argument(
        "-I",
        "--motifs-include",
        type=str,
        default=None,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.",
    )
    report_parser.add_argument(
        "-N",
        "--motif-names",
        type=str,
        default=None,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.",
    )

    report_parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="The path to the report output directory.",
    )

    report_parser.add_argument(
        "-W",
        "--modisco-region-width",
        type=int,
        default=inspect.signature(report).parameters["modisco_region_width"].default,
        help="The width of the region around each peak summit used by tfmodisco-lite.",
    )
    report_parser.add_argument(
        "-t",
        "--cwm-trim-threshold",
        type=float,
        default=inspect.signature(report).parameters["cwm_trim_threshold"].default,
        help="DEPRECATED: This information is now inferred from the outputs of `finemo call-hits`.",
    )
    report_parser.add_argument(
        "-n",
        "--no-recall",
        action="store_true",
        help="Do not compute motif recall metrics.",
    )
    report_parser.add_argument(
        "-s",
        "--no-seqlets",
        action="store_true",
        help="DEPRECATED: Please omit the `--modisco-h5` argument instead.",
    )

    collapse_hits_parser = subparsers.add_parser(
        "collapse-hits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Identify best hit by motif similarity among sets of overlapping hits.",
    )

    collapse_hits_parser.add_argument(
        "-i",
        "--hits",
        type=str,
        required=True,
        help="The `hits.tsv` or `hits_unique.tsv` file from `call-hits`.",
    )
    collapse_hits_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help='The path to the output .tsv file with an additional "is_primary" column.',
    )
    collapse_hits_parser.add_argument(
        "-O",
        "--overlap-frac",
        type=float,
        default=inspect.signature(collapse_hits).parameters["overlap_frac"].default,
        help="The threshold for determining overlapping hits. For two hits with lengths x and y, the minimum overlap is defined as `overlap_frac * (x + y) / 2`. The default value of 0.2 means that two hits must overlap by at least 20% of their average lengths to be considered overlapping.",
    )

    intersect_hits_parser = subparsers.add_parser(
        "intersect-hits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Intersect hits across multiple runs.",
    )

    intersect_hits_parser.add_argument(
        "-i",
        "--hits",
        type=str,
        required=True,
        nargs="+",
        help="One or more hits.tsv or hits_unique.tsv files, with paths delimited by whitespace.",
    )
    intersect_hits_parser.add_argument(
        "-o",
        "--out-path",
        type=str,
        required=True,
        help="The path to the output .tsv file. Duplicate columns are suffixed with the positional index of the input file.",
    )
    intersect_hits_parser.add_argument(
        "-r",
        "--relaxed",
        action="store_true",
        help="Use relaxed intersection criteria, using only motif names and untrimmed coordinates. By default, the intersection assumes consistent region definitions and motif trimming. This option is not recommended if genomic coordinates are unavailable.",
    )

    args = parser.parse_args()

    if args.cmd == "extract-regions-bw":
        extract_regions_bw(
            args.peaks,
            args.chrom_order,
            args.fasta,
            args.bigwigs,
            args.out_path,
            args.region_width,
        )

    elif args.cmd == "extract-regions-chrombpnet-h5":
        extract_regions_chrombpnet_h5(
            args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width
        )

    elif args.cmd == "extract-regions-h5":
        print(
            "WARNING: The `extract-regions-h5` command is deprecated. Use `extract-regions-chrombpnet-h5` instead."
        )
        extract_regions_chrombpnet_h5(
            args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width
        )

    elif args.cmd == "extract-regions-bpnet-h5":
        extract_regions_bpnet_h5(
            args.peaks, args.chrom_order, args.h5s, args.out_path, args.region_width
        )

    elif args.cmd == "extract-regions-modisco-fmt":
        extract_regions_modisco_fmt(
            args.peaks,
            args.chrom_order,
            args.attributions,
            args.sequences,
            args.out_path,
            args.region_width,
        )

    elif args.cmd == "call-hits":
        if args.alpha is not None:
            warnings.warn(
                "The `--alpha` flag is deprecated and will be removed in a future version. Please use the `--global-lambda` flag instead."
            )
            args.global_lambda = args.alpha
        if args.motif_alphas is not None:
            warnings.warn(
                "The `--motif-alphas` flag is deprecated and will be removed in a future version. Please use the `--motif-lambdas` flag instead."
            )
            args.motif_lambdas = args.motif_alphas

        call_hits(
            args.regions,
            args.peaks,
            args.modisco_h5,
            args.chrom_order,
            args.motifs_include,
            args.motif_names,
            args.motif_lambdas,
            args.out_dir,
            args.cwm_trim_coords,
            args.cwm_trim_thresholds,
            args.cwm_trim_threshold,
            args.global_lambda,
            args.step_size_max,
            args.step_size_min,
            args.sqrt_transform,
            args.convergence_tol,
            args.max_steps,
            args.batch_size,
            args.step_adjust,
            args.device,
            args.mode,
            args.no_post_filter,
            args.compile,
        )

    elif args.cmd == "report":
        report(
            args.regions,
            args.hits,
            args.modisco_h5,
            args.peaks,
            args.motifs_include,
            args.motif_names,
            args.out_dir,
            args.modisco_region_width,
            args.cwm_trim_threshold,
            not args.no_recall,
            not args.no_seqlets,
        )

    elif args.cmd == "collapse-hits":
        collapse_hits(args.hits, args.out_path, args.overlap_frac)

    elif args.cmd == "intersect-hits":
        intersect_hits(args.hits, args.out_path, args.relaxed)
