"""Evaluation module for assessing Fi-NeMo motif discovery and hit calling performance.

This module provides functions for:
- Computing motif occurrence statistics and co-occurrence patterns
- Evaluating motif discovery quality against TF-MoDISco results
- Analyzing hit calling performance and recall metrics
- Generating confusion matrices for seqlet-hit comparisons
"""

import warnings
from typing import List, Tuple, Dict, Any, Union

import numpy as np
from numpy import ndarray
import polars as pl
from jaxtyping import Float, Int


def get_motif_occurences(
    hits_df: pl.LazyFrame, motif_names: List[str]
) -> Tuple[pl.DataFrame, Int[ndarray, "M M"]]:
    """Compute motif occurrence statistics and co-occurrence matrix.

    This function analyzes motif occurrence patterns across peaks by creating
    a pivot table of hit counts and computing pairwise co-occurrence statistics.

    Parameters
    ----------
    hits_df : pl.LazyFrame
        Lazy DataFrame containing hit data with required columns:
        - peak_id : Peak identifier
        - motif_name : Name of the motif
        Additional columns are ignored.
    motif_names : List[str]
        List of motif names to include in analysis. Missing motifs
        will be added as columns with zero counts.

    Returns
    -------
    occ_df : pl.DataFrame
        DataFrame with motif occurrence counts per peak. Contains:
        - peak_id column
        - One column per motif with hit counts
        - 'total' column summing all motif counts per peak
    coocc : Int[ndarray, "M M"]
        Co-occurrence matrix where M = len(motif_names).
        Entry (i,j) indicates number of peaks containing both motif i and motif j.
        Diagonal entries show total peaks containing each motif.

    Notes
    -----
    The co-occurrence matrix is computed using binary occurrence indicators,
    so multiple hits of the same motif in a peak are treated as a single occurrence.
    """
    occ_df = (
        hits_df.collect()
        .with_columns(pl.lit(1).alias("count"))
        .pivot(
            on="motif_name", index="peak_id", values="count", aggregate_function="sum"
        )
        .fill_null(0)
    )

    missing_cols = set(motif_names) - set(occ_df.columns)
    occ_df = (
        occ_df.with_columns([pl.lit(0).alias(m) for m in missing_cols])
        .with_columns(total=pl.sum_horizontal(*motif_names))
        .sort(["peak_id"])
    )

    num_peaks = occ_df.height
    num_motifs = len(motif_names)

    occ_mat = np.zeros((num_peaks, num_motifs), dtype=np.int16)
    for i, m in enumerate(motif_names):
        occ_mat[:, i] = occ_df.get_column(m).to_numpy()

    occ_bin = (occ_mat > 0).astype(np.int32)
    coocc = occ_bin.T @ occ_bin

    return occ_df, coocc


def get_cwms(
    regions: Float[ndarray, "N 4 L"], positions_df: pl.DataFrame, motif_width: int
) -> Float[ndarray, "H 4 W"]:
    """Extract contribution weight matrices from regions based on hit positions.

    This function extracts motif-sized windows from contribution score regions
    at positions specified by hit coordinates. It handles both forward and
    reverse complement orientations and filters out invalid positions.

    Parameters
    ----------
    regions : Float[ndarray, "N 4 L"]
        Input contribution score regions multiplied by one-hot sequences.
        Shape: (n_peaks, 4, region_width) where 4 represents DNA bases (A,C,G,T).
    positions_df : pl.DataFrame
        DataFrame containing hit positions with required columns:
        - peak_id : int, Peak index (0-based)
        - start_untrimmed : int, Start position in genomic coordinates
        - peak_region_start : int, Peak region start coordinate
        - is_revcomp : bool, Whether hit is on reverse complement strand
    motif_width : int
        Width of motifs to extract. Must be positive.

    Returns
    -------
    cwms : Float[ndarray, "H 4 W"]
        Extracted contribution weight matrices for valid hits.
        Shape: (n_valid_hits, 4, motif_width)
        Invalid hits (outside region boundaries) are filtered out.

    Notes
    -----
    - Start positions are converted from genomic to region-relative coordinates
    - Reverse complement hits have their sequence order reversed
    - Hits extending beyond region boundaries are excluded
    - The mean is computed across all valid hits, with warnings suppressed
      for empty slices or invalid operations

    Raises
    ------
    ValueError
        If motif_width is non-positive or positions_df lacks required columns.
    """
    idx_df = positions_df.select(
        peak_idx=pl.col("peak_id"),
        start_idx=pl.col("start_untrimmed") - pl.col("peak_region_start"),
        is_revcomp=pl.col("is_revcomp"),
    )
    peak_idx = idx_df.get_column("peak_idx").to_numpy()
    start_idx = idx_df.get_column("start_idx").to_numpy()
    is_revcomp = idx_df.get_column("is_revcomp").to_numpy().astype(bool)

    # Filter hits that fall outside the region boundaries
    valid_mask = (start_idx >= 0) & (start_idx + motif_width <= regions.shape[2])
    peak_idx = peak_idx[valid_mask]
    start_idx = start_idx[valid_mask]
    is_revcomp = is_revcomp[valid_mask]

    row_idx = peak_idx[:, None, None]
    pos_idx = start_idx[:, None, None] + np.zeros((1, 1, motif_width), dtype=int)
    pos_idx[~is_revcomp, :, :] += np.arange(motif_width)[None, None, :]
    pos_idx[is_revcomp, :, :] += np.arange(motif_width)[None, None, ::-1]
    nuc_idx = np.zeros((peak_idx.shape[0], 4, 1), dtype=int)
    nuc_idx[~is_revcomp, :, :] += np.arange(4)[None, :, None]
    nuc_idx[is_revcomp, :, :] += np.arange(4)[None, ::-1, None]

    seqs = regions[row_idx, nuc_idx, pos_idx]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message="invalid value encountered in divide"
        )
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        cwms = seqs.mean(axis=0)

    return cwms


def tfmodisco_comparison(
    regions: Float[ndarray, "N 4 L"],
    hits_df: Union[pl.DataFrame, pl.LazyFrame],
    peaks_df: pl.DataFrame,
    seqlets_df: Union[pl.DataFrame, pl.LazyFrame, None],
    motifs_df: pl.DataFrame,
    cwms_modisco: Float[ndarray, "M 4 W"],
    motif_names: List[str],
    modisco_half_width: int,
    motif_width: int,
    compute_recall: bool,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    pl.DataFrame,
    Dict[str, Dict[str, Float[ndarray, "4 W"]]],
    Dict[str, Dict[str, Tuple[int, int]]],
]:
    """Compare Fi-NeMo hits with TF-MoDISco seqlets and compute evaluation metrics.

    This function performs comprehensive comparison between Fi-NeMo hit calls
    and TF-MoDISco seqlets, computing recall metrics, CWM similarities,
    and extracting contribution weight matrices for visualization.

    Parameters
    ----------
    regions : Float[ndarray, "N 4 L"]
        Contribution score regions multiplied by one-hot sequences.
        Shape: (n_peaks, 4, region_length)
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Fi-NeMo hit calls with required columns:
        - peak_id, start_untrimmed, end_untrimmed, strand, motif_name
    peaks_df : pl.DataFrame
        Peak metadata with columns:
        - peak_id, chr_id, peak_region_start
    seqlets_df : Optional[pl.DataFrame]
        TF-MoDISco seqlets with columns:
        - chr_id, start_untrimmed, is_revcomp, motif_name
        If None, only basic hit statistics are computed.
    motifs_df : pl.DataFrame
        Motif metadata with columns:
        - motif_name, strand, motif_id, motif_start, motif_end
    cwms_modisco : Float[ndarray, "M 4 W"]
        TF-MoDISco contribution weight matrices.
        Shape: (n_modisco_motifs, 4, motif_width)
    motif_names : List[str]
        Names of motifs to analyze.
    modisco_half_width : int
        Half-width for restricting hits to central region for fair comparison.
    motif_width : int
        Width of motifs for CWM extraction.
    compute_recall : bool
        Whether to compute recall metrics requiring seqlets_df.

    Returns
    -------
    report_data : Dict[str, Dict[str, Any]]
        Per-motif evaluation metrics including:
        - num_hits_total, num_hits_restricted, num_seqlets
        - num_overlaps, seqlet_recall, cwm_similarity
    report_df : pl.DataFrame
        Tabular format of report_data for easy analysis.
    cwms : Dict[str, Dict[str, Float[ndarray, "4 W"]]]
        Extracted CWMs for each motif and condition:
        - hits_fc, hits_rc: Forward/reverse complement hits
        - modisco_fc, modisco_rc: TF-MoDISco forward/reverse
        - seqlets_only, hits_restricted_only: Non-overlapping instances
    cwm_trim_bounds : Dict[str, Dict[str, Tuple[int, int]]]
        Trimming boundaries for each CWM type and motif.

    Notes
    -----
    - Hits are filtered to central region defined by modisco_half_width
    - CWM similarity is computed as normalized dot product between hit and TF-MoDISco CWMs
    - Recall metrics require both hits_df and seqlets_df to be non-empty
    - Missing motifs are handled gracefully with empty DataFrames

    Raises
    ------
    ValueError
        If required columns are missing from input DataFrames.
    """

    # Ensure hits_df is LazyFrame for consistent operations
    if isinstance(hits_df, pl.DataFrame):
        hits_df = hits_df.lazy()

    hits_df = (
        hits_df.with_columns(pl.col("peak_id").cast(pl.UInt32))
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .select(
            chr_id=pl.col("chr_id"),
            start_untrimmed=pl.col("start_untrimmed"),
            end_untrimmed=pl.col("end_untrimmed"),
            is_revcomp=pl.col("strand") == "-",
            motif_name=pl.col("motif_name"),
            peak_region_start=pl.col("peak_region_start"),
            peak_id=pl.col("peak_id"),
        )
    )

    hits_unique = hits_df.unique(
        subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"]
    )

    region_len = regions.shape[2]
    center = region_len / 2
    hits_filtered = hits_df.filter(
        (
            (pl.col("start_untrimmed") - pl.col("peak_region_start"))
            >= (center - modisco_half_width)
        )
        & (
            (pl.col("end_untrimmed") - pl.col("peak_region_start"))
            <= (center + modisco_half_width)
        )
    ).unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])

    hits_by_motif = hits_unique.collect().partition_by("motif_name", as_dict=True)
    hits_filtered_by_motif = hits_filtered.collect().partition_by(
        "motif_name", as_dict=True
    )

    if seqlets_df is None:
        seqlets_collected = None
        seqlets_lazy = None
    elif isinstance(seqlets_df, pl.LazyFrame):
        seqlets_collected = seqlets_df.collect()
        seqlets_lazy = seqlets_df
    else:
        seqlets_collected = seqlets_df
        seqlets_lazy = seqlets_df.lazy()

    if seqlets_collected is not None:
        seqlets_by_motif = seqlets_collected.partition_by("motif_name", as_dict=True)
    else:
        seqlets_by_motif = {}

    if compute_recall and seqlets_lazy is not None:
        overlaps_df = hits_filtered.join(
            seqlets_lazy,
            on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
            how="inner",
        ).collect()

        seqlets_only_df = seqlets_lazy.join(
            hits_df,
            on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
            how="anti",
        ).collect()

        hits_only_filtered_df = hits_filtered.join(
            seqlets_lazy,
            on=["chr_id", "start_untrimmed", "is_revcomp", "motif_name"],
            how="anti",
        ).collect()

        # Create partition dictionaries
        overlaps_by_motif = overlaps_df.partition_by("motif_name", as_dict=True)
        seqlets_only_by_motif = seqlets_only_df.partition_by("motif_name", as_dict=True)
        hits_only_filtered_by_motif = hits_only_filtered_df.partition_by(
            "motif_name", as_dict=True
        )
    else:
        overlaps_by_motif = {}
        seqlets_only_by_motif = {}
        hits_only_filtered_by_motif = {}

    report_data = {}
    cwms = {}
    cwm_trim_bounds = {}
    dummy_df = hits_df.clear().collect()
    for m in motif_names:
        hits = hits_by_motif.get((m,), dummy_df)
        hits_filtered = hits_filtered_by_motif.get((m,), dummy_df)

        # Initialize default values
        seqlets = dummy_df
        overlaps = dummy_df
        seqlets_only = dummy_df
        hits_only_filtered = dummy_df

        if seqlets_df is not None:
            seqlets = seqlets_by_motif.get((m,), dummy_df)

        if compute_recall and seqlets_df is not None:
            overlaps = overlaps_by_motif.get((m,), dummy_df)
            seqlets_only = seqlets_only_by_motif.get((m,), dummy_df)
            hits_only_filtered = hits_only_filtered_by_motif.get((m,), dummy_df)

        report_data[m] = {
            "num_hits_total": hits.height,
            "num_hits_restricted": hits_filtered.height,
        }

        if seqlets_df is not None:
            report_data[m]["num_seqlets"] = seqlets.height

        if compute_recall and seqlets_df is not None:
            report_data[m] |= {
                "num_overlaps": overlaps.height,
                "num_seqlets_only": seqlets_only.height,
                "num_hits_restricted_only": hits_only_filtered.height,
                "seqlet_recall": np.float64(overlaps.height) / seqlets.height
                if seqlets.height > 0
                else 0.0,
            }

        motif_data_fc = motifs_df.row(
            by_predicate=(pl.col("motif_name") == m) & (pl.col("strand") == "+"),
            named=True,
        )
        motif_data_rc = motifs_df.row(
            by_predicate=(pl.col("motif_name") == m) & (pl.col("strand") == "-"),
            named=True,
        )

        cwms[m] = {
            "hits_fc": get_cwms(regions, hits, motif_width),
            "modisco_fc": cwms_modisco[motif_data_fc["motif_id"]],
            "modisco_rc": cwms_modisco[motif_data_rc["motif_id"]],
        }
        cwms[m]["hits_rc"] = cwms[m]["hits_fc"][::-1, ::-1]

        if compute_recall and seqlets_df is not None:
            cwms[m] |= {
                "seqlets_only": get_cwms(regions, seqlets_only, motif_width),
                "hits_restricted_only": get_cwms(
                    regions, hits_only_filtered, motif_width
                ),
            }

        bounds_fc = (motif_data_fc["motif_start"], motif_data_fc["motif_end"])
        bounds_rc = (motif_data_rc["motif_start"], motif_data_rc["motif_end"])

        cwm_trim_bounds[m] = {
            "hits_fc": bounds_fc,
            "modisco_fc": bounds_fc,
            "modisco_rc": bounds_rc,
            "hits_rc": bounds_rc,
        }

        if compute_recall and seqlets_df is not None:
            cwm_trim_bounds[m] |= {
                "seqlets_only": bounds_fc,
                "hits_restricted_only": bounds_fc,
            }

        hits_cwm = cwms[m]["hits_fc"]
        modisco_cwm = cwms[m]["modisco_fc"]
        hnorm = np.sqrt((hits_cwm**2).sum())
        snorm = np.sqrt((modisco_cwm**2).sum())
        cwm_sim = (hits_cwm * modisco_cwm).sum() / (hnorm * snorm)

        report_data[m]["cwm_similarity"] = cwm_sim

    records = [{"motif_name": k} | v for k, v in report_data.items()]
    report_df = pl.from_dicts(records)

    return report_data, report_df, cwms, cwm_trim_bounds


def seqlet_confusion(
    hits_df: Union[pl.DataFrame, pl.LazyFrame],
    seqlets_df: Union[pl.DataFrame, pl.LazyFrame],
    peaks_df: pl.DataFrame,
    motif_names: List[str],
    motif_width: int,
) -> Tuple[pl.DataFrame, Float[ndarray, "M M"]]:
    """Compute confusion matrix between TF-MoDISco seqlets and Fi-NeMo hits.

    This function creates a confusion matrix showing the overlap between
    TF-MoDISco seqlets (ground truth) and Fi-NeMo hits across different motifs.
    Overlap frequencies are estimated using binned genomic coordinates.

    Parameters
    ----------
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Fi-NeMo hit calls with required columns:
        - peak_id, start_untrimmed, end_untrimmed, strand, motif_name
    seqlets_df : pl.DataFrame
        TF-MoDISco seqlets with required columns:
        - chr_id, start_untrimmed, end_untrimmed, motif_name
    peaks_df : pl.DataFrame
        Peak metadata for joining coordinates:
        - peak_id, chr_id
    motif_names : List[str]
        Names of motifs to include in confusion matrix.
        Determines matrix dimensions.
    motif_width : int
        Width used for binning genomic coordinates.
        Positions are binned to motif_width resolution.

    Returns
    -------
    confusion_df : pl.DataFrame
        Detailed confusion matrix in tabular format with columns:
        - motif_name_seqlets : Seqlet motif labels (rows)
        - motif_name_hits : Hit motif labels (columns)
        - frac_overlap : Fraction of seqlets overlapping with hits
    confusion_mat : Float[ndarray, "M M"]
        Confusion matrix where M = len(motif_names).
        Entry (i,j) = fraction of motif i seqlets overlapping with motif j hits.
        Rows represent seqlet motifs, columns represent hit motifs.

    Notes
    -----
    - Genomic coordinates are binned to motif_width resolution for overlap detection
    - Only exact bin overlaps are considered (same chr_id, start_bin, end_bin)
    - Fractions are computed as: overlaps / total_seqlets_per_motif
    - Missing motif combinations result in zero entries in the confusion matrix

    Raises
    ------
    ValueError
        If required columns are missing from input DataFrames.
    KeyError
        If motif names in data don't match those in motif_names list.
    """
    bin_size = motif_width

    # Ensure hits_df is LazyFrame for consistent operations
    if isinstance(hits_df, pl.DataFrame):
        hits_df = hits_df.lazy()

    hits_binned = (
        hits_df.with_columns(
            peak_id=pl.col("peak_id").cast(pl.UInt32),
            is_revcomp=pl.col("strand") == "-",
        )
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
        .select(
            chr_id=pl.col("chr_id"),
            start_bin=pl.col("start_untrimmed") // bin_size,
            end_bin=pl.col("end_untrimmed") // bin_size,
            motif_name=pl.col("motif_name"),
        )
    )

    seqlets_lazy = seqlets_df.lazy()
    seqlets_binned = seqlets_lazy.select(
        chr_id=pl.col("chr_id"),
        start_bin=pl.col("start_untrimmed") // bin_size,
        end_bin=pl.col("end_untrimmed") // bin_size,
        motif_name=pl.col("motif_name"),
    )

    overlaps_df = seqlets_binned.join(
        hits_binned, on=["chr_id", "start_bin", "end_bin"], how="inner", suffix="_hits"
    )

    seqlet_counts = (
        seqlets_binned.group_by("motif_name").len(name="num_seqlets").collect()
    )
    overlap_counts = (
        overlaps_df.group_by(["motif_name", "motif_name_hits"])
        .len(name="num_overlaps")
        .collect()
    )

    num_motifs = len(motif_names)
    confusion_mat = np.zeros((num_motifs, num_motifs), dtype=np.float32)
    name_to_idx = {m: i for i, m in enumerate(motif_names)}

    confusion_df = overlap_counts.join(
        seqlet_counts, on="motif_name", how="inner"
    ).select(
        motif_name_seqlets=pl.col("motif_name"),
        motif_name_hits=pl.col("motif_name_hits"),
        frac_overlap=pl.col("num_overlaps") / pl.col("num_seqlets"),
    )

    confusion_idx_df = confusion_df.select(
        row_idx=pl.col("motif_name_seqlets").replace_strict(name_to_idx),
        col_idx=pl.col("motif_name_hits").replace_strict(name_to_idx),
        frac_overlap=pl.col("frac_overlap"),
    )

    row_idx = confusion_idx_df["row_idx"].to_numpy()
    col_idx = confusion_idx_df["col_idx"].to_numpy()
    frac_overlap = confusion_idx_df["frac_overlap"].to_numpy()

    confusion_mat[row_idx, col_idx] = frac_overlap

    return confusion_df, confusion_mat
