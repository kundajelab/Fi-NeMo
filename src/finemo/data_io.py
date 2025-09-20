"""Data input/output module for the Fi-NeMo motif instance calling pipeline.

This module handles loading and processing of various genomic data formats including:
- Peak region files (ENCODE NarrowPeak format)
- Genome sequences (FASTA format)
- Contribution scores (bigWig, HDF5 formats)
- Neural network model outputs
- Motif data from TF-MoDISco
- Hit calling results

The module supports multiple input formats used for contribution scores
and provides utilities for data conversion and quality control.
"""

import json
import os
import warnings
from contextlib import ExitStack
from typing import List, Dict, Tuple, Optional, Any, Union, Callable

import numpy as np
from numpy import ndarray
import h5py
import hdf5plugin  # noqa: F401, imported for side effects (HDF5 plugin registration)
import polars as pl
import pyBigWig
import pyfaidx
from jaxtyping import Float, Int

from tqdm import tqdm


def load_txt(path: str) -> List[str]:
    """Load a text file containing one item per line.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    List[str]
        List of strings, one per line (first column if tab-delimited).
    """
    entries = []
    with open(path) as f:
        for line in f:
            item = line.rstrip("\n").split("\t")[0]
            entries.append(item)

    return entries


def load_mapping(path: str, value_type: Callable[[str], Any]) -> Dict[str, Any]:
    """Load a two-column tab-delimited mapping file.

    Parameters
    ----------
    path : str
        Path to the mapping file. Must be tab-delimited with exactly two columns.
    value_type : Callable[[str], Any]
        Type constructor to apply to values (e.g., int, float, str).
        Must accept a string and return the converted value.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping keys to values of the specified type.

    Raises
    ------
    ValueError
        If lines don't contain exactly two tab-separated values.
    FileNotFoundError
        If the specified file does not exist.
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            key, val = line.rstrip("\n").split("\t")
            mapping[key] = value_type(val)

    return mapping


def load_mapping_tuple(
    path: str, value_type: Callable[[str], Any]
) -> Dict[str, Tuple[Any, ...]]:
    """Load a mapping file where values are tuples from multiple columns.

    Parameters
    ----------
    path : str
        Path to the mapping file. Must be tab-delimited with multiple columns.
    value_type : Callable[[str], Any]
        Type constructor to apply to each value element.
        Must accept a string and return the converted value.

    Returns
    -------
    Dict[str, Tuple[Any, ...]]
        Dictionary mapping keys to tuples of values of the specified type.
        The first column is used as the key, remaining columns as tuple values.

    Raises
    ------
    ValueError
        If lines don't contain at least two tab-separated values.
    FileNotFoundError
        If the specified file does not exist.
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            entries = line.rstrip("\n").split("\t")
            key = entries[0]
            val = entries[1:]
            mapping[key] = tuple(value_type(i) for i in val)

    return mapping


# ENCODE NarrowPeak format column definitions
NARROWPEAK_SCHEMA: List[str] = [
    "chr",
    "peak_start",
    "peak_end",
    "peak_name",
    "peak_score",
    "peak_strand",
    "peak_signal",
    "peak_pval",
    "peak_qval",
    "peak_summit",
]
NARROWPEAK_DTYPES: List[Any] = [
    pl.String,
    pl.Int32,
    pl.Int32,
    pl.String,
    pl.UInt32,
    pl.String,
    pl.Float32,
    pl.Float32,
    pl.Float32,
    pl.Int32,
]


def load_peaks(
    peaks_path: str, chrom_order_path: Optional[str], half_width: int
) -> pl.DataFrame:
    """Load peak region data from ENCODE NarrowPeak format file.

    Parameters
    ----------
    peaks_path : str
        Path to the NarrowPeak format file.
    chrom_order_path : str, optional
        Path to file defining chromosome ordering. If None, uses order from peaks file.
    half_width : int
        Half-width of regions around peak summits.

    Returns
    -------
    pl.DataFrame
        DataFrame containing peak information with columns:
        - chr: Chromosome name
        - peak_region_start: Start coordinate of centered region
        - peak_name: Peak identifier
        - peak_id: Sequential peak index
        - chr_id: Numeric chromosome identifier
    """
    peaks = (
        pl.scan_csv(
            peaks_path,
            has_header=False,
            new_columns=NARROWPEAK_SCHEMA,
            separator="\t",
            quote_char=None,
            schema_overrides=NARROWPEAK_DTYPES,
            null_values=[".", "NA", "null", "NaN"],
        )
        .select(
            chr=pl.col("chr"),
            peak_region_start=pl.col("peak_start") + pl.col("peak_summit") - half_width,
            peak_name=pl.col("peak_name"),
        )
        .with_row_index(name="peak_id")
        .collect()
    )

    if chrom_order_path is not None:
        chrom_order = load_txt(chrom_order_path)
    else:
        chrom_order = []

    chrom_order_set = set(chrom_order)
    chrom_order_peaks = [
        i
        for i in peaks.get_column("chr").unique(maintain_order=True)
        if i not in chrom_order_set
    ]
    chrom_order.extend(chrom_order_peaks)
    chrom_ind_map = {val: ind for ind, val in enumerate(chrom_order)}

    peaks = peaks.with_columns(
        pl.col("chr").replace_strict(chrom_ind_map).alias("chr_id")
    )

    return peaks


# DNA sequence alphabet for one-hot encoding
SEQ_ALPHABET: np.ndarray = np.array(["A", "C", "G", "T"], dtype="S1")


def one_hot_encode(sequence: str, dtype: Any = np.int8) -> Int[ndarray, "4 L"]:
    """Convert DNA sequence string to one-hot encoded matrix.

    Parameters
    ----------
    sequence : str
        DNA sequence string containing A, C, G, T characters.
    dtype : np.dtype, default np.int8
        Data type for the output array.

    Returns
    -------
    Int[ndarray, "4 L"]
        One-hot encoded sequence where rows correspond to A, C, G, T and
        L is the sequence length.

    Notes
    -----
    The output array has shape (4, len(sequence)) with rows corresponding to
    nucleotides A, C, G, T in that order. Non-standard nucleotides (N, etc.)
    result in all-zero columns.
    """
    sequence = sequence.upper()

    seq_chararray = np.frombuffer(sequence.encode("UTF-8"), dtype="S1")
    one_hot = (seq_chararray[None, :] == SEQ_ALPHABET[:, None]).astype(dtype)

    return one_hot


def load_regions_from_bw(
    peaks: pl.DataFrame, fa_path: str, bw_paths: List[str], half_width: int
) -> Tuple[Int[ndarray, "N 4 L"], Float[ndarray, "N L"]]:
    """Load genomic sequences and contribution scores from FASTA and bigWig files.

    Parameters
    ----------
    peaks : pl.DataFrame
        Peak regions DataFrame from load_peaks() containing columns:
        'chr', 'peak_region_start'.
    fa_path : str
        Path to genome FASTA file (.fa or .fasta format).
    bw_paths : List[str]
        List of paths to bigWig files containing contribution scores.
        Must be non-empty.
    half_width : int
        Half-width of regions to extract around peak centers.
        Total region width will be 2 * half_width.

    Returns
    -------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of peaks,
        4 represents A,C,G,T nucleotides, and L is the region length (2 * half_width).
    contribs : Float[ndarray, "N L"]
        Contribution scores averaged across input bigWig files.
        Shape is (N peaks, L region_length).

    Notes
    -----
    BigWig files only provide projected contribution scores, not hypothetical scores.
    Regions extending beyond chromosome boundaries are zero-padded.
    Missing values in bigWig files are converted to zero.
    """
    num_peaks = peaks.height
    region_width = half_width * 2

    sequences = np.zeros((num_peaks, 4, region_width), dtype=np.int8)
    contribs = np.zeros((num_peaks, region_width), dtype=np.float16)

    # Load genome reference
    genome = pyfaidx.Fasta(fa_path, one_based_attributes=False)

    bws = [pyBigWig.open(i) for i in bw_paths]
    contrib_buffer = np.zeros((len(bw_paths), half_width * 2), dtype=np.float16)

    try:
        for ind, row in tqdm(
            enumerate(peaks.iter_rows(named=True)),
            disable=None,
            unit="regions",
            total=num_peaks,
        ):
            chrom = row["chr"]
            start = row["peak_region_start"]
            end = start + 2 * half_width

            sequence_data: pyfaidx.FastaRecord = genome[chrom][start:end]  # type: ignore
            sequence: str = sequence_data.seq  # type: ignore
            start_adj: int = sequence_data.start  # type: ignore
            end_adj: int = sequence_data.end  # type: ignore
            a = start_adj - start
            b = end_adj - start

            if b > a:
                sequences[ind, :, a:b] = one_hot_encode(sequence)

                for j, bw in enumerate(bws):
                    contrib_buffer[j, :] = np.nan_to_num(
                        bw.values(chrom, start_adj, end_adj, numpy=True)
                    )

                contribs[ind, a:b] = np.mean(contrib_buffer, axis=0)

    finally:
        for bw in bws:
            bw.close()

    return sequences, contribs


def load_regions_from_chrombpnet_h5(
    h5_paths: List[str], half_width: int
) -> Tuple[Int[ndarray, "N 4 L"], Float[ndarray, "N 4 L"]]:
    """Load genomic sequences and contribution scores from ChromBPNet HDF5 files.

    Parameters
    ----------
    h5_paths : List[str]
        List of paths to ChromBPNet HDF5 files containing sequences and SHAP scores.
        Must be non-empty and contain compatible data shapes.
    half_width : int
        Half-width of regions to extract around the center.
        Total region width will be 2 * half_width.

    Returns
    -------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of regions,
        4 represents A,C,G,T nucleotides, and L is the region length (2 * half_width).
    contribs : Float[ndarray, "N 4 L"]
        SHAP contribution scores averaged across input files.
        Shape is (N regions, 4 nucleotides, L region_length).

    Notes
    -----
    ChromBPNet files store sequences in 'raw/seq' and SHAP scores in 'shap/seq'.
    All input files must have the same dimensions and number of regions.
    Missing values in contribution scores are converted to zero.
    """
    with ExitStack() as stack:
        h5s = [stack.enter_context(h5py.File(i)) for i in h5_paths]

        start = h5s[0]["raw/seq"].shape[-1] // 2 - half_width  # type: ignore  # HDF5 array access
        end = start + 2 * half_width

        sequences = h5s[0]["raw/seq"][:, :, start:end].astype(np.int8)  # type: ignore  # HDF5 array access
        contribs = np.mean(
            [np.nan_to_num(f["shap/seq"][:, :, start:end]) for f in h5s],  # type: ignore  # HDF5 array access
            axis=0,
            dtype=np.float16,
        )

    return sequences, contribs  # type: ignore  # HDF5 arrays converted to NumPy


def load_regions_from_bpnet_h5(
    h5_paths: List[str], half_width: int
) -> Tuple[Int[ndarray, "N 4 L"], Float[ndarray, "N 4 L"]]:
    """Load genomic sequences and contribution scores from BPNet HDF5 files.

    Parameters
    ----------
    h5_paths : List[str]
        List of paths to BPNet HDF5 files containing sequences and contribution scores.
        Must be non-empty and contain compatible data shapes.
    half_width : int
        Half-width of regions to extract around the center.
        Total region width will be 2 * half_width.

    Returns
    -------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of regions,
        4 represents A,C,G,T nucleotides, and L is the region length (2 * half_width).
    contribs : Float[ndarray, "N 4 L"]
        Hypothetical contribution scores averaged across input files.
        Shape is (N regions, 4 nucleotides, L region_length).

    Notes
    -----
    BPNet files store sequences in 'input_seqs' and hypothetical scores in 'hyp_scores'.
    The data requires axis swapping to convert from (n, length, 4) to (n, 4, length) format.
    All input files must have the same dimensions and number of regions.
    Missing values in contribution scores are converted to zero.
    """
    with ExitStack() as stack:
        h5s = [stack.enter_context(h5py.File(i)) for i in h5_paths]

        start = h5s[0]["input_seqs"].shape[-2] // 2 - half_width  # type: ignore  # HDF5 array access
        end = start + 2 * half_width

        sequences = h5s[0]["input_seqs"][:, start:end, :].swapaxes(1, 2).astype(np.int8)  # type: ignore  # HDF5 array access with axis swap
        contribs = np.mean(
            [
                np.nan_to_num(f["hyp_scores"][:, start:end, :].swapaxes(1, 2))  # type: ignore  # HDF5 array access
                for f in h5s
            ],
            axis=0,
            dtype=np.float16,
        )

    return sequences, contribs


def load_npy_or_npz(path: str) -> ndarray:
    """Load array data from .npy or .npz file.

    Parameters
    ----------
    path : str
        Path to .npy or .npz file. File must exist and contain valid NumPy data.

    Returns
    -------
    ndarray
        Loaded array data. For .npz files, returns the first array ('arr_0').
        For .npy files, returns the array directly.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If .npz file does not contain 'arr_0' key.
    """
    f = np.load(path)
    if isinstance(f, np.ndarray):
        arr = f
    else:
        arr = f["arr_0"]

    return arr


def load_regions_from_modisco_fmt(
    shaps_paths: List[str], ohe_path: str, half_width: int
) -> Tuple[Int[ndarray, "N 4 L"], Float[ndarray, "N 4 L"]]:
    """Load genomic sequences and contribution scores from TF-MoDISco format files.

    Parameters
    ----------
    shaps_paths : List[str]
        List of paths to .npy/.npz files containing SHAP/attribution scores.
        Must be non-empty and all files must have compatible shapes.
    ohe_path : str
        Path to .npy/.npz file containing one-hot encoded sequences.
        Must have shape (n_regions, 4, sequence_length).
    half_width : int
        Half-width of regions to extract around the center.
        Total region width will be 2 * half_width.

    Returns
    -------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of regions,
        4 represents A,C,G,T nucleotides, and L is the region length (2 * half_width).
    contribs : Float[ndarray, "N 4 L"]
        SHAP contribution scores averaged across input files.
        Shape is (N regions, 4 nucleotides, L region_length).

    Notes
    -----
    All SHAP files must have the same shape as the sequence file.
    Missing values in contribution scores are converted to zero.
    The center of the input sequences is used as the reference point for extraction.
    """
    sequences_raw = load_npy_or_npz(ohe_path)

    start = sequences_raw.shape[-1] // 2 - half_width
    end = start + 2 * half_width

    sequences = sequences_raw[:, :, start:end].astype(np.int8)

    shaps = [np.nan_to_num(load_npy_or_npz(p)[:, :, start:end]) for p in shaps_paths]
    contribs = np.mean(shaps, axis=0, dtype=np.float16)

    return sequences, contribs


def load_regions_npz(
    npz_path: str,
) -> Tuple[
    Int[ndarray, "N 4 L"],
    Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]],
    pl.DataFrame,
    bool,
]:
    """Load preprocessed genomic regions from NPZ file.

    Parameters
    ----------
    npz_path : str
        Path to NPZ file containing sequences, contributions, and optional coordinates.
        Must contain 'sequences' and 'contributions' arrays at minimum.

    Returns
    -------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of regions,
        4 represents A,C,G,T nucleotides, and L is the region length.
    contributions : Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]]
        Contribution scores in either hypothetical format (N, 4, L) or
        projected format (N, L). Shape depends on the input data format.
    peaks_df : pl.DataFrame
        DataFrame containing peak region information with columns:
        'chr', 'chr_id', 'peak_region_start', 'peak_id', 'peak_name'.
    has_peaks : bool
        Whether the file contains genomic coordinate information.
        If False, placeholder coordinate data is used.

    Notes
    -----
    If genomic coordinates are not present in the NPZ file, creates placeholder
    coordinate data and issues a warning. The placeholder data uses 'NA' for
    chromosome names and sequential indices for peak IDs.

    Raises
    ------
    KeyError
        If required arrays 'sequences' or 'contributions' are missing from the file.
    """
    data = np.load(npz_path)

    if "chr" not in data.keys():
        warnings.warn(
            "No genome coordinates present in the input .npz file. Returning sequences and contributions only."
        )
        has_peaks = False
        num_regions = data["sequences"].shape[0]
        peak_data = {
            "chr": np.array(["NA"] * num_regions, dtype="U"),
            "chr_id": np.arange(num_regions, dtype=np.uint32),
            "peak_region_start": np.zeros(num_regions, dtype=np.int32),
            "peak_id": np.arange(num_regions, dtype=np.uint32),
            "peak_name": np.array(["NA"] * num_regions, dtype="U"),
        }

    else:
        has_peaks = True
        peak_data = {
            "chr": data["chr"],
            "chr_id": data["chr_id"],
            "peak_region_start": data["start"],
            "peak_id": data["peak_id"],
            "peak_name": data["peak_name"],
        }

    peaks_df = pl.DataFrame(peak_data)

    return data["sequences"], data["contributions"], peaks_df, has_peaks


def write_regions_npz(
    sequences: Int[ndarray, "N 4 L"],
    contributions: Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]],
    out_path: str,
    peaks_df: Optional[pl.DataFrame] = None,
) -> None:
    """Write genomic regions and contribution scores to compressed NPZ file.

    Parameters
    ----------
    sequences : Int[ndarray, "N 4 L"]
        One-hot encoded DNA sequences where N is the number of regions,
        4 represents A,C,G,T nucleotides, and L is the region length.
    contributions : Union[Float[ndarray, "N 4 L"], Float[ndarray, "N L"]]
        Contribution scores in either hypothetical format (N, 4, L) or
        projected format (N, L).
    out_path : str
        Output path for the NPZ file. Parent directory must exist.
    peaks_df : Optional[pl.DataFrame]
        DataFrame containing peak region information with columns:
        'chr', 'chr_id', 'peak_region_start', 'peak_id', 'peak_name'.
        If None, only sequences and contributions are saved.

    Raises
    ------
    ValueError
        If the number of regions in sequences/contributions doesn't match peaks_df.
    FileNotFoundError
        If the parent directory of out_path does not exist.

    Notes
    -----
    The output file is compressed using NumPy's savez_compressed format.
    If peaks_df is provided, genomic coordinate information is included
    in the output file for downstream analysis.
    """
    if peaks_df is None:
        warnings.warn(
            "No genome coordinates provided. Writing sequences and contributions only."
        )
        np.savez_compressed(out_path, sequences=sequences, contributions=contributions)

    else:
        num_regions = peaks_df.height
        if (num_regions != sequences.shape[0]) or (
            num_regions != contributions.shape[0]
        ):
            raise ValueError(
                f"Input sequences of shape {sequences.shape} and/or "
                f"input contributions of shape {contributions.shape} "
                f"are not compatible with peak region count of {num_regions}"
            )

        chr_arr = peaks_df.get_column("chr").to_numpy().astype("U")
        chr_id_arr = peaks_df.get_column("chr_id").to_numpy()
        start_arr = peaks_df.get_column("peak_region_start").to_numpy()
        peak_id_arr = peaks_df.get_column("peak_id").to_numpy()
        peak_name_arr = peaks_df.get_column("peak_name").to_numpy().astype("U")
        np.savez_compressed(
            out_path,
            sequences=sequences,
            contributions=contributions,
            chr=chr_arr,
            chr_id=chr_id_arr,
            start=start_arr,
            peak_id=peak_id_arr,
            peak_name=peak_name_arr,
        )


def trim_motif(cwm: Float[ndarray, "4 W"], trim_threshold: float) -> Tuple[int, int]:
    """Determine trimmed start and end positions for a motif based on contribution magnitude.

    This function identifies the core region of a motif by finding positions where
    the total absolute contribution exceeds a threshold relative to the maximum.

    Parameters
    ----------
    cwm : Float[ndarray, "4 W"]
        Contribution weight matrix for the motif where 4 represents A,C,G,T
        nucleotides and W is the motif width.
    trim_threshold : float
        Fraction of maximum score to use as trimming threshold (0.0 to 1.0).
        Higher values result in more aggressive trimming.

    Returns
    -------
    start : int
        Start position of the trimmed motif (inclusive).
    end : int
        End position of the trimmed motif (exclusive).

    Notes
    -----
    The trimming is based on the sum of absolute contributions across all nucleotides
    at each position. Positions with contributions below trim_threshold * max_score
    are removed from the motif edges.

    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L213-L236
    """
    score = np.sum(np.abs(cwm), axis=0)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.nonzero(score >= trim_thresh)
    start = max(int(np.min(pass_inds)), 0)  # type: ignore  # nonzero returns tuple of arrays
    end = min(int(np.max(pass_inds)) + 1, len(score))  # type: ignore  # nonzero returns tuple of arrays

    return start, end


def softmax(x: Float[ndarray, "4 W"], temp: float = 100) -> Float[ndarray, "4 W"]:
    """Apply softmax transformation with temperature scaling.

    Parameters
    ----------
    x : Float[ndarray, "4 W"]
        Input array to transform where 4 represents A,C,G,T nucleotides
        and W is the motif width.
    temp : float, default 100
        Temperature parameter for softmax scaling. Higher values create
        sharper probability distributions.

    Returns
    -------
    Float[ndarray, "4 W"]
        Softmax-transformed array with same shape as input. Each column
        sums to 1.0, representing nucleotide probabilities at each position.

    Notes
    -----
    The softmax is applied along the nucleotide axis (axis=0), normalizing
    each position to have probabilities that sum to 1. The temperature
    parameter controls the sharpness of the distribution.
    """
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    exp = np.exp(temp * norm_x)
    return exp / np.sum(exp, axis=0, keepdims=True)


def _motif_name_sort_key(data: Tuple[str, Any]) -> Union[Tuple[int, int], Tuple[int, str]]:
    """Generate sort key for TF-MoDISco motif names.

    This function creates a sort key that orders motifs by pattern number,
    with non-standard patterns sorted to the end.

    Parameters
    ----------
    data : Tuple[str, Any]
        Tuple containing motif name as first element and additional data.
        The motif name should follow the format 'pattern_N' or 'pattern#N' where N is an integer.

    Returns
    -------
    Union[Tuple[int, int], Tuple[int, str]]
        Sort key tuple for ordering motifs. Standard pattern names return
        (0, pattern_number) while non-standard names return (1, name).

    Notes
    -----
    This function is used internally by load_modisco_motifs to ensure
    consistent motif ordering across runs.
    """
    pattern_name = data[0]
    try:
        return (0, int(pattern_name.split("_")[-1]))
    except (ValueError, IndexError):
        try:
            return (0, int(pattern_name.split("#")[-1]))
        except (ValueError, IndexError):
            return (1, pattern_name)


MODISCO_PATTERN_GROUPS = ["pos_patterns", "neg_patterns"]


def load_modisco_motifs(
    modisco_h5_path: str,
    trim_coords: Optional[Dict[str, Tuple[int, int]]],
    trim_thresholds: Optional[Dict[str, float]],
    trim_threshold_default: float,
    motif_type: str,
    motifs_include: Optional[List[str]],
    motif_name_map: Optional[Dict[str, str]],
    motif_lambdas: Optional[Dict[str, float]],
    motif_lambda_default: float,
    include_rc: bool,
) -> Tuple[pl.DataFrame, Float[ndarray, "M 4 W"], Int[ndarray, "M W"], ndarray]:
    """Load motif data from TF-MoDISco HDF5 file with customizable processing options.

    This function extracts contribution weight matrices and associated metadata from
    TF-MoDISco results, with support for custom naming, trimming, and regularization
    parameters.

    Parameters
    ----------
    modisco_h5_path : str
        Path to TF-MoDISco HDF5 results file containing pattern groups.
    trim_coords : Optional[Dict[str, Tuple[int, int]]]
        Manual trim coordinates for specific motifs {motif_name: (start, end)}.
        Takes precedence over automatic trimming based on thresholds.
    trim_thresholds : Optional[Dict[str, float]]
        Custom trim thresholds for specific motifs {motif_name: threshold}.
        Values should be between 0.0 and 1.0.
    trim_threshold_default : float
        Default trim threshold for motifs not in trim_thresholds.
        Fraction of maximum contribution used for trimming.
    motif_type : str
        Type of motif to extract. Must be one of:
        - 'cwm': Contribution weight matrix (normalized)
        - 'hcwm': Hypothetical contribution weight matrix
        - 'pfm': Position frequency matrix
        - 'pfm_softmax': Softmax-transformed position frequency matrix
    motifs_include : Optional[List[str]]
        List of motif names to include. If None, includes all motifs found.
        Names should follow format 'pos_patterns.pattern_N' or 'neg_patterns.pattern_N'.
    motif_name_map : Optional[Dict[str, str]]
        Mapping from original to custom motif names {orig_name: new_name}.
        New names must be unique across all motifs.
    motif_lambdas : Optional[Dict[str, float]]
        Custom lambda regularization values for specific motifs {motif_name: lambda}.
        Higher values increase sparsity penalty for the corresponding motif.
    motif_lambda_default : float
        Default lambda value for motifs not specified in motif_lambdas.
    include_rc : bool
        Whether to include reverse complement motifs in addition to forward motifs.
        If True, doubles the number of motifs returned.

    Returns
    -------
    motifs_df : pl.DataFrame
        DataFrame containing motif metadata with columns: motif_id, motif_name,
        motif_name_orig, strand, motif_start, motif_end, motif_scale, lambda.
    cwms : Float[ndarray, "M 4 W"]
        Contribution weight matrices for all motifs where M is the number of motifs,
        4 represents A,C,G,T nucleotides, and W is the motif width.
    trim_masks : Int[ndarray, "M W"]
        Binary masks indicating core motif regions (1) vs trimmed regions (0).
        Shape is (M motifs, W motif_width).
    names : ndarray
        Array of unique motif names (forward strand only).

    Raises
    ------
    ValueError
        If motif_type is not one of the supported types, or if motif names
        in motif_name_map are not unique.
    FileNotFoundError
        If the specified HDF5 file does not exist.
    KeyError
        If required datasets are missing from the HDF5 file.

    Notes
    -----
    Motif trimming removes low-contribution positions from the edges based on
    the position-wise sum of absolute contributions across nucleotides. The trimming
    helps focus on the core binding site.

    Adapted from https://github.com/jmschrei/tfmodisco-lite/blob/570535ee5ccf43d670e898d92d63af43d68c38c5/modiscolite/report.py#L252-L272
    """
    motif_data_lsts = {
        "motif_name": [],
        "motif_name_orig": [],
        "strand": [],
        "motif_start": [],
        "motif_end": [],
        "motif_scale": [],
        "lambda": [],
    }
    motif_lst = []
    trim_mask_lst = []

    if motifs_include is not None:
        motifs_include_set = set(motifs_include)
    else:
        motifs_include_set = None

    if motif_name_map is None:
        motif_name_map = {}

    if motif_lambdas is None:
        motif_lambdas = {}

    if trim_coords is None:
        trim_coords = {}
    if trim_thresholds is None:
        trim_thresholds = {}

    if len(motif_name_map.values()) != len(set(motif_name_map.values())):
        raise ValueError("Specified motif names are not unique")

    with h5py.File(modisco_h5_path, "r") as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]
            for _, (pattern_name, pattern) in enumerate(
                sorted(metacluster.items(), key=_motif_name_sort_key)  # type: ignore  # HDF5 access
            ):
                pattern_tag = f"{name}.{pattern_name}"

                if (
                    motifs_include_set is not None
                    and pattern_tag not in motifs_include_set
                ):
                    continue

                motif_lambda = motif_lambdas.get(pattern_tag, motif_lambda_default)
                pattern_tag_orig = pattern_tag
                pattern_tag = motif_name_map.get(pattern_tag, pattern_tag)

                cwm_raw = pattern["contrib_scores"][:].T  # type: ignore
                cwm_norm = np.sqrt((cwm_raw**2).sum())

                cwm_fwd = cwm_raw / cwm_norm
                cwm_rev = cwm_fwd[::-1, ::-1]

                if pattern_tag in trim_coords:
                    start_fwd, end_fwd = trim_coords[pattern_tag]
                else:
                    trim_threshold = trim_thresholds.get(
                        pattern_tag, trim_threshold_default
                    )
                    start_fwd, end_fwd = trim_motif(cwm_fwd, trim_threshold)

                cwm_len = cwm_fwd.shape[1]
                start_rev, end_rev = cwm_len - end_fwd, cwm_len - start_fwd

                trim_mask_fwd = np.zeros(cwm_fwd.shape[1], dtype=np.int8)
                trim_mask_fwd[start_fwd:end_fwd] = 1
                trim_mask_rev = np.zeros(cwm_rev.shape[1], dtype=np.int8)
                trim_mask_rev[start_rev:end_rev] = 1

                if motif_type == "cwm":
                    motif_fwd = cwm_fwd
                    motif_rev = cwm_rev
                    motif_norm = cwm_norm

                elif motif_type == "hcwm":
                    motif_raw = pattern["hypothetical_contribs"][:].T  # type: ignore
                    motif_norm = np.sqrt((motif_raw**2).sum())

                    motif_fwd = motif_raw / motif_norm
                    motif_rev = motif_fwd[::-1, ::-1]

                elif motif_type == "pfm":
                    motif_raw = pattern["sequence"][:].T  # type: ignore
                    motif_norm = 1

                    motif_fwd = motif_raw / np.sum(motif_raw, axis=0, keepdims=True)
                    motif_rev = motif_fwd[::-1, ::-1]

                elif motif_type == "pfm_softmax":
                    motif_raw = pattern["sequence"][:].T  # type: ignore
                    motif_norm = 1

                    motif_fwd = softmax(motif_raw)
                    motif_rev = motif_fwd[::-1, ::-1]

                else:
                    raise ValueError(
                        f"Invalid motif_type: {motif_type}. Must be one of 'cwm', 'hcwm', 'pfm', 'pfm_softmax'."
                    )

                motif_data_lsts["motif_name"].append(pattern_tag)
                motif_data_lsts["motif_name_orig"].append(pattern_tag_orig)
                motif_data_lsts["strand"].append("+")
                motif_data_lsts["motif_start"].append(start_fwd)
                motif_data_lsts["motif_end"].append(end_fwd)
                motif_data_lsts["motif_scale"].append(motif_norm)
                motif_data_lsts["lambda"].append(motif_lambda)

                if include_rc:
                    motif_data_lsts["motif_name"].append(pattern_tag)
                    motif_data_lsts["motif_name_orig"].append(pattern_tag_orig)
                    motif_data_lsts["strand"].append("-")
                    motif_data_lsts["motif_start"].append(start_rev)
                    motif_data_lsts["motif_end"].append(end_rev)
                    motif_data_lsts["motif_scale"].append(motif_norm)
                    motif_data_lsts["lambda"].append(motif_lambda)

                    motif_lst.extend([motif_fwd, motif_rev])
                    trim_mask_lst.extend([trim_mask_fwd, trim_mask_rev])

                else:
                    motif_lst.append(motif_fwd)
                    trim_mask_lst.append(trim_mask_fwd)

    motifs_df = pl.DataFrame(motif_data_lsts).with_row_index(name="motif_id")
    cwms = np.stack(motif_lst, dtype=np.float16, axis=0)
    trim_masks = np.stack(trim_mask_lst, dtype=np.int8, axis=0)
    names = (
        motifs_df.filter(pl.col("strand") == "+").get_column("motif_name").to_numpy()
    )

    return motifs_df, cwms, trim_masks, names


def load_modisco_seqlets(
    modisco_h5_path: str,
    peaks_df: pl.DataFrame,
    motifs_df: pl.DataFrame,
    half_width: int,
    modisco_half_width: int,
    lazy: bool = False,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Load seqlet data from TF-MoDISco HDF5 file and convert to genomic coordinates.

    This function extracts seqlet instances from TF-MoDISco results and converts
    their relative positions to absolute genomic coordinates using peak region
    information.

    Parameters
    ----------
    modisco_h5_path : str
        Path to TF-MoDISco HDF5 results file containing seqlet data.
    peaks_df : pl.DataFrame
        DataFrame containing peak region information with columns:
        'peak_id', 'chr', 'chr_id', 'peak_region_start'.
    motifs_df : pl.DataFrame
        DataFrame containing motif metadata with columns:
        'motif_name_orig', 'strand', 'motif_name', 'motif_start', 'motif_end'.
    half_width : int
        Half-width of the current analysis regions.
    modisco_half_width : int
        Half-width of the regions used in the original TF-MoDISco analysis.
        Used to calculate coordinate offsets.
    lazy : bool, default False
        If True, returns a LazyFrame for efficient chaining of operations.
        If False, collects the result into a DataFrame.

    Returns
    -------
    Union[pl.DataFrame, pl.LazyFrame]
        Seqlets with genomic coordinates containing columns:
        - chr: Chromosome name
        - chr_id: Numeric chromosome identifier
        - start: Start coordinate of trimmed motif instance
        - end: End coordinate of trimmed motif instance
        - start_untrimmed: Start coordinate of full motif instance
        - end_untrimmed: End coordinate of full motif instance
        - is_revcomp: Whether the motif is reverse complemented
        - strand: Motif strand ('+' or '-')
        - motif_name: Motif name (may be remapped)
        - peak_id: Peak identifier
        - peak_region_start: Peak region start coordinate

    Notes
    -----
    Seqlets are deduplicated based on chromosome ID, start position (untrimmed),
    motif name, and reverse complement status to avoid redundant instances.

    The coordinate transformation accounts for differences in region sizes
    between the original TF-MoDISco analysis and the current analysis.
    """

    start_lst = []
    end_lst = []
    is_revcomp_lst = []
    strand_lst = []
    peak_id_lst = []
    pattern_tags = []

    with h5py.File(modisco_h5_path, "r") as modisco_results:
        for name in MODISCO_PATTERN_GROUPS:
            if name not in modisco_results.keys():
                continue

            metacluster = modisco_results[name]

            key = _motif_name_sort_key
            for _, (pattern_name, pattern) in enumerate(
                sorted(metacluster.items(), key=key)  # type: ignore  # HDF5 access
            ):
                pattern_tag = f"{name}.{pattern_name}"

                starts = pattern["seqlets/start"][:].astype(np.int32)  # type: ignore
                ends = pattern["seqlets/end"][:].astype(np.int32)  # type: ignore
                is_revcomps = pattern["seqlets/is_revcomp"][:].astype(bool)  # type: ignore
                strands = ["+" if not i else "-" for i in is_revcomps]
                peak_ids = pattern["seqlets/example_idx"][:].astype(np.uint32)  # type: ignore

                n_seqlets = int(pattern["seqlets/n_seqlets"][0])  # type: ignore

                start_lst.append(starts)
                end_lst.append(ends)
                is_revcomp_lst.append(is_revcomps)
                strand_lst.extend(strands)
                peak_id_lst.append(peak_ids)
                pattern_tags.extend([pattern_tag for _ in range(n_seqlets)])

    df_data = {
        "seqlet_start": np.concatenate(start_lst),
        "seqlet_end": np.concatenate(end_lst),
        "is_revcomp": np.concatenate(is_revcomp_lst),
        "strand": strand_lst,
        "peak_id": np.concatenate(peak_id_lst),
        "motif_name_orig": pattern_tags,
    }

    offset = half_width - modisco_half_width

    seqlets_df = (
        pl.LazyFrame(df_data)
        .join(motifs_df.lazy(), on=("motif_name_orig", "strand"), how="inner")
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .select(
            chr=pl.col("chr"),
            chr_id=pl.col("chr_id"),
            start=pl.col("peak_region_start")
            + pl.col("seqlet_start")
            + pl.col("motif_start")
            + offset,
            end=pl.col("peak_region_start")
            + pl.col("seqlet_start")
            + pl.col("motif_end")
            + offset,
            start_untrimmed=pl.col("peak_region_start")
            + pl.col("seqlet_start")
            + offset,
            end_untrimmed=pl.col("peak_region_start") + pl.col("seqlet_end") + offset,
            is_revcomp=pl.col("is_revcomp"),
            strand=pl.col("strand"),
            motif_name=pl.col("motif_name"),
            peak_id=pl.col("peak_id"),
            peak_region_start=pl.col("peak_region_start"),
        )
        .unique(subset=["chr_id", "start_untrimmed", "motif_name", "is_revcomp"])
    )

    seqlets_df = seqlets_df if lazy else seqlets_df.collect()

    return seqlets_df


def write_modisco_seqlets(
    seqlets_df: Union[pl.DataFrame, pl.LazyFrame], out_path: str
) -> None:
    """Write TF-MoDISco seqlets to TSV file.

    Parameters
    ----------
    seqlets_df : Union[pl.DataFrame, pl.LazyFrame]
        Seqlets DataFrame with genomic coordinates. Must contain columns
        that are safe to drop: 'chr_id', 'is_revcomp'.
    out_path : str
        Output TSV file path.

    Notes
    -----
    Removes internal columns 'chr_id' and 'is_revcomp' before writing
    to create a clean output format suitable for downstream analysis.
    """
    seqlets_df = seqlets_df.drop(["chr_id", "is_revcomp"])
    if isinstance(seqlets_df, pl.LazyFrame):
        seqlets_df = seqlets_df.collect()
    seqlets_df.write_csv(out_path, separator="\t")


HITS_DTYPES = {
    "chr": pl.String,
    "start": pl.Int32,
    "end": pl.Int32,
    "start_untrimmed": pl.Int32,
    "end_untrimmed": pl.Int32,
    "motif_name": pl.String,
    "hit_coefficient": pl.Float32,
    "hit_coefficient_global": pl.Float32,
    "hit_similarity": pl.Float32,
    "hit_correlation": pl.Float32,
    "hit_importance": pl.Float32,
    "hit_importance_sq": pl.Float32,
    "strand": pl.String,
    "peak_name": pl.String,
    "peak_id": pl.UInt32,
}
HITS_COLLAPSED_DTYPES = HITS_DTYPES | {"is_primary": pl.UInt32}


def load_hits(
    hits_path: str, lazy: bool = False, schema: Dict[str, Any] = HITS_DTYPES
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Load motif hit data from TSV file.

    Parameters
    ----------
    hits_path : str
        Path to TSV file containing motif hit results.
    lazy : bool, default False
        If True, returns a LazyFrame for efficient chaining operations.
        If False, collects the result into a DataFrame.
    schema : Dict[str, Any], default HITS_DTYPES
        Schema defining column names and data types for the hit data.

    Returns
    -------
    Union[pl.DataFrame, pl.LazyFrame]
        Hit data with an additional 'count' column set to 1 for aggregation.
    """
    hits_df = pl.scan_csv(
        hits_path, separator="\t", quote_char=None, schema=schema
    ).with_columns(pl.lit(1).alias("count"))

    return hits_df if lazy else hits_df.collect()


def write_hits_processed(
    hits_df: Union[pl.DataFrame, pl.LazyFrame],
    out_path: str,
    schema: Optional[Dict[str, Any]] = HITS_DTYPES,
) -> None:
    """Write processed hit data to TSV file with optional column filtering.

    Parameters
    ----------
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Hit data to write to file.
    out_path : str
        Output path for the TSV file.
    schema : Optional[Dict[str, Any]], default HITS_DTYPES
        Schema defining which columns to include in output.
        If None, all columns are written.
    """
    if schema is not None:
        hits_df = hits_df.select(schema.keys())

    if isinstance(hits_df, pl.LazyFrame):
        hits_df = hits_df.collect()

    hits_df.write_csv(out_path, separator="\t")


def write_hits(
    hits_df: Union[pl.DataFrame, pl.LazyFrame],
    peaks_df: pl.DataFrame,
    motifs_df: pl.DataFrame,
    qc_df: pl.DataFrame,
    out_dir: str,
    motif_width: int,
) -> None:
    """Write comprehensive hit results to multiple output files.

    This function combines hit data with peak, motif, and quality control information
    to generate complete output files including genomic coordinates and scores.

    Parameters
    ----------
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Hit data containing motif instance information.
    peaks_df : pl.DataFrame
        Peak region information for coordinate conversion.
    motifs_df : pl.DataFrame
        Motif metadata for annotation and trimming information.
    qc_df : pl.DataFrame
        Quality control data for normalization factors.
    out_dir : str
        Output directory for results files. Will be created if it doesn't exist.
    motif_width : int
        Width of motif instances for coordinate calculations.

    Notes
    -----
    Creates three output files:
    - hits.tsv: Complete hit data with all instances
    - hits_unique.tsv: Deduplicated hits by genomic position and motif (excludes rows with NA chromosome coordinates)
    - hits.bed: BED format file for genome browser visualization
    
    Rows where the chromosome field is NA are filtered out during deduplication
    to ensure that data_unique only contains well-defined genomic coordinates.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path_tsv = os.path.join(out_dir, "hits.tsv")
    out_path_tsv_unique = os.path.join(out_dir, "hits_unique.tsv")
    out_path_bed = os.path.join(out_dir, "hits.bed")

    data_all = (
        hits_df.lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .join(qc_df.lazy(), on="peak_id", how="inner")
        .join(motifs_df.lazy(), on="motif_id", how="inner")
        .select(
            chr_id=pl.col("chr_id"),
            chr=pl.col("chr"),
            start=pl.col("peak_region_start")
            + pl.col("hit_start")
            + pl.col("motif_start"),
            end=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_end"),
            start_untrimmed=pl.col("peak_region_start") + pl.col("hit_start"),
            end_untrimmed=pl.col("peak_region_start")
            + pl.col("hit_start")
            + motif_width,
            motif_name=pl.col("motif_name"),
            hit_coefficient=pl.col("hit_coefficient"),
            hit_coefficient_global=pl.col("hit_coefficient")
            * (pl.col("global_scale") ** 2),
            hit_similarity=pl.col("hit_similarity"),
            hit_correlation=pl.col("hit_similarity"),
            hit_importance=pl.col("hit_importance") * pl.col("global_scale"),
            hit_importance_sq=pl.col("hit_importance_sq")
            * (pl.col("global_scale") ** 2),
            strand=pl.col("strand"),
            peak_name=pl.col("peak_name"),
            peak_id=pl.col("peak_id"),
            motif_lambda=pl.col("lambda"),
        )
        .sort(["chr_id", "start"])
        .select(HITS_DTYPES.keys())
    )

    data_unique = data_all.filter(pl.col("chr").is_not_null()).unique(
        subset=["chr", "start", "motif_name", "strand"], maintain_order=True
    )

    data_bed = data_unique.select(
        chr=pl.col("chr"),
        start=pl.col("start"),
        end=pl.col("end"),
        motif_name=pl.col("motif_name"),
        score=pl.lit(0),
        strand=pl.col("strand"),
    )

    data_all.collect().write_csv(out_path_tsv, separator="\t")
    data_unique.collect().write_csv(out_path_tsv_unique, separator="\t")
    data_bed.collect().write_csv(out_path_bed, include_header=False, separator="\t")


def write_qc(qc_df: pl.DataFrame, peaks_df: pl.DataFrame, out_path: str) -> None:
    """Write quality control data with peak information to TSV file.

    Parameters
    ----------
    qc_df : pl.DataFrame
        Quality control metrics for each peak region.
    peaks_df : pl.DataFrame
        Peak region information for coordinate annotation.
    out_path : str
        Output path for the TSV file.
    """
    df = (
        qc_df.lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .sort(["chr_id", "peak_region_start"])
        .drop("chr_id")
        .collect()
    )
    df.write_csv(out_path, separator="\t")


def write_motifs_df(motifs_df: pl.DataFrame, out_path: str) -> None:
    """Write motif metadata to TSV file.

    Parameters
    ----------
    motifs_df : pl.DataFrame
        Motif metadata DataFrame.
    out_path : str
        Output path for the TSV file.
    """
    motifs_df.write_csv(out_path, separator="\t")


MOTIF_DTYPES = {
    "motif_id": pl.UInt32,
    "motif_name": pl.String,
    "motif_name_orig": pl.String,
    "strand": pl.String,
    "motif_start": pl.UInt32,
    "motif_end": pl.UInt32,
    "motif_scale": pl.Float32,
    "lambda": pl.Float32,
}


def load_motifs_df(motifs_path: str) -> Tuple[pl.DataFrame, ndarray]:
    """Load motif metadata from TSV file.

    Parameters
    ----------
    motifs_path : str
        Path to motif metadata TSV file.

    Returns
    -------
    motifs_df : pl.DataFrame
        Motif metadata with predefined schema.
    motif_names : ndarray
        Array of unique forward-strand motif names.
    """
    motifs_df = pl.read_csv(motifs_path, separator="\t", schema=MOTIF_DTYPES)
    motif_names = (
        motifs_df.filter(pl.col("strand") == "+").get_column("motif_name").to_numpy()
    )

    return motifs_df, motif_names


def write_motif_cwms(cwms: Float[ndarray, "M 4 W"], out_path: str) -> None:
    """Write motif contribution weight matrices to .npy file.

    Parameters
    ----------
    cwms : Float[ndarray, "M 4 W"]
        Contribution weight matrices for M motifs, 4 nucleotides, W width.
    out_path : str
        Output path for the .npy file.
    """
    np.save(out_path, cwms)


def load_motif_cwms(cwms_path: str) -> Float[ndarray, "M 4 W"]:
    """Load motif contribution weight matrices from .npy file.

    Parameters
    ----------
    cwms_path : str
        Path to .npy file containing CWMs.

    Returns
    -------
    Float[ndarray, "M 4 W"]
        Loaded contribution weight matrices.
    """
    return np.load(cwms_path)


def write_params(params: Dict[str, Any], out_path: str) -> None:
    """Write parameter dictionary to JSON file.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameter dictionary to serialize.
    out_path : str
        Output path for the JSON file.
    """
    with open(out_path, "w") as f:
        json.dump(params, f, indent=4)


def load_params(params_path: str) -> Dict[str, Any]:
    """Load parameter dictionary from JSON file.

    Parameters
    ----------
    params_path : str
        Path to JSON file containing parameters.

    Returns
    -------
    Dict[str, Any]
        Loaded parameter dictionary.
    """
    with open(params_path) as f:
        params = json.load(f)

    return params


def write_occ_df(occ_df: pl.DataFrame, out_path: str) -> None:
    """Write occurrence data to TSV file.

    Parameters
    ----------
    occ_df : pl.DataFrame
        Occurrence data DataFrame.
    out_path : str
        Output path for the TSV file.
    """
    occ_df.write_csv(out_path, separator="\t")


def write_seqlet_confusion_df(seqlet_confusion_df: pl.DataFrame, out_path: str) -> None:
    """Write seqlet confusion matrix data to TSV file.

    Parameters
    ----------
    seqlet_confusion_df : pl.DataFrame
        Seqlet confusion matrix DataFrame.
    out_path : str
        Output path for the TSV file.
    """
    seqlet_confusion_df.write_csv(out_path, separator="\t")


def write_report_data(
    report_df: pl.DataFrame, cwms: Dict[str, Dict[str, ndarray]], out_dir: str
) -> None:
    """Write comprehensive motif report data including CWMs and metadata.

    Parameters
    ----------
    report_df : pl.DataFrame
        Report metadata DataFrame.
    cwms : Dict[str, Dict[str, ndarray]]
        Nested dictionary of motif names to CWM types to arrays.
    out_dir : str
        Output directory for report files.
    """
    cwms_dir = os.path.join(out_dir, "CWMs")
    os.makedirs(cwms_dir, exist_ok=True)

    for m, v in cwms.items():
        motif_dir = os.path.join(cwms_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            np.savetxt(os.path.join(motif_dir, f"{cwm_type}.txt"), cwm)

    report_df.write_csv(os.path.join(out_dir, "motif_report.tsv"), separator="\t")
