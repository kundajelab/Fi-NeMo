"""Post-processing utilities for Fi-NeMo hit calling results.

This module provides functions for:
- Collapsing overlapping hits based on similarity scores
- Intersecting hit sets across multiple runs
- Quality control and filtering operations

The main operations are optimized using Numba for efficient processing
of large hit datasets.
"""

import heapq
from typing import List, Union

import numpy as np
from numpy import ndarray
import polars as pl
from numba import njit
from numba.types import Array, uint32, int32, float32  # type: ignore[attr-defined]
from jaxtyping import Float, Int


@njit(
    uint32[:](
        Array(uint32, 1, "C", readonly=True),
        Array(int32, 1, "C", readonly=True),
        Array(int32, 1, "C", readonly=True),
        Array(float32, 1, "C", readonly=True),
    ),
    cache=True,
)
def _collapse_hits(
    chrom_ids: Int[ndarray, " N"],
    starts: Int[ndarray, " N"],
    ends: Int[ndarray, " N"],
    similarities: Float[ndarray, " N"],
) -> Int[ndarray, " N"]:
    """Identify primary hits among overlapping hits using a sweep line algorithm.

    This function uses a heap-based sweep line algorithm to efficiently identify
    the best hit (highest similarity) among sets of overlapping hits within each
    chromosome. Only one hit per overlapping group is marked as primary.

    Parameters
    ----------
    chrom_ids : Int[ndarray, "N"]
        Chromosome identifiers for each hit, where N is the number of hits.
        Dtype should be uint32 for Numba compatibility.
    starts : Int[ndarray, "N"]
        Start positions of hits (adjusted for overlap computation).
        Dtype should be int32 for Numba compatibility.
    ends : Int[ndarray, "N"]
        End positions of hits (adjusted for overlap computation).
        Dtype should be int32 for Numba compatibility.
    similarities : Float[ndarray, "N"]
        Similarity scores used for selecting the best hit.
        Dtype should be float32 for Numba compatibility.

    Returns
    -------
    Int[ndarray, "N"]
        Binary array where 1 indicates the hit is primary, 0 otherwise.
        Returns uint32 array for consistency with input types.

    Notes
    -----
    This function is JIT-compiled with Numba for performance on large datasets.
    The algorithm maintains active intervals in a heap and resolves overlaps
    by keeping only the hit with the highest similarity score.

    The sweep line algorithm processes hits in order and maintains a heap of
    currently active intervals. When a new interval is encountered, it is
    compared against all overlapping intervals in the heap, and only the
    interval with the highest similarity score remains marked as primary.
    """
    n = chrom_ids.shape[0]
    out = np.ones(n, dtype=np.uint32)
    heap = [(np.uint32(0), np.int32(0), -1) for _ in range(0)]

    for i in range(n):
        chrom_new = chrom_ids[i]
        start_new = starts[i]
        end_new = ends[i]
        sim_new = similarities[i]

        # Remove expired intervals from heap
        while heap and heap[0] < (chrom_new, start_new, -1):
            heapq.heappop(heap)

        # Check overlaps with active intervals
        for _, _, idx in heap:
            cmp = sim_new > similarities[idx]
            out[idx] &= not cmp  # the new hit is better: demote the old one
            out[i] &= cmp  # the new hit is not better: demote the new one

        # Add current interval to heap
        heapq.heappush(heap, (chrom_new, end_new, i))

    return out


def collapse_hits(
    hits_df: Union[pl.DataFrame, pl.LazyFrame], overlap_frac: float
) -> pl.DataFrame:
    """Collapse overlapping hits by selecting the best hit per overlapping group.

    This function identifies overlapping hits and marks only the highest-similarity
    hit as primary in each overlapping group. Overlap is determined by a fractional
    threshold based on the average length of the two hits being compared.

    Parameters
    ----------
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Hit data containing required columns: chr (or peak_id if no chr), start, end,
        hit_similarity. Will be collected to DataFrame if passed as LazyFrame.
    overlap_frac : float
        Overlap fraction threshold for considering hits as overlapping.
        For two hits with lengths x and y, minimum overlap = overlap_frac * (x + y) / 2.
        Must be between 0 and 1, where 0 means any overlap and 1 means complete overlap.

    Returns
    -------
    pl.DataFrame
        Original hit data with an additional 'is_primary' column (1 for primary hits, 0 otherwise).
        All original columns are preserved, with the new column added at the end.

    Raises
    ------
    KeyError
        If required columns (chr/peak_id, start, end, hit_similarity) are missing.

    Notes
    -----
    The algorithm transforms coordinates by scaling by 2 and adjusting by the overlap
    fraction to create effective overlap regions for efficient processing. This allows
    using a sweep line algorithm to identify overlaps in a single pass.

    The transformation works as follows:
    - Original coordinates: [start, end]
    - Length = end - start
    - Adjusted start = start * 2 + length * overlap_frac
    - Adjusted end = end * 2 - length * overlap_frac

    This creates regions that overlap only when the original regions have sufficient
    overlap according to the specified fraction.

    Examples
    --------
    >>> hits_collapsed = collapse_hits(hits_df, overlap_frac=0.2)
    >>> primary_hits = hits_collapsed.filter(pl.col("is_primary") == 1)
    >>> print(f"Kept {primary_hits.height}/{hits_df.height} hits as primary")
    """
    # Ensure we're working with a DataFrame
    if isinstance(hits_df, pl.LazyFrame):
        hits_df = hits_df.collect()

    chr_col = hits_df["chr"]
    has_coords = chr_col.null_count() == 0 and not (chr_col == "NA").any()

    if has_coords:
        chroms = chr_col.unique(maintain_order=True)
        chrom_to_id = {chrom: i for i, chrom in enumerate(chroms)}
        # Transform coordinates for overlap computation
        # Scale by 2 and adjust by overlap fraction to create effective overlap regions
        df = hits_df.select(
            chrom_id=pl.col("chr").replace_strict(chrom_to_id, return_dtype=pl.UInt32),
            start_trim=pl.col("start") * 2
            + ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            end_trim=pl.col("end") * 2
            - ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            similarity=pl.col("hit_similarity"),
        )
    else:
        # Fall back to peak_id when chr column is not available
        df = hits_df.select(
            chrom_id=pl.col("peak_id").cast(pl.UInt32),
            start_trim=pl.col("start") * 2
            + ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            end_trim=pl.col("end") * 2
            - ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            similarity=pl.col("hit_similarity"),
        )

    # The sweep needs hits ordered by (chrom_id, start_trim); remember the input order
    df = df.with_row_index("row").sort(["chrom_id", "start_trim"]).rechunk()
    chrom_ids = df["chrom_id"].to_numpy(allow_copy=False)
    starts = df["start_trim"].to_numpy(allow_copy=False)
    ends = df["end_trim"].to_numpy(allow_copy=False)
    similarities = df["similarity"].to_numpy(allow_copy=False)

    # Run the collapse algorithm, then map results back to the input row order
    is_primary_sorted = _collapse_hits(chrom_ids, starts, ends, similarities)
    is_primary = np.empty_like(is_primary_sorted)
    is_primary[df["row"].to_numpy()] = is_primary_sorted

    # Add primary indicator column to original DataFrame
    df_out = hits_df.with_columns(is_primary=pl.Series(is_primary, dtype=pl.UInt32))

    return df_out


def intersect_hits(
    hits_dfs: List[Union[pl.DataFrame, pl.LazyFrame]], relaxed: bool
) -> pl.DataFrame:
    """Intersect hit datasets across multiple runs to find common hits.

    This function finds hits that appear consistently across multiple Fi-NeMo
    runs, which can be useful for identifying robust motif instances that are
    not sensitive to parameter variations or random initialization.

    Parameters
    ----------
    hits_dfs : List[Union[pl.DataFrame, pl.LazyFrame]]
        List of hit DataFrames from different Fi-NeMo runs. Each DataFrame must
        contain the columns specified by the intersection criteria. LazyFrames
        will be collected before processing.
    relaxed : bool
        If True, uses relaxed intersection criteria with only motif names and
        untrimmed coordinates. If False, uses strict criteria including all
        coordinate and metadata columns.

    Returns
    -------
    pl.DataFrame
        DataFrame containing hits that appear in all input datasets.
        Columns from later datasets are suffixed with their index (e.g., '_1', '_2').
        The first dataset's columns retain their original names.

    Raises
    ------
    ValueError
        If fewer than one hits DataFrame is provided.
    KeyError
        If required columns for the specified intersection criteria are missing
        from any of the input DataFrames.

    Notes
    -----
    Relaxed intersection is useful when comparing results across different
    region definitions or motif trimming parameters, but may produce less
    precise matches. Strict intersection requires identical region definitions
    and is recommended for most use cases.

    The intersection columns used are:
    - Relaxed: ["chr", "start_untrimmed", "end_untrimmed", "motif_name", "strand"]
    - Strict: ["chr", "start", "end", "start_untrimmed", "end_untrimmed",
               "motif_name", "strand", "peak_name", "peak_id"]

    The function performs successive inner joins starting with the first DataFrame,
    so the final result contains only hits present in all input datasets.

    Examples
    --------
    >>> common_hits = intersect_hits([hits_df1, hits_df2], relaxed=False)
    >>> print(f"Found {common_hits.height} hits common to both runs")
    >>>
    >>> # Compare relaxed vs strict intersection
    >>> relaxed_hits = intersect_hits([hits_df1, hits_df2], relaxed=True)
    >>> strict_hits = intersect_hits([hits_df1, hits_df2], relaxed=False)
    >>> print(f"Relaxed: {relaxed_hits.height}, Strict: {strict_hits.height}")
    """
    if relaxed:
        # Relaxed criteria: only motif identity and untrimmed positions
        join_cols = ["chr", "start_untrimmed", "end_untrimmed", "motif_name", "strand"]
    else:
        # Strict criteria: all coordinate and metadata columns
        join_cols = [
            "chr",
            "start",
            "end",
            "start_untrimmed",
            "end_untrimmed",
            "motif_name",
            "strand",
            "peak_name",
            "peak_id",
        ]

    if len(hits_dfs) < 1:
        raise ValueError("At least one hits dataframe required")

    # Ensure all DataFrames are collected
    collected_dfs = []
    for df in hits_dfs:
        if isinstance(df, pl.LazyFrame):
            collected_dfs.append(df.collect())
        else:
            collected_dfs.append(df)

    # Start with first DataFrame and successively intersect with others
    hits_df = collected_dfs[0]
    for i in range(1, len(collected_dfs)):
        hits_df = hits_df.join(
            collected_dfs[i],
            on=join_cols,
            how="inner",
            suffix=f"_{i}",
            join_nulls=True,
            coalesce=True,
        )

    return hits_df


def motif_alignments(
    cwms: Float[ndarray, "M 4 W"], trim_masks: Float[ndarray, "M W"]
) -> tuple:
    """Best aligned cosine similarity between every ordered pair of trimmed motifs.

    Parameters
    ----------
    cwms : Float[ndarray, "M 4 W"]
        Motif CWMs as call-hits uses them (one row per motif and strand), untrimmed.
    trim_masks : Float[ndarray, "M W"]
        Binary trim masks (1 inside each motif's trimmed span).

    Returns
    -------
    similarity : Float[ndarray, "M M"]
        ``similarity[a, b]``: the largest cosine similarity between trimmed motifs ``a`` and ``b``
        over all relative positions, with both trimmed CWMs at unit norm (so a partial overlap
        counts only the overlapping part).
    shift : Int[ndarray, "M M"]
        ``shift[a, b]``: the offset at that maximum, as the untrimmed start of ``b`` minus the
        untrimmed start of ``a``.
    """
    import torch
    import torch.nn.functional as F

    X = torch.as_tensor(np.asarray(cwms) * np.asarray(trim_masks)[:, None, :], dtype=torch.float64)
    W = X.shape[2]
    norms = X.flatten(1).norm(dim=1).clamp(min=1e-300)
    # out[a, b, tau] = sum_x X[a, :, x + tau - (W - 1)] . X[b, :, x]: b starts tau - (W - 1) after a
    out = F.conv1d(F.pad(X, (W - 1, W - 1)), X)
    cos = out / (norms[:, None, None] * norms[None, :, None])
    best = cos.max(dim=2)
    return best.values.numpy(), (best.indices - (W - 1)).numpy().astype(np.int32)


@njit(cache=True)
def _merge_sweep(groups, starts, motif_ids, ok, shift, max_shift, reach):
    """Union placements of similar motifs at the expected offset; inputs sorted by (group, start)."""
    n = groups.shape[0]
    parent = np.arange(n)
    for i in range(n):
        j = i + 1
        while j < n and groups[j] == groups[i] and starts[j] - starts[i] <= reach:
            a = motif_ids[i]
            b = motif_ids[j]
            if a != b and ok[a, b] and abs(starts[j] - starts[i] - shift[a, b]) <= max_shift:
                ri = i
                while parent[ri] != ri:
                    ri = parent[ri]
                rj = j
                while parent[rj] != rj:
                    rj = parent[rj]
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            j += 1
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    return parent


def merge_sites(
    hits_df: Union[pl.DataFrame, pl.LazyFrame],
    motifs_df: pl.DataFrame,
    cwms: Float[ndarray, "M 4 W"],
    min_cosine: float = 0.9,
    max_shift: int = 1,
) -> pl.DataFrame:
    """Merge placements that describe the same site into one site per group.

    The optimizer can split one site's weight over two placements that explain it almost
    equally well: the two strands of a palindromic motif, or two nearly identical motifs.
    Placements in the same region are merged when their motifs' trimmed CWMs have an aligned
    cosine similarity of at least ``min_cosine`` and the placements sit at that alignment's
    offset, within ``max_shift`` bp; merging is transitive. Unlike ``collapse_hits``, which keeps
    the best placement, the merged site keeps the summed coefficient.

    Parameters
    ----------
    hits_df : Union[pl.DataFrame, pl.LazyFrame]
        Hits from call-hits (``hits.tsv``).
    motifs_df : pl.DataFrame
        ``motif_data.tsv`` of the same run (motif_id, motif_name, strand, motif_start, motif_end).
    cwms : Float[ndarray, "M 4 W"]
        ``motif_cwms.npy`` of the same run.
    min_cosine : float, default 0.9
        Minimum aligned cosine similarity between two motifs for their placements to merge.
    max_shift : int, default 1
        Tolerance in bp on the offset between two placements.

    Returns
    -------
    pl.DataFrame
        One row per site: chr, start, end (union of the members' trimmed spans), motif_name and
        strand of the member with the largest coefficient, site_coefficient and
        site_coefficient_global (sums over members), site_similarity (largest member similarity),
        n_placements, placements (``motif(strand)@start:coefficient`` for each member), peak_name
        and peak_id.
    """
    if isinstance(hits_df, pl.LazyFrame):
        hits_df = hits_df.collect()

    M, _, W = cwms.shape
    trim_masks = np.zeros((M, W), dtype=np.float64)
    ids = motifs_df["motif_id"].to_numpy()
    for k, s, e in zip(ids, motifs_df["motif_start"].to_numpy(), motifs_df["motif_end"].to_numpy()):
        trim_masks[k, s:e] = 1
    similarity, shift = motif_alignments(cwms, trim_masks)
    ok = similarity >= min_cosine

    key = motifs_df.select(
        pl.col("motif_name"), pl.col("strand"), pl.col("motif_id").cast(pl.Int32)
    )
    df = (
        hits_df.drop("count", strict=False)
        .join(key, on=["motif_name", "strand"], how="left")
        .sort(["peak_id", "start_untrimmed"], maintain_order=True)
    )
    if df["motif_id"].null_count():
        raise ValueError("Hits contain motifs that are not in the motif table")

    parent = _merge_sweep(
        df["peak_id"].to_numpy().astype(np.int64),
        df["start_untrimmed"].to_numpy().astype(np.int64),
        df["motif_id"].to_numpy().astype(np.int64),
        ok,
        shift.astype(np.int64),
        int(max_shift),
        int(W - 1 + max_shift),
    )

    member = pl.concat_str(
        [
            pl.col("motif_name"),
            pl.lit("("),
            pl.col("strand"),
            pl.lit(")@"),
            pl.col("start").cast(pl.String),
            pl.lit(":"),
            pl.col("hit_coefficient").round(4).cast(pl.String),
        ]
    )
    by_coef = dict(by="hit_coefficient", descending=True)
    sites = (
        df.with_columns(site=pl.Series(parent), member=member)
        .group_by("site", maintain_order=True)
        .agg(
            chr=pl.col("chr").first(),
            start=pl.col("start").min(),
            end=pl.col("end").max(),
            motif_name=pl.col("motif_name").sort_by(**by_coef).first(),
            strand=pl.col("strand").sort_by(**by_coef).first(),
            site_coefficient=pl.col("hit_coefficient").sum(),
            site_coefficient_global=pl.col("hit_coefficient_global").sum(),
            site_similarity=pl.col("hit_similarity").max(),
            n_placements=pl.len().cast(pl.UInt32),
            placements=pl.col("member").sort_by(**by_coef).str.join(";"),
            peak_name=pl.col("peak_name").first(),
            peak_id=pl.col("peak_id").first(),
        )
        .drop("site")
    )
    return sites
