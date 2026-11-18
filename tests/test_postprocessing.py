"""Tests of collapse-hits."""

import numpy as np
import polars as pl
import pytest

from finemo import postprocessing


def _hits(chrs, starts, ends, sims, peaks):
    return pl.DataFrame({
        "chr": chrs,
        "start": pl.Series(starts, dtype=pl.Int32),
        "end": pl.Series(ends, dtype=pl.Int32),
        "hit_similarity": pl.Series(sims, dtype=pl.Float32),
        "peak_id": pl.Series(peaks, dtype=pl.UInt32),
    })


def _brute_force(df, frac):
    """Demote the lower-similarity hit of every overlapping pair (same chromosome, or same region without one)."""
    length = df["end"] - df["start"]
    st = (df["start"] * 2 + (length * frac).cast(pl.Int32)).to_numpy()
    en = (df["end"] * 2 - (length * frac).cast(pl.Int32)).to_numpy()
    chr_col = df["chr"]
    key = (chr_col.to_numpy() if chr_col.null_count() == 0 and not (chr_col == "NA").any()
           else df["peak_id"].to_numpy())
    sim = df["hit_similarity"].to_numpy()
    out = np.ones(df.height, np.uint32)
    for i in range(df.height):
        for j in range(i + 1, df.height):
            if key[i] == key[j] and st[i] <= en[j] and st[j] <= en[i]:
                out[j if sim[j] < sim[i] else i] = 0
    return out


def test_keeps_the_higher_similarity_hit():
    d = _hits(["chr1"] * 3, [100, 105, 300], [120, 125, 320], [0.9, 0.5, 0.7], [0, 0, 0])
    assert postprocessing.collapse_hits(d, 0.2)["is_primary"].to_list() == [1, 0, 1]


def test_overlap_missed_when_start_order_differs_from_trimmed_order():
    # C[98,103) and B[102,107) overlap by the threshold; A[100,130) sits between them in start order
    d = _hits(["chr1"] * 3, [98, 100, 102], [103, 130, 107], [0.8, 0.95, 0.6], [0, 0, 0])
    assert postprocessing.collapse_hits(d, 0.2)["is_primary"].to_list() == _brute_force(d, 0.2).tolist()


@pytest.mark.parametrize("chr_value", [None, "NA"])
def test_no_comparisons_across_regions_without_coordinates(chr_value):
    d = _hits([chr_value, chr_value], [10, 10], [30, 30], [0.9, 0.8], [0, 1])
    assert postprocessing.collapse_hits(d, 0.2)["is_primary"].to_list() == [1, 1]


@pytest.mark.parametrize("shuffle", [False, True])
def test_matches_brute_force_on_random_hits(shuffle):
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 40
        s = rng.integers(0, 400, n)
        d = _hits(list(rng.choice(["chr1", "chr2"], n)), s, s + rng.integers(5, 40, n),
                  rng.permutation(n) / n, [0] * n).sort(["chr", "start"])
        if shuffle:  # input order must not matter
            d = d.sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 30)))
        got = postprocessing.collapse_hits(d, 0.2)["is_primary"].to_numpy()
        np.testing.assert_array_equal(got, _brute_force(d, 0.2))
