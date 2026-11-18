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


W = 10


def _onehot(seq, start):
    m = np.zeros((4, W))
    for k, b in enumerate(seq):
        m["ACGT".index(b), start + k] = 1.0
    return m / np.linalg.norm(m)


def _motifs():
    """P: palindrome ACGCGT; R: GATAAG; Q: R one bp further left in its window; Z: unrelated."""
    fwd = {"P": (_onehot("ACGCGT", 2), 2, 8), "R": (_onehot("GATAAG", 2), 2, 8),
           "Q": (_onehot("GATAAG", 1), 1, 7), "Z": (_onehot("TTCTCT", 2), 2, 8)}
    rows, cwms = [], []
    for name, (m, s, e) in fwd.items():
        for strand, mat, ss, ee in (("+", m, s, e), ("-", m[::-1, ::-1], W - e, W - s)):
            rows.append({"motif_id": len(rows), "motif_name": name, "strand": strand, "motif_start": ss,
                         "motif_end": ee})
            cwms.append(mat)
    return pl.DataFrame(rows).with_columns(pl.col("motif_id").cast(pl.UInt32)), np.stack(cwms)


def _site_hits(rows):
    md, _ = _motifs()
    trim = {(r["motif_name"], r["strand"]): (r["motif_start"], r["motif_end"]) for r in md.iter_rows(named=True)}
    out = []
    for peak, name, strand, u, coef in rows:
        s, e = trim[(name, strand)]
        out.append({"chr": "chr1", "start": u + s, "end": u + e, "start_untrimmed": u, "end_untrimmed": u + W,
                    "motif_name": name, "hit_coefficient": coef, "hit_coefficient_global": 10 * coef,
                    "hit_similarity": 0.9, "hit_correlation": 0.9, "hit_importance": 1.0,
                    "hit_importance_sq": 1.0, "strand": strand, "peak_name": f"p{peak}", "peak_id": peak})
    return pl.DataFrame(out).with_columns(pl.col("peak_id").cast(pl.UInt32))


def test_motif_alignments():
    md, cwms = _motifs()
    masks = np.zeros((len(cwms), W))
    for r in md.iter_rows(named=True):
        masks[r["motif_id"], r["motif_start"]:r["motif_end"]] = 1
    sim, shift = postprocessing.motif_alignments(cwms, masks)
    ids = {(r["motif_name"], r["strand"]): r["motif_id"] for r in md.iter_rows(named=True)}
    p, pm, r_, q, z = ids["P", "+"], ids["P", "-"], ids["R", "+"], ids["Q", "+"], ids["Z", "+"]
    assert sim[p, pm] == pytest.approx(1.0) and shift[p, pm] == 0  # a palindrome matches its reverse complement
    assert sim[r_, q] == pytest.approx(1.0) and shift[r_, q] == 1  # same site: Q's window starts 1 bp after R's
    assert shift[q, r_] == -1
    assert sim[r_, z] < 0.5


def test_merge_sites():
    md, cwms = _motifs()
    hits = _site_hits([
        (0, "P", "+", 100, 2.0), (0, "P", "-", 100, 1.0),   # both strands of a palindrome
        (0, "R", "+", 200, 5.0), (0, "Q", "+", 201, 3.0),   # near-identical motifs at their offset
        (0, "R", "+", 300, 1.0), (0, "Q", "+", 305, 1.0),   # the same pair at the wrong offset
        (0, "Z", "+", 400, 1.0), (0, "R", "+", 401, 1.0),   # overlapping but unrelated
        (1, "P", "+", 100, 4.0),                            # another region
    ])
    sites = postprocessing.merge_sites(hits, md, cwms, min_cosine=0.9, max_shift=1)
    assert sites.height == 7
    assert sites["site_coefficient"].sum() == pytest.approx(hits["hit_coefficient"].sum())
    pal = sites.filter((pl.col("peak_id") == 0) & (pl.col("motif_name") == "P"))
    assert pal["n_placements"].to_list() == [2] and pal["site_coefficient"].to_list() == [3.0]
    assert pal["strand"].to_list() == ["+"]  # labelled by the larger member
    rq = sites.filter(pl.col("n_placements") == 2).filter(pl.col("motif_name") == "R")
    assert rq["site_coefficient"].to_list() == [8.0] and rq["site_coefficient_global"].to_list() == [80.0]
    assert rq["start"].to_list() == [202] and rq["end"].to_list() == [208]  # union of trimmed spans
    assert sites.filter(pl.col("peak_id") == 1).height == 1
    # above any attainable similarity nothing merges
    assert postprocessing.merge_sites(hits, md, cwms, min_cosine=1.01, max_shift=1).height == hits.height
