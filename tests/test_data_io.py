"""Tests of region extraction, motif loading and the softmax motif type."""

import h5py
import numpy as np
import polars as pl
import pyBigWig
import pytest
from scipy.special import softmax as scipy_softmax

from finemo import data_io


def test_softmax_is_a_column_softmax():
    x = np.random.default_rng(0).random((4, 12))
    np.testing.assert_allclose(data_io.softmax(x, temp=10), scipy_softmax(10 * x, axis=0), atol=1e-12)


@pytest.fixture
def genome(tmp_path):
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1\nACGTACGTACGTACGTACGT\n>chr2\n" + "A" * 30 + "\n")
    bw_path = str(tmp_path / "c.bw")
    bw = pyBigWig.open(bw_path, "w")
    bw.addHeader([("chr1", 20), ("chr2", 30)])
    for chrom, n in (("chr1", 20), ("chr2", 30)):
        bw.addEntries([chrom] * n, list(range(n)), ends=list(range(1, n + 1)),
                      values=[float(v + 1) for v in range(n)])
    bw.close()
    return str(fa), bw_path


def test_regions_overhanging_a_chromosome_end_are_zero_padded(genome):
    fa, bw = genome
    peaks = pl.DataFrame({"chr": ["chr1", "chr1", "chr2"], "peak_region_start": [-3, 15, 10]})
    seqs, con = data_io.load_regions_from_bw(peaks, fa, [bw], 5)
    assert con[0].astype(float).tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]  # before the start
    assert seqs[0, :, :3].sum() == 0 and seqs[0, :, 3:].sum() == 7
    assert con[1].astype(float).tolist() == [16, 17, 18, 19, 20, 0, 0, 0, 0, 0]  # past the end
    assert con[2].astype(float).tolist() == list(range(11, 21))  # inside


def _modisco_h5(path, n_patterns=2, width=20, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for k in range(n_patterns):
            f.create_dataset(f"pos_patterns/pattern_{k}/contrib_scores",
                             data=rng.normal(size=(width, 4)).astype(np.float32))
    return path


@pytest.mark.parametrize("key", ["pos_patterns.pattern_0", "FOO"])
def test_custom_trims_apply_after_renaming(tmp_path, key):
    """Trims are documented by tfmodisco name; the renamed name is still accepted."""
    h5 = _modisco_h5(str(tmp_path / "m.h5"))
    names = {"pos_patterns.pattern_0": "FOO"}
    md, _, masks, _ = data_io.load_modisco_motifs(h5, {key: (3, 7)}, None, 0.3, "cwm", None, names, None, 0.7, True)
    row = md.filter((pl.col("motif_name") == "FOO") & (pl.col("strand") == "+"))
    assert (row["motif_start"][0], row["motif_end"][0]) == (3, 7)
    md, _, _, _ = data_io.load_modisco_motifs(h5, None, {key: 0.0}, 0.9, "cwm", None, names, None, 0.7, True)
    row = md.filter((pl.col("motif_name") == "FOO") & (pl.col("strand") == "+"))
    assert (row["motif_start"][0], row["motif_end"][0]) == (0, 20)  # threshold 0 keeps the whole CWM
