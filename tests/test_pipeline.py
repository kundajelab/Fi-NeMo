"""End-to-end call-hits on a synthetic modisco file, and the report's occurrence table."""

import os

import h5py
import numpy as np
import polars as pl
import pytest

from finemo import data_io, evaluation, main

from conftest import synthetic_problem


@pytest.fixture
def inputs(tmp_path):
    prob = synthetic_problem(seed=5, n=6, L=240, M=3, W=12)
    h5 = str(tmp_path / "modisco.h5")
    with h5py.File(h5, "w") as f:
        for k in range(prob["cwms"].shape[0]):
            f.create_dataset(f"pos_patterns/pattern_{k}/contrib_scores", data=prob["cwms"][k].T)
    return prob, h5, tmp_path


def _call(regions, h5, out_dir):
    main.call_hits(regions, None, h5, None, None, None, None, str(out_dir), device="cpu", batch_size=4)
    read = lambda name: pl.read_csv(os.path.join(out_dir, name), separator="\t")  # noqa: E731
    return read("hits.tsv"), read("hits_unique.tsv"), read("peaks_qc.tsv")


def test_hits_unique_excludes_regions_without_coordinates(inputs):
    prob, h5, tmp = inputs
    regions = str(tmp / "regions.npz")
    data_io.write_regions_npz(prob["sequences"], prob["contribs"], regions)
    hits, unique, qc = _call(regions, h5, tmp / "out")
    assert hits.height > 0
    assert unique.height == 0  # placeholder "NA" coordinates are not merged across regions
    assert qc.height == prob["sequences"].shape[0]


def test_hits_unique_with_coordinates(inputs):
    prob, h5, tmp = inputs
    n = prob["sequences"].shape[0]
    peaks = pl.DataFrame({"chr": ["chr1"] * n, "chr_id": np.zeros(n, np.uint32),
                          "peak_region_start": np.arange(n, dtype=np.int32) * 10_000,
                          "peak_id": np.arange(n, dtype=np.uint32), "peak_name": [f"p{i}" for i in range(n)]})
    regions = str(tmp / "regions.npz")
    data_io.write_regions_npz(prob["sequences"], prob["contribs"], regions, peaks_df=peaks)
    hits, unique, _ = _call(regions, h5, tmp / "out")
    assert unique.height == hits.height  # regions are disjoint: nothing to deduplicate


def test_occurrences_include_regions_without_hits():
    hits = pl.DataFrame({"peak_id": pl.Series([0, 0, 2], dtype=pl.UInt32), "motif_name": ["m1", "m2", "m1"]})
    occ, co = evaluation.get_motif_occurences(hits.lazy(), ["m3", "m1", "m2"], np.arange(4, dtype=np.uint32))
    assert occ["peak_id"].to_list() == [0, 1, 2, 3]
    assert occ["total"].to_list() == [2, 0, 1, 0]
    assert co.shape == (3, 3)
