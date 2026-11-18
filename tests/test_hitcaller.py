"""Tests of the optimizer: the convergence certificate, the optimum it reaches, and the fitting loop."""

import os

import numpy as np
import polars as pl
import pytest
import torch

from finemo import hitcaller

from conftest import Reference, dense_coefficients, synthetic_problem


def _tensors(prob, ref):
    f = torch.float32
    return {
        "importance_scale": ref.s.to(f),
        "cwms": ref.W.to(f),
        "contribs": ref.X.to(f),
        "sequences": ref.seq.to(f),
        "lambdas": ref.lam.to(f),
    }


def test_prox_grad_step_returns_the_valid_duality_gap(problem):
    """The gap must be the duality gap at the returned (saved) coefficients, not at the momentum point."""
    ref = Reference(problem)
    t = _tensors(problem, ref)
    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 0.5, ref.shape) * (rng.random(ref.shape) < 0.02)  # sparse, with negative entries
    c_next, gap, nll = hitcaller.prox_grad_step(
        torch.as_tensor(z, dtype=torch.float32), t["importance_scale"], t["cwms"], t["contribs"],
        t["sequences"], t["lambdas"], torch.full((ref.shape[0], 1, 1), 0.5),
    )
    assert (c_next >= 0).all()
    P, G = ref.objective_and_gap(c_next.double())
    np.testing.assert_allclose(gap.numpy() / ref.L, G, rtol=1e-4, atol=1e-6)
    assert (gap.numpy() >= -1e-6).all()
    r = ref.X - ref.A(c_next.double())
    np.testing.assert_allclose(nll.numpy(), (r**2).sum(dim=(1, 2)).numpy(), rtol=1e-4)


def test_gap_is_one_when_every_correlation_is_negative():
    """t must be 1, not 1/q, when max(A'r / lambda) < 0 (an all-negative region: c = 0 is optimal)."""
    prob = synthetic_problem(seed=2, n=2)
    prob["cwms"] = np.abs(prob["cwms"])
    prob["contribs"] = -np.abs(prob["contribs"])
    ref = Reference(prob)
    t = _tensors(prob, ref)
    c_next, gap, _ = hitcaller.prox_grad_step(
        torch.zeros(ref.shape), t["importance_scale"], t["cwms"], t["contribs"], t["sequences"],
        t["lambdas"], torch.full((ref.shape[0], 1, 1), 1.0),
    )
    assert float(c_next.abs().max()) == 0.0
    np.testing.assert_allclose(gap.numpy(), 0.0, atol=1e-5)


@pytest.mark.parametrize("use_hypothetical", [False, True])
def test_fit_reaches_the_reference_optimum(problem, cpu, use_hypothetical):
    """At convergence the objective is within the tolerance of a float64 reference optimum."""
    tol = 1e-5
    ref = Reference(problem, use_hypothetical=use_hypothetical)
    _, P_star = ref.solve()
    hits, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], use_hypothetical,
        problem["lambdas"], convergence_tol=tol, post_filter=False, device=cpu, batch_size=4,
    )
    c = dense_coefficients(hits, ref.shape)
    P, G = ref.objective_and_gap(c)
    qc = qc.sort("peak_id")
    assert (P - P_star <= tol + 1e-6).all(), P - P_star
    assert (qc["dual_gap"].to_numpy() <= tol).all()
    # the reported gap is a valid bound: it is the gap of the coefficients that were saved
    np.testing.assert_allclose(qc["dual_gap"].to_numpy(), G, rtol=1e-3, atol=1e-6)


def test_all_negative_region_stops_at_once(cpu):
    prob = synthetic_problem(seed=3, n=2)
    prob["cwms"] = np.abs(prob["cwms"])
    prob["contribs"] = -np.abs(prob["contribs"])
    hits, qc = hitcaller.fit_contribs(
        prob["cwms"], prob["contribs"], prob["sequences"], prob["mask"], False, prob["lambdas"],
        max_steps=500, device=cpu, batch_size=2, check_every=1,
    )
    assert hits.height == 0
    assert (qc["num_steps"].to_numpy() <= 2).all()


def test_compile_optimizer_does_not_rebind_the_module_function(problem, cpu, monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(fn)
        return lambda *args, **kw: fn(*args, **kw)  # a distinct object, as torch.compile returns

    monkeypatch.setattr(torch, "compile", fake_compile)
    original = hitcaller.optimizer_step
    hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], max_steps=20, device=cpu, compile_optimizer=True, batch_size=4,
    )
    assert calls and hitcaller.optimizer_step is original


FIXTURE = os.environ.get("FINEMO_TEST_FIXTURE")


@pytest.mark.skipif(not FIXTURE, reason="set FINEMO_TEST_FIXTURE to the SPI1 ENCSR000BGQ region 24734 fixture")
def test_spi1_region_that_stopped_early(cpu):
    """The region where Fi-NeMo 0.30/0.41 stopped at 3.3x the optimum (249 placements against 6)."""
    import polars as pl

    from finemo import data_io

    seqs, con, _, _ = data_io.load_regions_npz(os.path.join(FIXTURE, "region_spi1_bgq_24734.npz"))
    cwms = np.load(os.path.join(FIXTURE, "motif_cwms.npy")).astype(np.float32)
    md = pl.read_csv(os.path.join(FIXTURE, "motif_data.tsv"), separator="\t").sort("motif_id")
    mask = np.zeros(cwms.shape[::2], np.float32)
    for r in md.iter_rows(named=True):
        mask[r["motif_id"], r["motif_start"] : r["motif_end"]] = 1
    prob = {"cwms": cwms, "mask": mask, "sequences": seqs, "contribs": con,
            "lambdas": md["lambda"].to_numpy().astype(np.float32)}
    ref = Reference(prob)
    hits, qc = hitcaller.fit_contribs(cwms, con, seqs, mask, False, prob["lambdas"], post_filter=False,
                                      device=cpu, batch_size=1)
    P, _ = ref.objective_and_gap(dense_coefficients(hits, ref.shape))
    assert P[0] - 0.4702005 <= 5e-5 + 1e-6  # float64 optimum of this problem (unit-norm trimmed CWMs)
    assert hits.height == 8


def test_qc_counts_steps_across_restarts(problem, cpu):
    """A step far too large forces restarts; num_steps restarts at 0 but num_steps_total does not."""
    _, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], step_size_max=60.0, device=cpu, batch_size=4,
    )
    restarted = qc["num_restarts"].to_numpy() > 0
    assert restarted.any()
    total, since = qc["num_steps_total"].to_numpy(), qc["num_steps"].to_numpy()
    assert (total[restarted] > since[restarted]).all()
    assert (total[~restarted] == since[~restarted]).all()


def test_qc_counts_placements_removed_by_the_post_filter(problem, cpu):
    args = (problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False, problem["lambdas"])
    hits_on, qc_on = hitcaller.fit_contribs(*args, post_filter=True, device=cpu, batch_size=4)
    hits_off, qc_off = hitcaller.fit_contribs(*args, post_filter=False, device=cpu, batch_size=4)
    assert (qc_off["num_filtered"].to_numpy() == 0).all()
    n_on = hits_on.group_by("peak_id").len().rename({"len": "on"})
    n_off = hits_off.group_by("peak_id").len().rename({"len": "off"})
    counts = qc_on.select("peak_id", "num_filtered").join(n_off, on="peak_id", how="left").join(
        n_on, on="peak_id", how="left").fill_null(0)
    assert (counts["num_filtered"] == counts["off"] - counts["on"]).all()


def test_stall_safeguard_keeps_the_optimum(problem, cpu, default_reference):
    """Forced reductions (every 10 steps after 20) still converge to the optimum, and are counted."""
    tol = 1e-5
    ref, P_star = default_reference
    hits, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], convergence_tol=tol, post_filter=False, device=cpu, batch_size=4,
        stall_start=20, stall_window=10, stall_improvement=1.0,
    )
    qc = qc.sort("peak_id")
    slow = qc["num_steps_total"].to_numpy() > 20
    assert slow.any() and (qc["num_step_reductions"].to_numpy()[slow] > 0).all()
    P, _ = ref.objective_and_gap(dense_coefficients(hits, ref.shape))
    assert (P - P_star <= tol + 1e-6).all()


def test_stall_safeguard_is_idle_on_converging_regions(problem, cpu):
    _, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], device=cpu, batch_size=4,
    )
    assert (qc["num_step_reductions"].to_numpy() == 0).all()


def test_max_steps_counts_restarts(problem, cpu):
    """The step budget is per region, restarts included, so restarts cannot extend it."""
    _, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], step_size_max=60.0, max_steps=30, convergence_tol=1e-12, device=cpu, batch_size=4,
        check_every=1,
    )
    assert (qc["num_steps_total"].to_numpy() <= 31).all()


def test_early_restart_catches_divergence_before_overflow(problem, cpu, default_reference):
    """With a step far too large, restarting once the fit is worse than empty saves the steps spent overflowing."""
    tol = 1e-5
    args = (problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False, problem["lambdas"])
    ref, P_star = default_reference
    qcs = {}
    for step in (3.0, 60.0):
        for early in (False, True):
            hits, qc = hitcaller.fit_contribs(*args, step_size_max=step, convergence_tol=tol, post_filter=False,
                                              device=cpu, batch_size=4, early_restart=early)
            P, _ = ref.objective_and_gap(dense_coefficients(hits, ref.shape))
            assert (P - P_star <= tol + 1e-6).all()
            qcs[step, early] = qc.sort("peak_id")
    # at the default step nothing diverges and the rule never fires
    assert qcs[3.0, True]["num_steps_total"].to_list() == qcs[3.0, False]["num_steps_total"].to_list()
    # at a step far too large it restarts the same regions as often, but sooner
    late, early = qcs[60.0, False], qcs[60.0, True]
    assert (late["num_restarts"].to_numpy() > 0).any()
    assert early["num_restarts"].to_list() == late["num_restarts"].to_list()
    assert early["num_steps_total"].sum() < late["num_steps_total"].sum()


def test_prox_grad_step_can_skip_the_gap(problem):
    ref = Reference(problem)
    t = _tensors(problem, ref)
    z = torch.as_tensor(np.random.default_rng(2).random(ref.shape) * 0.01, dtype=torch.float32)
    step = torch.full((ref.shape[0], 1, 1), 0.5)
    args = (z, t["importance_scale"], t["cwms"], t["contribs"], t["sequences"], t["lambdas"], step)
    c_with, gap, _ = hitcaller.prox_grad_step(*args)
    c_without, no_gap, no_nll = hitcaller.prox_grad_step(*args, compute_gap=False)
    assert no_gap is None and no_nll is None and gap is not None
    assert torch.equal(c_with, c_without)


@pytest.mark.parametrize("check_every", [5, 20])
def test_sparse_checks_reach_the_optimum(problem, cpu, default_reference, check_every):
    tol = 1e-5
    ref, P_star = default_reference
    hits, qc = hitcaller.fit_contribs(
        problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
        problem["lambdas"], convergence_tol=tol, post_filter=False, device=cpu, batch_size=4,
        check_every=check_every,
    )
    P, G = ref.objective_and_gap(dense_coefficients(hits, ref.shape))
    assert (P - P_star <= tol + 1e-6).all()
    assert (qc["num_steps_total"].to_numpy() % check_every == 0).all()
    np.testing.assert_allclose(qc.sort("peak_id")["dual_gap"].to_numpy(), G, rtol=1e-3, atol=1e-6)


def test_check_every_must_divide_the_stall_window(problem, cpu):
    with pytest.raises(ValueError):
        hitcaller.fit_contribs(
            problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
            problem["lambdas"], device=cpu, check_every=7,
        )


@pytest.mark.parametrize("allow", [True, False])
def test_tf32_setting_is_restored(problem, cpu, allow):
    before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    seen = []
    original = hitcaller.prox_grad_step

    def spy(*args, **kwargs):
        seen.append(torch.backends.cudnn.allow_tf32)
        return original(*args, **kwargs)

    hitcaller.prox_grad_step = spy
    try:
        hitcaller.fit_contribs(problem["cwms"], problem["contribs"], problem["sequences"], problem["mask"], False,
                               problem["lambdas"], max_steps=10, device=cpu, batch_size=4, allow_tf32=allow)
    finally:
        hitcaller.prox_grad_step = original
    assert seen and all(v == allow for v in seen)
    assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == before
