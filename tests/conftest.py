"""Shared helpers for the Fi-NeMo tests: small synthetic problems and an independent float64 reference.

The reference re-implements the problem that ``hitcaller.fit_contribs`` solves (projected or hypothetical
scores over their RMS, trimmed CWMs at unit norm, importance scale ``(window energy + 1) ** -0.5`` capped at 10)
so that tests can compare the solver against the true optimum instead of against itself.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def synthetic_problem(seed=0, n=6, L=240, M=4, W=12, sites=3, noise=0.05, trim=True, lam=0.7):
    """Random one-hot sequences with planted, scaled CWM instances plus Gaussian noise.

    Returns a dict with ``cwms`` (M, 4, W), ``mask`` (M, W), ``sequences`` (n, 4, L), hypothetical
    ``contribs`` (n, 4, L) and ``lambdas`` (M,).
    """
    rng = np.random.default_rng(seed)
    cwms = rng.normal(0.0, 0.1, (M, 4, W))
    for m in range(M):
        cwms[m, rng.integers(0, 4, W), np.arange(W)] += 1.0
    cwms /= np.sqrt((cwms**2).sum(axis=(1, 2), keepdims=True))
    mask = np.zeros((M, W), np.float32)
    if trim:
        mask[:, 1 : W - 1] = 1
    else:
        mask[:] = 1

    seqs = np.zeros((n, 4, L), np.int8)
    seqs[np.arange(n)[:, None], rng.integers(0, 4, (n, L)), np.arange(L)[None, :]] = 1
    contribs = rng.normal(0.0, noise, (n, 4, L))
    for r in range(n):
        starts = rng.choice(np.arange(0, L - W, W + 2), size=sites, replace=False)
        for p in starts:
            m = rng.integers(M)
            contribs[r, :, p : p + W] += rng.uniform(1.0, 3.0) * cwms[m]
            seqs[r, :, p : p + W] = 0
            seqs[r, cwms[m].argmax(axis=0), np.arange(p, p + W)] = 1
    return {
        "cwms": cwms.astype(np.float32),
        "mask": mask,
        "sequences": seqs,
        "contribs": contribs.astype(np.float32),
        "lambdas": np.full(M, lam, np.float32),
    }


class Reference:
    """float64 version of the problem fit_contribs solves, for independent objectives, gaps and optima."""

    def __init__(self, prob, use_hypothetical=False, lambdas=None):
        D = torch.float64
        seq = torch.as_tensor(prob["sequences"], dtype=D)
        X = torch.as_tensor(prob["contribs"], dtype=D)
        if X.dim() == 2:
            X = X[:, None, :]
        if use_hypothetical:
            self.seq = 1.0
        else:
            X = X * seq
            self.seq = seq
        self.L = X.shape[-1]
        gs = ((X**2).sum(dim=(1, 2)) / self.L).sqrt()
        self.X = torch.nan_to_num(X / gs[:, None, None])
        M4 = torch.as_tensor(prob["mask"], dtype=D)[:, None, :].repeat(1, 4, 1)
        Wm = torch.as_tensor(prob["cwms"], dtype=D) * M4
        self.W = Wm / Wm.norm(dim=(1, 2), keepdim=True)
        self.M4 = M4
        self.s = ((F.conv1d(self.X**2, M4) + 1.0) ** -0.5).clamp(max=10)
        lam = prob["lambdas"] if lambdas is None else lambdas
        self.lam = torch.as_tensor(np.asarray(lam), dtype=D)[None, :, None]
        self.shape = (self.X.shape[0], self.W.shape[0], self.L - self.W.shape[2] + 1)

    def A(self, c):
        return F.conv_transpose1d(c * self.s, self.W) * self.seq

    def At(self, r):
        return F.conv1d(r, self.W) * self.s

    def objective_and_gap(self, c):
        """Per-bp objective P(c) and valid duality gap at c (both per region, numpy)."""
        c = torch.as_tensor(c, dtype=torch.float64)
        r = self.X - self.A(c)
        g = self.At(r)
        t = 1.0 / (g / self.lam).amax(dim=(1, 2)).clamp(min=1.0)
        rr = (r**2).sum(dim=(1, 2))
        ry = (r * self.X).sum(dim=(1, 2))
        l1 = (c * self.lam).sum(dim=(1, 2))
        P = 0.5 * rr + l1
        G = 0.5 * (1 + t**2) * rr - t * ry + l1
        return (P / self.L).numpy(), (G / self.L).numpy()

    def solve(self, tol=1e-10, max_iter=50000):
        """FISTA with step 1/L (power iteration) and function-value restart, to a valid gap below tol."""
        v = torch.randn(self.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        for _ in range(100):
            v = self.At(self.A(v))
            v = v / v.norm(dim=(1, 2), keepdim=True).clamp(min=1e-300)
        lip = (self.At(self.A(v)).norm(dim=(1, 2)) / v.norm(dim=(1, 2))).max().item()
        step = 1.0 / lip
        c = torch.zeros(self.shape, dtype=torch.float64)
        y, t = c.clone(), 1.0
        P_old = np.full(self.shape[0], np.inf)
        for k in range(max_iter):
            c_new = F.relu(y + step * (self.At(self.X - self.A(y)) - self.lam))
            P_new, G = self.objective_and_gap(c_new)
            if (P_new > P_old + 1e-15).any():
                y, t = c.clone(), 1.0
                continue
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = c_new + (t - 1) / t_new * (c_new - c)
            c, t, P_old = c_new, t_new, P_new
            if G.max() < tol:
                break
        return c.numpy(), P_old


def dense_coefficients(hits_df, shape):
    """(n, M, P) coefficients from a hits table (post_filter off keeps every nonzero coefficient)."""
    c = np.zeros(shape)
    if hits_df.height:
        c[hits_df["peak_id"].to_numpy(), hits_df["motif_id"].to_numpy(), hits_df["hit_start"].to_numpy()] = (
            hits_df["hit_coefficient"].to_numpy()
        )
    return c


@pytest.fixture
def problem():
    return synthetic_problem()


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def default_reference():
    """Reference and float64 optimum (per bp, per region) of the default synthetic problem, solved once."""
    ref = Reference(synthetic_problem())
    _, P_star = ref.solve()
    return ref, P_star
