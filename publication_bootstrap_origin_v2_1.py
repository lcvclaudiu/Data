#!/usr/bin/env python3
"""Autocorrelation-aware block bootstrap for STITCH umbrella/MBAR data.

Designed for publication statistics and OriginPro export.

Method
------
1. Each original umbrella trajectory is kept as an independent time series.
2. A conservative circular moving-block bootstrap resamples within each trajectory.
3. MBAR is refit for every bootstrap replicate (so umbrella normalization
   uncertainty is propagated).
4. Target chemical potential is re-solved by equal phase weight in each
   bootstrap replicate.
5. Expensive fermionic grand potentials are precomputed once on a fine local
   chemical-potential grid and then interpolated only inside a single grid cell.

The output is plain columnar .dat text suitable for OriginPro, plus a compact
JSON metadata file for reproducibility.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict

import numpy as np
from pymbar import MBAR, timeseries

_BOOT_CTX = None

def _init_boot_worker(ctx):
    """Initialize bootstrap context in a spawn-created worker process."""
    global _BOOT_CTX
    _BOOT_CTX = ctx



def replica_from_name(path):
    m = re.search(r"_r(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def safe_g(x):
    x = np.asarray(x, dtype=float)
    if x.size < 4 or np.all(x == x[0]):
        return 1.0
    try:
        g = float(timeseries.statistical_inefficiency(x, fast=False))
        if not np.isfinite(g) or g < 1.0:
            return 1.0
        return g
    except Exception:
        return float("nan")


def bias_vec(nh, center, kappa, N):
    x = nh / N
    xc = center / N
    return 0.5 * kappa * N * (x - xc) ** 2


def omega_from_sv(sv, T, mu, chunk=2048):
    M = sv.shape[0]
    out = np.empty(M, dtype=np.float64)
    for s in range(0, M, chunk):
        x = sv[s:s + chunk]
        out[s:s + len(x)] = -T * (
            np.logaddexp(0.0, -(x - mu) / T)
            + np.logaddexp(0.0, -(-x - mu) / T)
        ).sum(axis=1)
    return out


def logsumexp_cols(A):
    mx = np.max(A, axis=0)
    return mx + np.log(np.sum(np.exp(A - mx), axis=0))


def mbar_logden(nh, Om0, centers, kappas, N_k, T, N):
    K = len(centers)
    M = len(nh)
    u_kn = np.empty((K, M), dtype=np.float64)
    for k in range(K):
        u_kn[k] = (Om0 + bias_vec(nh, centers[k], kappas[k], N)) / T
    mbar = MBAR(
        u_kn, N_k, verbose=False,
        relative_tolerance=1e-10, maximum_iterations=10000
    )
    f_k = np.asarray(mbar.f_k, dtype=float)
    A = np.vstack([
        np.log(N_k[k]) + f_k[k] - u_kn[k]
        for k in range(K) if N_k[k] > 0
    ])
    return logsumexp_cols(A)


def normalized_weights_from_omega(omega, T, logden):
    lw = -omega / T - logden
    mx = np.max(lw)
    w = np.exp(lw - mx)
    w /= np.sum(w)
    return w


def phase_balance_from_omega(omega, T, logden, mvals):
    w = normalized_weights_from_omega(omega, T, logden)
    return float(w[mvals > 0].sum() - w[mvals < 0].sum())


def solve_grid_root_indexed(omega_grid_all, sel, mu_grid, T, logden, mvals):
    """Locate equal-phase-weight root using exact precomputed grid columns.

    Only the final interval is linearly interpolated. With the recommended
    5e-5 grid spacing this interpolation error is much smaller than the MC
    uncertainty targeted here.
    """
    G = len(mu_grid)
    cache = {}

    def col(j):
        return omega_grid_all[sel, j]

    def bal(j):
        if j not in cache:
            cache[j] = phase_balance_from_omega(
                col(j), T, logden, mvals
            )
        return cache[j]

    b0 = bal(0)
    b1 = bal(G - 1)
    if b0 == 0.0:
        return float(mu_grid[0]), col(0), 0, 0, b0, b0
    if b1 == 0.0:
        return float(mu_grid[-1]), col(G - 1), G - 1, G - 1, b1, b1
    if b0 * b1 > 0:
        raise RuntimeError(
            f"bootstrap equal-weight root not bracketed by grid: {b0}, {b1}"
        )

    lo, hi = 0, G - 1
    # Assumes the physically observed monotonic crossing in the chosen local grid.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        bm = bal(mid)
        if bm == 0.0:
            return float(mu_grid[mid]), col(mid), mid, mid, bm, bm
        if b0 * bm <= 0:
            hi = mid
            b1 = bm
        else:
            lo = mid
            b0 = bm

    mu_lo, mu_hi = float(mu_grid[lo]), float(mu_grid[hi])
    # Linear interpolation of the balance to get the root location.
    if b1 == b0:
        alpha = 0.5
    else:
        alpha = -b0 / (b1 - b0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mu = mu_lo + alpha * (mu_hi - mu_lo)
    omega = (1.0 - alpha) * col(lo) + alpha * col(hi)
    return mu, omega, lo, hi, b0, b1


def distribution_metrics(nH, w, N, T):
    P = np.bincount(nH.astype(int), weights=w, minlength=N + 1).astype(float)
    s = P.sum()
    if s > 0:
        P /= s
    mgrid = 2.0 * np.arange(N + 1) / N - 1.0

    left_idx = np.where(mgrid < 0)[0]
    right_idx = np.where(mgrid > 0)[0]
    peak_left = int(left_idx[np.argmax(P[left_idx])]) if len(left_idx) else -1
    peak_right = int(right_idx[np.argmax(P[right_idx])]) if len(right_idx) else -1

    delta_m_peak = float("nan")
    valley_idx = -1
    barrier = float("nan")
    bimodal_flag = 0
    if peak_left >= 0 and peak_right >= 0 and peak_right > peak_left:
        delta_m_peak = float(mgrid[peak_right] - mgrid[peak_left])
        interior = np.arange(peak_left + 1, peak_right)
        if len(interior):
            # Choose minimum *probability* between the two side peaks, but only
            # among bins that were actually sampled. Zero-probability holes are
            # not treated as infinite physical barriers.
            nz = interior[P[interior] > 0]
            if len(nz):
                valley_idx = int(nz[np.argmin(P[nz])])
                pl, pr, pv = P[peak_left], P[peak_right], P[valley_idx]
                if pl > 0 and pr > 0 and pv > 0:
                    barrier = float(-T * np.log(pv / math.sqrt(pl * pr)))
                    if pv < min(pl, pr) and barrier > 0:
                        bimodal_flag = 1

    return {
        "P": P,
        "mgrid": mgrid,
        "peak_left_nH": peak_left,
        "peak_right_nH": peak_right,
        "valley_nH": valley_idx,
        "delta_m_peak": delta_m_peak,
        "barrier": barrier,
        "bimodal_flag": bimodal_flag,
    }


def observables_from_root(nH, omega, T, logden, N):
    w = normalized_weights_from_omega(omega, T, logden)
    m = 2.0 * nH / N - 1.0
    mean = lambda x: float(np.sum(w * x))
    m1 = mean(m)
    m2 = mean(m * m)
    m4 = mean(m ** 4)
    binder = 1.0 - m4 / (3.0 * m2 * m2)
    chi = N / T * (m2 - m1 * m1)
    b2 = np.where(nH > 0, nH, 1)
    b1 = np.where(nH > 0, 2 * N + 1 + nH, 2 * N + 2)
    dm = distribution_metrics(nH, w, N, T)
    return {
        "mean_abs_m": mean(np.abs(m)),
        "mean_m": m1,
        "binder": float(binder),
        "chi_m": float(chi),
        "chi_m_T_over_N": float(chi * T / N),
        "mean_b1_over_N": mean(b1 / N),
        "mean_b2_over_N": mean(b2 / N),
        "phase_weight_pos": float(w[m > 0].sum()),
        "phase_weight_neg": float(w[m < 0].sum()),
        "phase_weight_zero": float(w[m == 0].sum()),
        "P": dm["P"],
        "mgrid": dm["mgrid"],
        "peak_left_nH": dm["peak_left_nH"],
        "peak_right_nH": dm["peak_right_nH"],
        "valley_nH": dm["valley_nH"],
        "delta_m_peak": dm["delta_m_peak"],
        "barrier": dm["barrier"],
        "bimodal_flag": dm["bimodal_flag"],
    }


def circular_block_indices(rng, start, length, block_length):
    """Return global indices for one circular moving-block bootstrap trajectory."""
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    B = max(1, min(int(block_length), int(length)))
    nblocks = int(math.ceil(length / B))
    starts = rng.integers(0, length, size=nblocks)
    local = np.empty(nblocks * B, dtype=np.int64)
    p = 0
    base = np.arange(B, dtype=np.int64)
    for s in starts:
        local[p:p + B] = (s + base) % length
        p += B
    local = local[:length]
    return start + local


def bootstrap_worker(iboot):
    c = _BOOT_CTX
    rng = np.random.default_rng(c["seed"] + 104729 * int(iboot))

    # Resample each independent trajectory, then regroup by MBAR umbrella state.
    selected_by_state = defaultdict(list)
    for tr in c["trajectories"]:
        idx = circular_block_indices(
            rng, tr["start"], tr["length"], c["block_length"]
        )
        selected_by_state[tr["state"]].append(idx)

    selected_parts = []
    N_k = []
    for k in range(len(c["centers"])):
        parts = selected_by_state[k]
        idxk = np.concatenate(parts)
        selected_parts.append(idxk)
        N_k.append(len(idxk))
    sel = np.concatenate(selected_parts)
    N_k = np.asarray(N_k, dtype=int)

    nH = c["nH"][sel]
    Om0 = c["Om0"][sel]
    mvals = 2.0 * nH / c["N"] - 1.0

    logden = mbar_logden(
        nH, Om0, c["centers"], c["kappas"], N_k, c["T"], c["N"]
    )
    try:
        mu_eq, omega, _, _, _, _ = solve_grid_root_indexed(
            c["omega_grid"], sel, c["mu_grid"], c["T"], logden, mvals
        )
        obs = observables_from_root(nH, omega, c["T"], logden, c["N"])
        row = {
            "boot": int(iboot), "ok": 1, "mu_eq": float(mu_eq),
            **{k: v for k, v in obs.items() if k not in ("P", "mgrid")},
            "P": obs["P"],
        }
    except Exception as e:
        row = {"boot": int(iboot), "ok": 0, "error": str(e), "P": None}
    return row


def write_dat(path, header, rows, comments=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if comments:
            for line in comments:
                f.write(f"# {line}\n")
        f.write("\t".join(header) + "\n")
        for row in rows:
            vals = []
            for x in row:
                if isinstance(x, str):
                    vals.append(x)
                elif x is None:
                    vals.append("nan")
                else:
                    try:
                        xx = float(x)
                        if np.isnan(xx): vals.append("nan")
                        elif np.isposinf(xx): vals.append("inf")
                        elif np.isneginf(xx): vals.append("-inf")
                        else: vals.append(f"{xx:.16g}")
                    except Exception:
                        vals.append(str(x))
            f.write("\t".join(vals) + "\n")


def stat_summary(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [float("nan")] * 7
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    q = np.percentile(a, [2.5, 16.0, 50.0, 84.0, 97.5])
    return [float(np.mean(a)), sd, float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])]


def load_mu_eq(args):
    if args.mu_eq is not None:
        return float(args.mu_eq)
    if args.diagnostics:
        with open(args.diagnostics, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["full_mbar"]["mu_eq"])
    raise SystemExit("provide either --mu-eq or --diagnostics to center the bootstrap grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob for one L,T umbrella dataset")
    ap.add_argument("--diagnostics", help="diagnostics JSON containing full_mbar.mu_eq")
    ap.add_argument("--mu-eq", type=float, default=None)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--block-factor", type=float, default=5.0,
                    help="global block length = ceil(block_factor * max tau_int)")
    ap.add_argument("--block-length", type=int, default=None,
                    help="override automatic global block length in recorded samples")
    ap.add_argument("--grid-span", type=float, default=0.004,
                    help="mu grid half-width around full-data mu_eq")
    ap.add_argument("--grid-step", type=float, default=5e-5)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--prefix", default="origin/PUB")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit("no files matched")

    mu_eq_full = load_mu_eq(args)

    # Load trajectories in file order, retaining trajectory boundaries.
    raw = []
    Ls, Ts, mu0s = set(), set(), set()
    centers_set = set()
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            rec = {
                "file": f,
                "replica": replica_from_name(f),
                "nH": np.asarray(d["nH"], dtype=np.int32).copy(),
                "Om0": np.asarray(d["Omega"], dtype=np.float64).copy(),
                "sv": np.sqrt(np.clip(np.asarray(d["lam"], dtype=np.float64), 0.0, None)),
                "L": int(d["L"]), "T": float(d["T"]), "mu0": float(d["mu"]),
                "center": float(d["center"]), "kappa": float(d["kappa"]),
                "acceptance": float(d["acceptance"]) if "acceptance" in d else float("nan"),
                "elapsed": float(d["elapsed"]) if "elapsed" in d else float("nan"),
            }
            raw.append(rec)
            Ls.add(rec["L"]); Ts.add(rec["T"]); mu0s.add(rec["mu0"])
            centers_set.add((rec["center"], rec["kappa"]))

    if len(Ls) != 1 or len(Ts) != 1 or len(mu0s) != 1:
        raise SystemExit("inconsistent L/T/mu metadata across files")
    L = next(iter(Ls)); T = next(iter(Ts)); mu0 = next(iter(mu0s)); N = L * L
    keys = sorted(centers_set)
    centers = np.array([x[0] for x in keys], dtype=float)
    kappas = np.array([x[1] for x in keys], dtype=float)
    key_to_state = {k: i for i, k in enumerate(keys)}

    nH_parts, Om_parts, sv_parts = [], [], []
    trajectories = []
    offset = 0
    tau_rows = []
    for r in raw:
        ns = len(r["nH"])
        g = safe_g(r["nH"])
        tau = (g - 1.0) / 2.0 if np.isfinite(g) else float("nan")
        state = key_to_state[(r["center"], r["kappa"])]
        trajectories.append({"start": offset, "length": ns, "state": state})
        tau_rows.append((r["center"], r["replica"], ns, g, tau, ns / g if np.isfinite(g) and g > 0 else float("nan")))
        nH_parts.append(r["nH"]); Om_parts.append(r["Om0"]); sv_parts.append(r["sv"])
        offset += ns

    nH_all = np.concatenate(nH_parts)
    Om0_all = np.concatenate(Om_parts)
    sv_all = np.concatenate(sv_parts, axis=0)
    # free per-file spectral arrays after pooling
    for r in raw:
        r.pop("sv", None); r.pop("nH", None); r.pop("Om0", None)

    finite_taus = [x[4] for x in tau_rows if np.isfinite(x[4])]
    max_tau = max(finite_taus) if finite_taus else 0.0
    if args.block_length is None:
        block_length = max(32, int(math.ceil(args.block_factor * max_tau)))
    else:
        block_length = int(args.block_length)
    min_traj = min(t["length"] for t in trajectories)
    block_length = min(block_length, max(1, min_traj // 4))

    # Full-data MBAR denominator from exact stored Omega(mu0).
    state_indices = defaultdict(list)
    for tr in trajectories:
        state_indices[tr["state"]].append(np.arange(tr["start"], tr["start"] + tr["length"], dtype=np.int64))
    ordered_parts, N_k_full = [], []
    for k in range(len(keys)):
        idxk = np.concatenate(state_indices[k])
        ordered_parts.append(idxk); N_k_full.append(len(idxk))
    full_sel = np.concatenate(ordered_parts)
    N_k_full = np.asarray(N_k_full, dtype=int)
    identity_order = np.array_equal(full_sel, np.arange(len(full_sel), dtype=np.int64))
    if identity_order:
        nH_full = nH_all
        Om0_full = Om0_all
        sv_full = sv_all
    else:
        nH_full = nH_all[full_sel]
        Om0_full = Om0_all[full_sel]
        sv_full = sv_all[full_sel]
        del nH_all, Om0_all, sv_all
    logden_full = mbar_logden(nH_full, Om0_full, centers, kappas, N_k_full, T, N)

    # Build a fine local mu grid with the exact full-data mu_eq as one grid point.
    nside = int(math.ceil(args.grid_span / args.grid_step))
    offsets = np.arange(-nside, nside + 1, dtype=float) * args.grid_step
    mu_grid = mu_eq_full + offsets
    G = len(mu_grid)
    print(f"Precomputing Omega(mu) grid: {G} points, {len(nH_full)} samples, spectral width {sv_full.shape[1]}")
    omega_grid_full = np.empty((len(nH_full), G), dtype=np.float64)
    for j, mu in enumerate(mu_grid):
        omega_grid_full[:, j] = omega_from_sv(sv_full, T, float(mu), chunk=args.chunk)
        if (j + 1) % max(1, G // 10) == 0 or j + 1 == G:
            print(f"  grid {j+1}/{G}", flush=True)

    # Reindex the globally pooled arrays into the same state-ordered layout used above.
    # Bootstrap trajectory starts need therefore be remapped into this ordered layout.
    # Build per-original-global-index -> state-ordered position map once.
    posmap = np.empty(len(full_sel), dtype=np.int64)
    posmap[full_sel] = np.arange(len(full_sel), dtype=np.int64)
    ordered_trajectories = []
    for tr in trajectories:
        original = np.arange(tr["start"], tr["start"] + tr["length"], dtype=np.int64)
        mapped = posmap[original]
        # Each trajectory remains contiguous because full_sel pools complete files by state.
        if len(mapped) > 1 and not np.all(np.diff(mapped) == 1):
            raise RuntimeError("internal trajectory remapping is not contiguous")
        ordered_trajectories.append({"start": int(mapped[0]), "length": tr["length"], "state": tr["state"]})

    # Full-data observables at the exact grid-center mu_eq.
    center_j = nside
    omega_exact_full = omega_grid_full[:, center_j]
    full_obs = observables_from_root(nH_full, omega_exact_full, T, logden_full, N)

    # Export base Origin data immediately.
    common_comment = [
        f"STITCH publication statistics; L={L}; T={T:.12g}; mu0={mu0:.16g}",
        f"full-data equal-weight mu_eq={mu_eq_full:.16g}",
        f"circular moving-block bootstrap; block_length={block_length} recorded samples; nboot={args.nboot}",
        "Columns are tab-separated and directly importable into OriginPro.",
    ]
    write_dat(
        args.prefix + "_trajectory_autocorrelation.dat",
        ["center", "replica", "n_samples", "g_nH", "tau_int_samples", "time_series_neff"],
        tau_rows, common_comment,
    )

    Ffull = np.full(N + 1, np.nan, dtype=float)
    mask = full_obs["P"] > 0
    Ffull[mask] = -T * np.log(full_obs["P"][mask])
    if np.any(np.isfinite(Ffull)):
        Ffull -= np.nanmin(Ffull)
    dist_base_rows = [
        (i, full_obs["mgrid"][i], full_obs["P"][i], Ffull[i])
        for i in range(N + 1)
    ]
    write_dat(
        args.prefix + "_distribution_full.dat",
        ["nH", "m", "P_full", "Frel_full"],
        dist_base_rows, common_comment,
    )

    # Set fork-shared bootstrap context.
    global _BOOT_CTX
    _BOOT_CTX = {
        "seed": int(args.seed), "block_length": int(block_length),
        "trajectories": ordered_trajectories,
        "centers": centers, "kappas": kappas,
        "nH": nH_full, "Om0": Om0_full,
        "omega_grid": omega_grid_full, "mu_grid": mu_grid,
        "T": T, "N": N,
    }

    print(f"Running {args.nboot} bootstrap replicates with jobs={args.jobs} ...", flush=True)
    if args.jobs > 1:
        # Use "spawn", not "fork": JAX/PyMBAR may already have started threads,
        # and forking a multithreaded JAX process can deadlock.
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.jobs,
            initializer=_init_boot_worker,
            initargs=(_BOOT_CTX,),
            maxtasksperchild=50,
        ) as pool:
            results = []
            for i, row in enumerate(pool.imap_unordered(bootstrap_worker, range(args.nboot), chunksize=1), 1):
                results.append(row)
                if i % max(1, args.nboot // 20) == 0 or i == args.nboot:
                    print(f"  bootstrap {i}/{args.nboot}", flush=True)
    else:
        results = []
        for i in range(args.nboot):
            results.append(bootstrap_worker(i))
            if (i + 1) % max(1, args.nboot // 20) == 0 or i + 1 == args.nboot:
                print(f"  bootstrap {i+1}/{args.nboot}", flush=True)

    results.sort(key=lambda x: x["boot"])
    good = [r for r in results if r.get("ok") == 1]
    bad = [r for r in results if r.get("ok") != 1]
    if len(good) < max(20, int(0.9 * args.nboot)):
        raise RuntimeError(f"too many bootstrap failures: {len(bad)}/{args.nboot}")

    scalar_keys = [
        "mu_eq", "mean_abs_m", "mean_m", "binder", "chi_m", "chi_m_T_over_N",
        "mean_b1_over_N", "mean_b2_over_N", "phase_weight_pos", "phase_weight_neg",
        "phase_weight_zero", "delta_m_peak", "barrier", "bimodal_flag",
        "peak_left_nH", "peak_right_nH", "valley_nH",
    ]

    boot_rows = []
    for r in good:
        boot_rows.append([r["boot"]] + [r.get(k, float("nan")) for k in scalar_keys])
    write_dat(
        args.prefix + "_bootstrap_samples.dat",
        ["boot"] + scalar_keys,
        boot_rows,
        common_comment + [f"successful_bootstrap_replicates={len(good)}; failed={len(bad)}"],
    )

    full_scalar = {
        "mu_eq": mu_eq_full,
        **{k: v for k, v in full_obs.items() if k not in ("P", "mgrid")}
    }
    summary_rows = []
    for k in scalar_keys:
        vals = [r.get(k, float("nan")) for r in good]
        mean_b, sd_b, q025, q16, q50, q84, q975 = stat_summary(vals)
        summary_rows.append((
            k, full_scalar.get(k, float("nan")), mean_b, sd_b,
            q025, q16, q50, q84, q975
        ))
    write_dat(
        args.prefix + "_summary.dat",
        ["observable", "full_estimate", "bootstrap_mean", "bootstrap_sd",
         "ci95_low", "ci68_low", "bootstrap_median", "ci68_high", "ci95_high"],
        summary_rows,
        common_comment + ["Percentile confidence intervals from autocorrelation-aware block bootstrap."],
    )

    # Distribution uncertainty bands; each replicate is shifted to F_min=0 independently.
    Pboot = np.vstack([r["P"] for r in good])
    Pmean = np.mean(Pboot, axis=0)
    Psd = np.std(Pboot, axis=0, ddof=1)
    Pq = np.percentile(Pboot, [2.5, 16, 50, 84, 97.5], axis=0)
    Fboot = np.full_like(Pboot, np.nan, dtype=float)
    for ib in range(Pboot.shape[0]):
        mm = Pboot[ib] > 0
        if np.any(mm):
            Fboot[ib, mm] = -T * np.log(Pboot[ib, mm])
            Fboot[ib] -= np.nanmin(Fboot[ib])
    Fmean = np.nanmean(Fboot, axis=0)
    Fsd = np.nanstd(Fboot, axis=0, ddof=1)
    Fq = np.nanpercentile(Fboot, [2.5, 16, 50, 84, 97.5], axis=0)

    dist_rows = []
    for i in range(N + 1):
        dist_rows.append((
            i, full_obs["mgrid"][i], full_obs["P"][i],
            Pmean[i], Psd[i], Pq[0, i], Pq[1, i], Pq[2, i], Pq[3, i], Pq[4, i],
            Ffull[i], Fmean[i], Fsd[i], Fq[0, i], Fq[1, i], Fq[2, i], Fq[3, i], Fq[4, i]
        ))
    write_dat(
        args.prefix + "_distribution_bootstrap.dat",
        ["nH", "m", "P_full", "P_boot_mean", "P_boot_sd", "P_ci95_low", "P_ci68_low",
         "P_boot_median", "P_ci68_high", "P_ci95_high", "Frel_full", "Frel_boot_mean",
         "Frel_boot_sd", "Frel_ci95_low", "Frel_ci68_low", "Frel_boot_median",
         "Frel_ci68_high", "Frel_ci95_high"],
        dist_rows,
        common_comment + ["Frel=-T*ln(P), shifted independently so each distribution has min(Frel)=0."],
    )

    metadata = {
        "L": L, "T": T, "mu0": mu0, "mu_eq_full": mu_eq_full,
        "n_files": len(files), "n_windows": len(keys), "n_samples": int(len(nH_full)),
        "nboot_requested": int(args.nboot), "nboot_success": len(good), "nboot_failed": len(bad),
        "seed": int(args.seed), "block_factor": float(args.block_factor),
        "max_tau_int_samples": float(max_tau), "block_length_samples": int(block_length),
        "grid_span": float(args.grid_span), "grid_step": float(args.grid_step),
        "grid_points": int(G), "jobs": int(args.jobs),
        "method": "trajectory-wise circular moving-block bootstrap; MBAR refit each replicate; equal-weight mu solved on precomputed exact Omega grid with local linear interpolation; multiprocessing uses spawn when jobs>1",
        "failed_bootstraps": [{"boot": r["boot"], "error": r.get("error", "")} for r in bad],
    }
    with open(args.prefix + "_bootstrap_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(args.prefix + "_bootstrap_metadata.txt", "w", encoding="utf-8") as f:
        for k, v in metadata.items():
            if k == "failed_bootstraps":
                continue
            f.write(f"{k}\t{v}\n")
        if metadata["failed_bootstraps"]:
            f.write("failed_bootstraps\t" + json.dumps(metadata["failed_bootstraps"]) + "\n")

    print("Done.")
    print(f"  block length = {block_length} recorded samples (max tau_int={max_tau:.3f})")
    print(f"  successful bootstrap replicates = {len(good)}/{args.nboot}")
    print(f"  Origin files prefix: {args.prefix}_*.dat")


if __name__ == "__main__":
    main()
