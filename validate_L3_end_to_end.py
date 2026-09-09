#!/usr/bin/env python3
"""Exhaustive L=3 validation for the finite-size simulation workflow.

The L=3 system contains nine binary cell variables, hence 512 attachment
configurations.  This program enumerates all of them, checks the homology
mapping, compares every directed one-cell flip with direct diagonalization,
and measures the stationary-distribution error of the finite Matsubara
update used by the production sampler.

The optional --run-mbar stage also generates the 20 standard L=3 umbrella
files (ten centers, two replicas) and analyses them with analyze_mbar.py.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs

from stitch_fss import build_B2_sparse, first_root
from fermion_core import delta_decomp, omega_from_lam


L = 3
N = L * L
N_STATES = 1 << N


def state_array(code):
    bits = [(code >> k) & 1 for k in range(N)]
    return np.asarray(bits, dtype=np.int8).reshape(L, L)


def real_space_B1(L):
    """Construct the vertex-edge boundary matrix used by build_B2_sparse."""
    N = L * L

    def corner(i, j):
        return (i % L) * L + (j % L)

    def private(i, j, role):
        cell = (i % L) * L + (j % L)
        return N + 2 * cell + (0 if role == 5 else 1)

    role_coord = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
    label_to_role = {1: 6, 2: 3, 3: 2, 4: 4, 5: 1, 6: 5}

    def role_vertex(i, j, role):
        if role <= 4:
            di, dj = role_coord[role]
            return corner(i + di, j + dj)
        return private(i, j, role)

    def label_vertex(i, j, label):
        return role_vertex(i, j, label_to_role[label])

    edges = set()
    for i in range(L):
        for j in range(L):
            vertices = [label_vertex(i, j, label) for label in range(1, 7)]
            for a in range(6):
                for b in range(a + 1, 6):
                    edges.add(tuple(sorted((vertices[a], vertices[b]))))

    edges = sorted(edges)
    rows, cols, data = [], [], []
    for col, (u, v) in enumerate(edges):
        rows.extend((u, v))
        cols.extend((col, col))
        data.extend((-1.0, 1.0))

    B1 = coo_matrix(
        (data, (rows, cols)), shape=(3 * N, len(edges)), dtype=float
    ).tocsr()
    return B1, edges


def enumerate_states():
    B1, edges = real_space_B1(L)
    rank_B1 = int(np.linalg.matrix_rank(B1.toarray()))
    b0 = B1.shape[0] - rank_B1

    states = []
    topology_ok = b0 == 1
    max_chain_error = 0.0

    t0 = time.time()
    for code in range(N_STATES):
        typ = state_array(code)
        B2 = build_B2_sparse(L, typ)
        G = (B2.T @ B2).toarray()
        lam, Q = eigh(G, check_finite=False, driver="evd")

        # Rank is determined from B2 itself. Squaring to B2.T @ B2 can lift
        # roundoff-scale zero modes when their eigenvalues are square-rooted.
        singular = np.linalg.svd(B2.toarray(), compute_uv=False)
        rank_B2 = int(np.count_nonzero(singular > 1e-10))
        b2 = B2.shape[1] - rank_B2
        b1 = B1.shape[1] - rank_B1 - rank_B2
        nH = int(typ.sum())

        expected_b2 = nH if nH > 0 else 1
        expected_b1 = 2 * N + 1 + nH if nH > 0 else 2 * N + 2
        topology_ok &= b1 == expected_b1 and b2 == expected_b2

        chain = B1 @ B2
        if chain.nnz:
            max_chain_error = max(max_chain_error, float(np.max(np.abs(chain.data))))

        states.append(
            {
                "code": code,
                "typ": typ,
                "nH": nH,
                "G": G,
                "lam": lam,
                "Q": Q,
                "b1": int(b1),
                "b2": int(b2),
            }
        )

    topology_ok &= len(edges) == 13 * N and max_chain_error < 1e-12
    elapsed = time.time() - t0
    return states, bool(topology_ok), float(max_chain_error), elapsed


def omega_vector(states, T, mu):
    return np.asarray([omega_from_lam(s["lam"], T, mu) for s in states])


def normalized_weights(energy, T):
    x = -(energy - np.min(energy)) / T
    w = np.exp(x)
    return w / np.sum(w)


def exact_mu_equal_weight(states, T, mu0):
    nH = np.asarray([s["nH"] for s in states])

    def balance(mu):
        p = normalized_weights(omega_vector(states, T, mu), T)
        return float(p[nH > N / 2].sum() - p[nH < N / 2].sum())

    span = 0.02
    for _ in range(8):
        lo, hi = mu0 - span, mu0 + span
        flo, fhi = balance(lo), balance(hi)
        if flo * fhi <= 0:
            return float(brentq(balance, lo, hi, xtol=1e-13))
        span *= 2
    raise RuntimeError("could not bracket the exact equal-weight chemical potential")


def matsubara_cutoffs(state, next_state, T, mu, cutoffs):
    """Evaluate several frequency cutoffs without repeating the matrix setup."""
    G, Gp = state["G"], next_state["G"]
    U, c = delta_decomp(Gp - G)
    if len(c) == 0:
        return {n: 0.0 for n in cutoffs}

    lam, Q = state["lam"], state["Q"]
    W = Q.T @ U
    wanted = set(cutoffs)
    total = 0.0
    out = {}

    for n in range(max(cutoffs)):
        om = (2 * n + 1) * np.pi * T
        z = (om + 1j * mu) ** 2
        R = (W.T * (1.0 / (lam + z))) @ W
        M = np.eye(len(c), dtype=complex) + c[:, None] * R
        sign, logabs = np.linalg.slogdet(M)
        total += float(np.real(np.log(sign) + logabs))
        count = n + 1
        if count in wanted:
            out[count] = -2.0 * T * total

    return out


def stationary_distribution(delta_approx, exact_omega, nH, T, center, kappa):
    x = nH / N
    xc = center / N
    bias = 0.5 * kappa * N * (x - xc) ** 2
    target = normalized_weights(exact_omega + bias, T)

    rows, cols, vals = [], [], []
    for code in range(N_STATES):
        row_sum = 0.0
        for k in range(N):
            nxt = code ^ (1 << k)
            dbias = bias[nxt] - bias[code]
            loga = -(delta_approx[code, k] + dbias) / T
            accept = 1.0 if loga >= 0.0 else math.exp(loga)
            value = accept / N
            rows.append(code)
            cols.append(nxt)
            vals.append(value)
            row_sum += value
        rows.append(code)
        cols.append(code)
        vals.append(1.0 - row_sum)

    P = csr_matrix((vals, (rows, cols)), shape=(N_STATES, N_STATES))
    eigval, eigvec = eigs(P.T, k=1, which="LM", tol=1e-13, maxiter=100000)
    stationary = np.real(eigvec[:, 0])
    if stationary.sum() < 0:
        stationary = -stationary
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()

    tv = 0.5 * float(np.abs(stationary - target).sum())
    return tv, float(np.real(eigval[0]))


def check_updates(states, T, mu0, nfreqs, kappa):
    exact_omega = omega_vector(states, T, mu0)
    nH = np.asarray([s["nH"] for s in states], dtype=float)

    approx = {nfreq: np.empty((N_STATES, N), dtype=float) for nfreq in nfreqs}
    max_error = {nfreq: 0.0 for nfreq in nfreqs}

    for code, state in enumerate(states):
        for k in range(N):
            nxt = code ^ (1 << k)
            values = matsubara_cutoffs(state, states[nxt], T, mu0, nfreqs)
            delta_exact = exact_omega[nxt] - exact_omega[code]
            for nfreq in nfreqs:
                value = values[nfreq]
                approx[nfreq][code, k] = value
                max_error[nfreq] = max(max_error[nfreq], abs(value - delta_exact))

    results = {}
    for nfreq in nfreqs:
        center_tvs = []
        center_eigenvalues = []
        for center in range(N + 1):
            tv, ev = stationary_distribution(
                approx[nfreq], exact_omega, nH, T, center, kappa
            )
            center_tvs.append(tv)
            center_eigenvalues.append(ev)
        results[str(nfreq)] = {
            "max_abs_deltaOmega_error": float(max_error[nfreq]),
            "max_stationary_TV": float(max(center_tvs)),
            "stationary_TV_by_center": [float(x) for x in center_tvs],
            "stationary_eigenvalue_by_center": [float(x) for x in center_eigenvalues],
        }
    return results


def exact_p_nH(states, T, mu):
    nH = np.asarray([s["nH"] for s in states], dtype=int)
    p = normalized_weights(omega_vector(states, T, mu), T)
    return np.bincount(nH, weights=p, minlength=N + 1)


def umbrella_seed(T, center, replica):
    return 100000 * L + 1000 * int(round(1e5 * T)) + 100 * center + replica


def run_mbar_stage(states, T, mu0, mu_exact, nfreq, kappa, outdir, overwrite=False):
    """Generate the standard 20 L=3 windows and run the repository MBAR analysis."""
    root = Path(__file__).resolve().parent
    run_dir = Path(outdir) / f"T{int(round(T * 1e5)):04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for center in range(N + 1):
        for replica in range(2):
            out = run_dir / f"L3_T{T:.5f}_c{center:04d}_r{replica}.npz"
            if out.exists() and not overwrite:
                continue
            cmd = [
                sys.executable,
                str(root / "umbrella_window.py"),
                "--L", "3",
                "--T", f"{T:.15g}",
                "--mu", f"{mu0:.15g}",
                "--center", str(center),
                "--kappa", f"{kappa:.15g}",
                "--equil-sweeps", "20",
                "--meas-sweeps", "120",
                "--thin", "4",
                "--nfreq", str(nfreq),
                "--seed", str(umbrella_seed(T, center, replica)),
                "--out", str(out),
            ]
            subprocess.run(cmd, check=True)

    result_file = run_dir / "mbar_summary.json"
    cmd = [
        sys.executable,
        str(root / "analyze_mbar.py"),
        str(run_dir / "*.npz"),
        "--out", str(result_file),
    ]
    subprocess.run(cmd, check=True)

    with result_file.open() as f:
        mbar = json.load(f)

    p_exact = exact_p_nH(states, T, mu_exact)
    p_mbar = np.asarray(mbar["P_nH"], dtype=float)
    tv = 0.5 * float(np.abs(p_exact - p_mbar).sum())

    return {
        "n_files": int(mbar["n_files"]),
        "mu_eq": float(mbar["mu_eq"]),
        "delta_mu": float(mbar["mu_eq"] - mu_exact),
        "P_nH_TV": tv,
        "result_file": str(result_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Exhaustive L=3 validation")
    parser.add_argument(
        "--temps", nargs="+", type=float, default=[0.0038],
        help="temperatures T/J to validate",
    )
    parser.add_argument(
        "--nfreqs", nargs="+", type=int, default=[32, 64, 128, 256],
        help="Matsubara cutoffs to compare",
    )
    parser.add_argument(
        "--production-nfreq", type=int, default=128,
        help="cutoff used by the optional umbrella/MBAR stage",
    )
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument(
        "--run-mbar", action="store_true",
        help="also generate 20 L=3 umbrella files and run MBAR",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing L=3 umbrella files in the output directory",
    )
    parser.add_argument("--outdir", default="validation/L3")
    args = parser.parse_args()

    nfreqs = sorted(set(args.nfreqs))
    if not nfreqs or min(nfreqs) < 1:
        raise SystemExit("nfreqs must contain positive integers")

    print(f"Enumerating all {N_STATES} L=3 configurations ...", flush=True)
    states, topology_ok, chain_error, elapsed = enumerate_states()
    print(
        f"Enumeration finished in {elapsed:.2f} s; topology mapping pass={topology_ok}",
        flush=True,
    )

    report = {
        "L": L,
        "n_configurations": N_STATES,
        "n_directed_flips": N_STATES * N,
        "topology_mapping_pass": topology_ok,
        "max_chain_error": chain_error,
        "kappa": args.kappa,
        "temperatures": {},
    }

    for T in args.temps:
        mu0 = float(first_root(L, T)[0])
        mu_exact = exact_mu_equal_weight(states, T, mu0)

        print(f"\n=== L=3 exact validation at T/J={T:g} ===")
        print(f"homogeneous mu0={mu0:.15g}; exact ensemble mu_eq={mu_exact:.15g}")
        print(f"Exhaustively checking all {N_STATES * N} directed local flips ...", flush=True)

        update_results = check_updates(states, T, mu0, nfreqs, args.kappa)
        for nfreq in nfreqs:
            row = update_results[str(nfreq)]
            print(
                f"nfreq={nfreq}: max |deltaOmega error|="
                f"{row['max_abs_deltaOmega_error']:.3e} J; "
                f"max stationary TV={row['max_stationary_TV']:.3e}",
                flush=True,
            )

        item = {
            "mu0": mu0,
            "exact_mu_eq": mu_exact,
            "updates": update_results,
        }

        if args.run_mbar:
            print("Generating 20 production-format L=3 umbrella files; running MBAR ...")
            mbar = run_mbar_stage(
                states,
                T,
                mu0,
                mu_exact,
                args.production_nfreq,
                args.kappa,
                args.outdir,
                overwrite=args.overwrite,
            )
            print(
                f"MBAR mu_eq={mbar['mu_eq']:.15g}; "
                f"delta mu={mbar['delta_mu']:+.3e}; "
                f"P(nH) TV={mbar['P_nH_TV']:.3e}"
            )
            item["mbar"] = mbar

        report["temperatures"][f"{T:g}"] = item

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_file = outdir / "L3_validation_report.json"
    with report_file.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\nWrote {report_file}")
    if not topology_ok:
        raise SystemExit("FAIL: topology mapping check did not pass")


if __name__ == "__main__":
    main()
