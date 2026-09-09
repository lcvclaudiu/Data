#!/usr/bin/env python3
"""Publication diagnostics for STITCH umbrella/MBAR data.

This script is intentionally diagnostic rather than a replacement for
analyze_mbar.py.  It checks metadata, time-series correlation, independent
replicas, formal MBAR state overlap, and target-state reweighting ESS.

Requires the same environment as STITCH_final_umbrella_workflow.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import brentq
from pymbar import MBAR, timeseries


def bias_vec(nh, center, kappa, N):
    x = nh / N
    xc = center / N
    return 0.5 * kappa * N * (x - xc) ** 2


def logsumexp_cols(A):
    mx = np.max(A, axis=0)
    return mx + np.log(np.sum(np.exp(A - mx), axis=0))


def omega_batch(lams, T, mu, chunk=4096):
    """Exact grand potential from stored Gram eigenvalues, vectorized in chunks."""
    M = lams.shape[0]
    out = np.empty(M, dtype=np.float64)
    for s in range(0, M, chunk):
        x = np.asarray(lams[s:s+chunk], dtype=np.float64)
        sv = np.sqrt(np.clip(x, 0.0, None))
        out[s:s+len(x)] = -T * (
            np.logaddexp(0.0, -(sv - mu) / T)
            + np.logaddexp(0.0, -(-sv - mu) / T)
        ).sum(axis=1)
    return out


def replica_from_name(path):
    m = re.search(r"_r(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else None


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


def build_analysis(records, mu_span=0.02, chunk=4096):
    """Run the same central MBAR reconstruction as analyze_mbar.py plus diagnostics."""
    if not records:
        raise ValueError("empty record set")
    d0 = records[0]["d"]
    L = int(d0["L"]); T = float(d0["T"]); mu0 = float(d0["mu"]); N = L * L

    keys = []
    for r in records:
        key = (float(r["d"]["center"]), float(r["d"]["kappa"]))
        if key not in keys:
            keys.append(key)
    keys.sort()
    centers = np.array([k[0] for k in keys], dtype=float)
    kappas = np.array([k[1] for k in keys], dtype=float)

    nH_parts, lam_parts, N_k = [], [], []
    for key in keys:
        rs = [r for r in records if (float(r["d"]["center"]), float(r["d"]["kappa"])) == key]
        nH_parts.append(np.concatenate([np.asarray(r["d"]["nH"], dtype=np.int32) for r in rs]))
        lam_parts.append(np.concatenate([np.asarray(r["d"]["lam"], dtype=np.float64) for r in rs], axis=0))
        N_k.append(sum(len(r["d"]["nH"]) for r in rs))

    nH = np.concatenate(nH_parts)
    lams = np.concatenate(lam_parts, axis=0)
    N_k = np.asarray(N_k, dtype=int)
    Om0 = omega_batch(lams, T, mu0, chunk=chunk)

    K = len(keys); M = len(nH)
    u_kn = np.empty((K, M), dtype=np.float64)
    for k in range(K):
        u_kn[k] = (Om0 + bias_vec(nH, centers[k], kappas[k], N)) / T

    mbar = MBAR(u_kn, N_k, verbose=False, relative_tolerance=1e-10, maximum_iterations=10000)
    f_k = np.asarray(mbar.f_k)
    A = np.vstack([np.log(N_k[k]) + f_k[k] - u_kn[k] for k in range(K) if N_k[k] > 0])
    logden = logsumexp_cols(A)

    omega_cache = {float(mu0): Om0}
    def Om(mu):
        key = float(mu)
        if key not in omega_cache:
            omega_cache[key] = omega_batch(lams, T, key, chunk=chunk)
        return omega_cache[key]

    def weights(mu):
        lw = -Om(mu) / T - logden
        m = np.max(lw)
        lw = lw - (m + np.log(np.sum(np.exp(lw - m))))
        return np.exp(lw)

    mvals = 2.0 * nH / N - 1.0
    def phase_balance(mu):
        w = weights(mu)
        return float(w[mvals > 0].sum() - w[mvals < 0].sum())

    lo, hi = mu0 - mu_span, mu0 + mu_span
    flo, fhi = phase_balance(lo), phase_balance(hi)
    if flo * fhi > 0:
        raise RuntimeError(f"equal-weight root not bracketed in [{lo},{hi}]: {flo}, {fhi}")
    mueq = float(brentq(phase_balance, lo, hi, xtol=1e-12))
    w = weights(mueq)

    def mean(x): return float(np.sum(w * x))
    m2 = mean(mvals * mvals); m4 = mean(mvals ** 4)
    binder = 1.0 - m4 / (3.0 * m2 * m2)
    chi = N / T * (m2 - mean(mvals) ** 2)

    overlap = mbar.compute_overlap()
    overlap_matrix = np.asarray(overlap["matrix"], dtype=float)
    overlap_eigs = np.asarray(overlap["eigenvalues"], dtype=float)
    overlap_scalar = float(overlap["scalar"])
    try:
        state_neff = np.asarray(mbar.compute_effective_sample_number(verbose=False), dtype=float)
    except TypeError:
        state_neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)

    kish = float(1.0 / np.sum(w * w))
    pos = w[w > 0]
    entropy_ess = float(np.exp(-np.sum(pos * np.log(pos))))

    adjacent = []
    for k in range(K - 1):
        adjacent.append({
            "centers": [float(centers[k]), float(centers[k+1])],
            "O_k_kplus1": float(overlap_matrix[k, k+1]),
            "O_kplus1_k": float(overlap_matrix[k+1, k]),
            "min_directional": float(min(overlap_matrix[k, k+1], overlap_matrix[k+1, k])),
        })

    return {
        "L": L, "T": T, "mu0": mu0, "mu_eq": mueq,
        "n_files": len(records), "n_windows": K, "n_samples": M,
        "mean_abs_m": mean(np.abs(mvals)), "mean_m": mean(mvals),
        "binder": float(binder), "chi_m": float(chi), "chi_m_T_over_N": float(chi*T/N),
        "phase_weight_pos": float(w[mvals > 0].sum()),
        "phase_weight_neg": float(w[mvals < 0].sum()),
        "target_kish_ess": kish,
        "target_entropy_ess": entropy_ess,
        "target_kish_fraction": float(kish / M),
        "mbar_overlap_scalar": overlap_scalar,
        "mbar_overlap_eigenvalues": overlap_eigs.tolist(),
        "mbar_overlap_matrix": overlap_matrix.tolist(),
        "mbar_adjacent_overlap": adjacent,
        "mbar_state_effective_samples": state_neff.tolist(),
        "centers": centers.tolist(),
        "N_k": N_k.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob for one L,T umbrella dataset")
    ap.add_argument("--mu-span", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--loo", action="store_true", help="also perform leave-one-interior-window-out MBAR")
    ap.add_argument("--out", default="publication_diagnostics.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit("no files matched")

    records = []
    file_rows = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        rep = replica_from_name(f)
        g = safe_g(d["nH"])
        ns = len(d["nH"])
        file_rows.append({
            "file": f,
            "center": float(d["center"]),
            "replica": rep,
            "n_samples": int(ns),
            "acceptance": float(d["acceptance"]) if "acceptance" in d else None,
            "elapsed_hours": float(d["elapsed"]) / 3600.0 if "elapsed" in d else None,
            "nH_min": int(np.min(d["nH"])),
            "nH_max": int(np.max(d["nH"])),
            "nH_mean": float(np.mean(d["nH"])),
            "nH_std": float(np.std(d["nH"], ddof=1)) if ns > 1 else 0.0,
            "g_nH": g,
            "tau_int_nH_samples": float((g - 1.0) / 2.0) if np.isfinite(g) else None,
            "time_series_neff_nH": float(ns / g) if np.isfinite(g) and g > 0 else None,
            "thin": int(d["thin"]) if "thin" in d else None,
            "nfreq": int(d["nfreq"]) if "nfreq" in d else None,
            "equil_sweeps": int(d["equil_sweeps"]) if "equil_sweeps" in d else None,
            "meas_sweeps": int(d["meas_sweeps"]) if "meas_sweeps" in d else None,
        })
        records.append({"file": f, "replica": rep, "d": d})

    # Hard metadata consistency checks.
    meta_fields = ["L", "T", "mu", "kappa", "thin", "nfreq", "equil_sweeps", "meas_sweeps"]
    metadata = {}
    metadata_ok = True
    for fld in meta_fields:
        vals = []
        for r in records:
            if fld in r["d"]:
                v = np.asarray(r["d"][fld]).item()
                vals.append(v)
        uniq = []
        for v in vals:
            if not any(np.isclose(v, u, rtol=0, atol=1e-12) if isinstance(v, (float, np.floating)) else v == u for u in uniq):
                uniq.append(v)
        metadata[fld] = [float(x) if isinstance(x, (float, np.floating)) else int(x) if isinstance(x, (int, np.integer)) else x for x in uniq]
        if len(uniq) > 1:
            metadata_ok = False

    full = build_analysis(records, args.mu_span, args.chunk)

    # Independent-replica reconstructions: one trajectory per center.
    reps = sorted({r["replica"] for r in records if r["replica"] is not None})
    replica_results = {}
    for rep in reps:
        rr = [r for r in records if r["replica"] == rep]
        # only run if the replica spans all centers
        if len({float(r["d"]["center"]) for r in rr}) == full["n_windows"]:
            replica_results[str(rep)] = build_analysis(rr, args.mu_span, args.chunk)
        else:
            replica_results[str(rep)] = {"error": "replica does not span all umbrella centers"}

    loo = []
    if args.loo:
        centers = sorted({float(r["d"]["center"]) for r in records})
        for c in centers[1:-1]:
            rr = [r for r in records if float(r["d"]["center"]) != c]
            try:
                z = build_analysis(rr, args.mu_span, args.chunk)
                loo.append({
                    "removed_center": c,
                    "mu_eq": z["mu_eq"],
                    "delta_mu_eq": z["mu_eq"] - full["mu_eq"],
                    "binder": z["binder"],
                    "delta_binder": z["binder"] - full["binder"],
                    "chi_m_T_over_N": z["chi_m_T_over_N"],
                    "delta_chi_m_T_over_N": z["chi_m_T_over_N"] - full["chi_m_T_over_N"],
                    "target_kish_ess": z["target_kish_ess"],
                })
            except Exception as e:
                loo.append({"removed_center": c, "error": str(e)})

    # concise replica spread for immediate gate
    rep_numeric = [v for v in replica_results.values() if "mu_eq" in v]
    replica_spread = {}
    for key in ["mu_eq", "mean_abs_m", "binder", "chi_m_T_over_N"]:
        vals = np.array([r[key] for r in rep_numeric], dtype=float)
        if len(vals):
            replica_spread[key] = {
                "min": float(vals.min()), "max": float(vals.max()),
                "range": float(vals.max() - vals.min()),
                "mean": float(vals.mean()),
                "sd_across_replicas": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }

    out = {
        "metadata_consistent": bool(metadata_ok),
        "metadata_unique_values": metadata,
        "file_diagnostics": file_rows,
        "full_mbar": full,
        "replica_mbar": replica_results,
        "replica_spread": replica_spread,
        "leave_one_window_out": loo,
        "notes": [
            "g_nH is a time-series statistical inefficiency estimate for the recorded nH trajectory; tau_int=(g-1)/2 in recorded-sample units.",
            "mbar_overlap is the formal MBAR state overlap diagnostic; it is stronger than integer-bin sharing but still does not replace block/bootstrap uncertainty.",
            "target_kish_ess=1/sum(w^2) measures reweighting concentration at mu_eq; it is not autocorrelation-corrected.",
            "Independent replica reconstructions are a robustness check, not by themselves a final confidence interval.",
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    concise = {
        "metadata_consistent": out["metadata_consistent"],
        "L": full["L"], "T": full["T"], "n_files": full["n_files"],
        "n_windows": full["n_windows"], "n_samples": full["n_samples"],
        "mu_eq": full["mu_eq"], "mean_abs_m": full["mean_abs_m"],
        "binder": full["binder"], "chi_m_T_over_N": full["chi_m_T_over_N"],
        "mbar_overlap_scalar": full["mbar_overlap_scalar"],
        "min_adjacent_directional_overlap": min(x["min_directional"] for x in full["mbar_adjacent_overlap"]) if full["mbar_adjacent_overlap"] else None,
        "target_kish_ess": full["target_kish_ess"],
        "target_kish_fraction": full["target_kish_fraction"],
        "replica_spread": replica_spread,
        "loo_n": len(loo),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
