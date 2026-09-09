#!/usr/bin/env python3
"""Convert STITCH publication_diagnostics JSON to tabular OriginPro .dat files."""
import argparse
import json
import os
import numpy as np


def fmt(x):
    if x is None:
        return "nan"
    if isinstance(x, str):
        return x
    try:
        y = float(x)
        if np.isnan(y): return "nan"
        return f"{y:.16g}"
    except Exception:
        return str(x)


def write(path, header, rows, comments):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in comments:
            f.write(f"# {c}\n")
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(fmt(x) for x in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--prefix", default="origin/diagnostics")
    a = ap.parse_args()
    with open(a.json_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    full = d["full_mbar"]
    L, T = full["L"], full["T"]
    comments = [f"STITCH publication diagnostics; L={L}; T={T}", "Tab-separated; OriginPro-ready."]

    # Full summary, one row.
    keys = ["L", "T", "mu0", "mu_eq", "n_files", "n_windows", "n_samples",
            "mean_abs_m", "mean_m", "binder", "chi_m", "chi_m_T_over_N",
            "phase_weight_pos", "phase_weight_neg", "target_kish_ess",
            "target_entropy_ess", "target_kish_fraction", "mbar_overlap_scalar"]
    write(a.prefix + "_full_summary.dat", keys, [[full.get(k) for k in keys]], comments)

    rows = []
    for r in d.get("file_diagnostics", []):
        rows.append([r.get(k) for k in ["center", "replica", "n_samples", "acceptance", "elapsed_hours",
                    "nH_min", "nH_max", "nH_mean", "nH_std", "g_nH", "tau_int_nH_samples",
                    "time_series_neff_nH", "thin", "nfreq", "equil_sweeps", "meas_sweeps"]])
    write(a.prefix + "_trajectory.dat",
          ["center", "replica", "n_samples", "acceptance", "elapsed_hours", "nH_min", "nH_max",
           "nH_mean", "nH_std", "g_nH", "tau_int_samples", "time_series_neff", "thin", "nfreq",
           "equil_sweeps", "meas_sweeps"], rows, comments)

    rows = []
    for r in full.get("mbar_adjacent_overlap", []):
        rows.append([r["centers"][0], r["centers"][1], r["O_k_kplus1"], r["O_kplus1_k"], r["min_directional"]])
    write(a.prefix + "_adjacent_overlap.dat",
          ["center_left", "center_right", "O_left_right", "O_right_left", "O_min"], rows, comments)

    rows = []
    centers = full.get("centers", [])
    neff = full.get("mbar_state_effective_samples", [])
    Nk = full.get("N_k", [])
    for i in range(min(len(centers), len(neff), len(Nk))):
        rows.append([centers[i], Nk[i], neff[i]])
    write(a.prefix + "_mbar_state_neff.dat", ["center", "N_k", "mbar_state_neff"], rows, comments)

    rows = []
    for rep, r in sorted(d.get("replica_mbar", {}).items(), key=lambda x: int(x[0])):
        if "error" in r: continue
        rows.append([rep] + [r.get(k) for k in ["mu_eq", "mean_abs_m", "mean_m", "binder", "chi_m",
                    "chi_m_T_over_N", "phase_weight_pos", "phase_weight_neg", "target_kish_ess",
                    "target_entropy_ess", "target_kish_fraction", "mbar_overlap_scalar"]])
    write(a.prefix + "_replica_mbar.dat",
          ["replica", "mu_eq", "mean_abs_m", "mean_m", "binder", "chi_m", "chi_m_T_over_N",
           "phase_weight_pos", "phase_weight_neg", "target_kish_ess", "target_entropy_ess",
           "target_kish_fraction", "mbar_overlap_scalar"], rows, comments)

    rows = []
    for r in d.get("leave_one_window_out", []):
        rows.append([r.get(k) for k in ["removed_center", "mu_eq", "delta_mu_eq", "binder", "delta_binder",
                    "chi_m_T_over_N", "delta_chi_m_T_over_N", "target_kish_ess"]])
    write(a.prefix + "_LOO.dat",
          ["removed_center", "mu_eq", "delta_mu_eq", "binder", "delta_binder", "chi_m_T_over_N",
           "delta_chi_m_T_over_N", "target_kish_ess"], rows, comments)


if __name__ == "__main__":
    main()
