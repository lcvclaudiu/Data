# Source notes

This public package was assembled from the computational files supplied with the project:

- `py-code(1).zip`
- `simulations(1).zip`
- `origin_ready_current_L8.zip`

The external papers, crystallographic reference cards, proposal template, and internal long-form theory documents supplied elsewhere in the project are not part of this software/data repository. They are not required to run the numerical workflow, and several are third-party publications or licensed reference material.

The supplied single-dataset runner was named `run_one_stats-old.sh`, while the batch script calls `run_one_stats.sh`. The same runner is therefore provided under the expected name at the repository root. The supplied file is retained under `legacy/` for provenance.

The original batch script is also retained under `legacy/`. `run_all_stats.sh` is the corresponding current list of the fourteen `(L,T)` ensembles represented by the processed files in this package.

No raw `.npz` production trajectories were present in the source archives used here, so this repository is a code-and-processed-data companion rather than a complete raw-data deposit.


The exhaustive L=3 validation was originally executed interactively rather than kept as a separate source file. `validate_L3_end_to_end.py` reconstructs that check from the repository's boundary-matrix, grand-potential, low-rank update, umbrella, and MBAR routines. The retained numerical output is kept under `validation/`.
