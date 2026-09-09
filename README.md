# STITCH simulation and analysis files

This repository contains the simulation code and the machine-readable analysis files used in the finite-size study of the fermion-coupled Hodge model.

The repository is intentionally kept small enough for ordinary Git use. Production trajectory files (`.npz`) are not included here. They should be deposited separately with the archival data record if the full production data are made public.

## Contents

The Python programs in the repository root implement the model, umbrella sampling, overlap checks, MBAR reconstruction, validation, and statistical postprocessing. `run_one_stats.sh` runs the analysis for one completed `(L,T)` data set, and `run_all_stats.sh` lists the production ensembles represented by the archived processed results.

`data/diagnostics/` contains JSON diagnostics and MBAR summaries for the production ensembles. `data/processed/` contains the tabular output used for plotting, finite-size analysis, and uncertainty estimates. These files include the BF5 and BF10 bootstrap products that were supplied with the project data.

## Python environment

A clean environment can be created with

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When several umbrella windows are run at the same time, one BLAS thread per process is usually the safer choice:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Local-update validation

`validate_updates.py` compares the low-rank Matsubara update with direct diagonalization. For example,

```bash
python validate_updates.py --L 6 --T 0.0038 --moves 32 --out val_L6.json
```

The frequency cutoff can be increased with the command-line options in the script when a tighter update check is needed.

## Exhaustive L=3 validation

`validate_L3_end_to_end.py` packages the small-system check that was originally run interactively. It enumerates all 512 L=3 configurations, checks all 4608 directed local flips, verifies the homology mapping from the boundary ranks, and compares the finite-cutoff Markov kernels with their exact stationary targets.

For the production cutoff at all four temperatures:

```bash
python validate_L3_end_to_end.py --temps 0.0038 0.005 0.006 0.010 \
  --nfreqs 128 --outdir validation/L3_check
```

The retained reference output and a short description are in `validation/`. Add `--run-mbar` to generate the 20 standard L=3 umbrella trajectories and pass them through the production MBAR analysis.

## Generating umbrella runs

`make_jobs.py` writes a shell script containing independent umbrella-window jobs. A typical command is

```bash
python make_jobs.py --L 8 --T 0.0038 --width 6 --replicas 2 \
  --outdir runs/T0038/L8 --script run_L8.sh
```

Each job writes one `.npz` trajectory. These trajectory files are large and are excluded by `.gitignore`.

## Checking and analysing a completed ensemble

For a set of completed windows,

```bash
python check_overlap.py 'runs/T0038/L8/*.npz' --out overlap_T0038_L8.json
python analyze_mbar.py 'runs/T0038/L8/*.npz' --out result_T0038_L8.json
```

The publication postprocessing used here is collected in `run_one_stats.sh`. It checks the expected number of trajectories, writes the overlap and leave-one-window-out diagnostics, and produces the BF5 and BF10 bootstrap tables.

## Archived data

The processed tables are the compact, machine-readable record behind the statistical analysis. The JSON files preserve the corresponding overlap and MBAR diagnostics. The raw production trajectories are not part of this GitHub package because they were not contained in the uploaded source archives used to assemble it.

A checksum list for all files in this release is provided in `MANIFEST.sha256`.
