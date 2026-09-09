#!/usr/bin/env bash
set -euo pipefail

L="$1"
T="$2"
RUNS="$3"
BASE="$4"
EXPECTED="$5"

mkdir -p diagnostics
mkdir -p overlap_stats
mkdir -p origin
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=True

echo
echo "============================================================"
echo "STITCH PUBLICATION STATISTICS"
echo "Dataset : ${BASE}"
echo "L       : ${L}"
echo "T       : ${T}"
echo "Input   : ${RUNS}"
echo "Started : $(date)"
echo "============================================================"

NFILES=$(find "${RUNS}" -maxdepth 1 -type f -name '*.npz' | wc -l)

echo "Found ${NFILES} NPZ files; expected ${EXPECTED}"

if [ "${NFILES}" -ne "${EXPECTED}" ]; then
    echo "ERROR: incomplete or unexpected dataset."
    exit 2
fi

DIAG="diagnostics/diagnostics_LOO_${BASE}.json"

echo
echo "[1/5] Shared-bin overlap precheck"
python check_overlap.py "${RUNS}/*.npz" \
  --out "overlap_stats/overlap_${BASE}.json"

echo
echo "[2/5] Formal MBAR diagnostics + LOO"
python publication_diagnostics.py "${RUNS}/*.npz" \
  --loo \
  --out "${DIAG}"

echo
echo "[3/5] Export diagnostics for OriginPro"
python diagnostics_to_origin.py "${DIAG}" \
  --prefix "origin/${BASE}"

echo
echo "[4/5] BF5 bootstrap: 500 replicas"
python publication_bootstrap_origin_v2_1.py "${RUNS}/*.npz" \
  --diagnostics "${DIAG}" \
  --nboot 500 \
  --jobs 2 \
  --block-factor 5 \
  --prefix "origin/${BASE}"

echo
echo "[5/5] BF10 bootstrap: 200 replicas"
python publication_bootstrap_origin_v2_1.py "${RUNS}/*.npz" \
  --diagnostics "${DIAG}" \
  --nboot 200 \
  --jobs 2 \
  --block-factor 10 \
  --prefix "origin/${BASE}_BF10"

echo
echo "============================================================"
echo "SUCCESS: ${BASE}"
echo "Finished: $(date)"
echo "============================================================"
