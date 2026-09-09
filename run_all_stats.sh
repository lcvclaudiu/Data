#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "STITCH statistical postprocessing"
echo "Started: $(date)"
echo "============================================================"

# T/J = 0.0038
bash run_one_stats.sh 6  0.0038 runs/T0038/L6  T0038_L6  20
bash run_one_stats.sh 8  0.0038 runs/T0038/L8  T0038_L8  24
bash run_one_stats.sh 10 0.0038 runs/T0038/L10 T0038_L10 33
bash run_one_stats.sh 12 0.0038 runs/T0038/L12 T0038_L12 39

# T/J = 0.0050
bash run_one_stats.sh 8  0.0050 runs/PUB_T0050/L8  PUB_T0050_L8  24
bash run_one_stats.sh 10 0.0050 runs/PUB_T0050/L10 PUB_T0050_L10 22
bash run_one_stats.sh 12 0.0050 runs/PUB_T0050/L12 PUB_T0050_L12 39

# T/J = 0.0060
bash run_one_stats.sh 8  0.0060 runs/PUB_T0060/L8  PUB_T0060_L8  24
bash run_one_stats.sh 10 0.0060 runs/PUB_T0060/L10 PUB_T0060_L10 22
bash run_one_stats.sh 12 0.0060 runs/PUB_T0060/L12 PUB_T0060_L12 39

# T/J = 0.0100
bash run_one_stats.sh 6  0.0100 runs/T001/L6  T001_L6  20
bash run_one_stats.sh 8  0.0100 runs/T001/L8  T001_L8  24
bash run_one_stats.sh 10 0.0100 runs/T001/L10 T001_L10 33
bash run_one_stats.sh 12 0.0100 runs/T001/L12 T001_L12 39

echo
echo "============================================================"
echo "Postprocessing complete"
echo "Finished: $(date)"
echo "============================================================"
