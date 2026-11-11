#!/usr/bin/env bash
# TAP-Vid-DAVIS (1.67 GB, public GCS). Fine to run on a RANGE login node.
set -euo pipefail
D=/scratch/CLUSTER_USER/datasets/tapvid
mkdir -p "$D"; cd "$D"
wget -c https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip -n tapvid_davis.zip
ls -la "$D"/tapvid_davis
