#!/usr/bin/env bash
# Raw downloads for data mixes 1-3 (docs/data_mixes.md) onto RANGE scratch.
# Run on a LOGIN NODE under tmux (jobs without a GPU are reaped):
#   tmux new -d -s dl_hf     "bash scripts/range/download_mix.sh hf     >> /scratch/CLUSTER_USER/logs/dl_hf.log 2>&1"
#   tmux new -d -s dl_egodex "bash scripts/range/download_mix.sh egodex >> /scratch/CLUSTER_USER/logs/dl_egodex.log 2>&1"
# Everything is resumable: snapshot_download skips finished files, curl -C -
# continues partial zips, and each finished source leaves RAW/<name>/.complete.
# The HF token is read from ~/.cache/huggingface/token, so HF_HOME is not set.
set -euo pipefail
RAW=/scratch/CLUSTER_USER/datasets/raw
PY=/scratch/CLUSTER_USER/conda-envs/levjepa-venv/bin/python
mkdir -p "$RAW"

hf() {  # hf <name> <repo> [allow patterns...]
  local name=$1 repo=$2; shift 2
  [ -f "$RAW/$name/.complete" ] && { echo "skip $name"; return; }
  echo "$(date) start $name ($repo)"
  "$PY" - "$repo" "$RAW/$name" "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, out, *pats = sys.argv[1:]
snapshot_download(repo, repo_type='dataset', local_dir=out, max_workers=8,
                  allow_patterns=pats or None)
EOF
  touch "$RAW/$name/.complete"
  echo "$(date) done $name: $(du -sh "$RAW/$name" | cut -f1)"
}

case ${1:-} in
  hf)
    # Order = priority: eval-critical and small first, then the big web sets.
    hf ssv2 morpheushoc/something-something-v2
    for v in imagenetv2 imagenet-r imagenet-a imagenet_sketch objectnet; do
      hf "in1k_variants/$v" "clip-benchmark/wds_$v"
    done
    hf pe_video facebook/PE-Video 'train/*' 'extended/*' README.md   # test/ left out
    hf vidaforge VidaForge/VidaForge-3M 'data/*' 'meta/*' README.md
    hf droid lerobot/droid_1.0.1
    hf cc12m pixparse/cc12m-wds
    ;;
  egodex)
    # Apple CDN, 5 x 300 GB training parts + 200 GB post-split extra; the 16 GB
    # test split is EgoDex's own benchmark and is left out.
    # The CDN gives ~15 MB/s per connection, so the six zips download in parallel.
    mkdir -p "$RAW/egodex"
    fetch() {  # resume until the byte count matches: the CDN closes long transfers (curl 18)
      local p=$1 url="https://ml-site.cdn-apple.com/datasets/egodex/$1.zip" want have
      [ -f "$RAW/egodex/$p.zip.ok" ] && { echo "skip $p"; return; }
      echo "$(date) start egodex $p"
      want=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
      for attempt in $(seq 1 500); do
        curl -sS -L -C - --speed-limit 1000000 --speed-time 120 -o "$RAW/egodex/$p.zip" "$url" \
          || echo "$(date) curl exit $? on $p"      # set -e must not end the resume loop
        have=$(stat -c %s "$RAW/egodex/$p.zip" 2>/dev/null || echo 0)
        [ "$want" = "$have" ] && break
        echo "$(date) egodex $p at $have of $want, resuming (attempt $attempt)"; sleep 15
      done
      [ "$want" = "$have" ] || { echo "giving up on $p: $have of $want"; return 1; }
      touch "$RAW/egodex/$p.zip.ok"
      echo "$(date) done egodex $p"
    }
    parts=${EGODEX_PARTS:-"part1 part2 part3 part4 part5 extra"}
    for p in $parts; do fetch $p & done
    wait
    for p in part1 part2 part3 part4 part5 extra; do [ -f "$RAW/egodex/$p.zip.ok" ] || { echo "egodex incomplete"; exit 1; }; done
    touch "$RAW/egodex/.complete"
    ;;
  *) echo "usage: $0 hf|egodex" >&2; exit 2 ;;
esac
