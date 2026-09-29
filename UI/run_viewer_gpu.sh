#!/usr/bin/env bash
# Launch the LIFT layout editor ON A GPU MACHINE with one-click video generation: the editor and
# the LIFT pipeline (UI/generate_single_video.py -> scripts/infer.py) run in the same process, so
# "Generate video" renders the current camera path + last-frame boxes directly.
#
# Usage:
#   ./run_viewer_gpu.sh                       # pre-load ../examples/example1, port 8080
#   ./run_viewer_gpu.sh <clip_dir> [port]     # pre-load another clip directory
#   CLIP=none ./run_viewer_gpu.sh             # start empty: upload an image or a folder in the browser
# Uploads need MoGe-2 for the point cloud: pip install git+https://github.com/microsoft/MoGe.git
# Env:
#   WANGEN_CKPT      LIFT transformer dir      (default ../models/LIFT/transformer)
#   LIFT_BASE_MODEL  Wan2.1-Fun base model dir (default ../models/Wan2.1-Fun-V1.1-1.3B-Control-Camera)
#   PYTHON, PORT, THREADS, BACKGROUND=1       as in run_viewer.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO="$(dirname "$SCRIPT_DIR")"

CLIP="${CLIP:-${1:-../examples/example1}}"
PORT="${PORT:-${2:-8080}}"
THREADS="${THREADS:-4}"
PYTHON="${PYTHON:-python}"
export WANGEN_CKPT="${WANGEN_CKPT:-$REPO/models/LIFT/transformer}"
export LIFT_BASE_MODEL="${LIFT_BASE_MODEL:-$REPO/models/Wan2.1-Fun-V1.1-1.3B-Control-Camera}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"

if [[ "$CLIP" != "none" && ! -d "$CLIP" ]]; then echo "ERROR: clip dir not found: $CLIP" >&2; exit 1; fi
"$PYTHON" -c "import viser" 2>/dev/null || { echo "ERROR: viser not importable; pip install -r requirements.txt" >&2; exit 1; }
"$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || echo "WARN: torch.cuda not available; generation will fail" >&2
[[ -d "$WANGEN_CKPT" ]] || echo "WARN: LIFT transformer dir not found: $WANGEN_CKPT" >&2
[[ -d "$LIFT_BASE_MODEL" ]] || echo "WARN: base model dir not found: $LIFT_BASE_MODEL" >&2

echo "[run_viewer_gpu] port=$PORT host=$(hostname) clip=$CLIP"
echo "[run_viewer_gpu] weights: $WANGEN_CKPT  (base: $LIFT_BASE_MODEL)"
echo "[run_viewer_gpu] the model loads in the background; the editor is usable right away."
echo "[run_viewer_gpu] forward the port from your laptop:  ssh -L $PORT:localhost:$PORT $USER@$(hostname)"
echo "[run_viewer_gpu] then open http://localhost:$PORT/"
CMD=("$PYTHON" -u viewer/serve.py --port "$PORT")
[[ "$CLIP" != "none" ]] && CMD+=(--clip "$CLIP")
if [[ "${BACKGROUND:-0}" == "1" ]]; then
  nohup "${CMD[@]}" > "$SCRIPT_DIR/viewer_gpu.log" 2>&1 &
  echo "[run_viewer_gpu] pid $! (logs: viewer_gpu.log; stop with: kill $!)"
else
  exec "${CMD[@]}"
fi
