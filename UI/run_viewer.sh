#!/usr/bin/env bash
# Launch the LIFT layout editor (viser) WITHOUT video generation or MoGe (no GPU needed): pre-load a
# clip directory, inspect its camera path and point cloud, edit keyframe poses, record a new trajectory,
# draw last-frame boxes, and export camera_da3_edited.npz + layout_edited.json into the session dir
# (feed those to scripts/infer.py, or use run_viewer_gpu.sh for one-click generation and uploads).
#
# Usage:
#   ./run_viewer.sh                         # ../examples/example1, port 8081
#   ./run_viewer.sh <clip_dir> [port]
#   CLIP=<dir> PORT=<n> THREADS=<n> ./run_viewer.sh
#   BACKGROUND=1 ./run_viewer.sh            # detach + log to viewer.log
#
# The thread caps keep numpy/opencv/scipy from spawning one OpenMP thread per core, which on
# shared login nodes exceeds the process limit and kills the server ("OMP: Error #34").
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLIP="${CLIP:-${1:-../examples/example1}}"
PORT="${PORT:-${2:-8081}}"
THREADS="${THREADS:-4}"
PYTHON="${PYTHON:-python}"

export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" OMP_THREAD_LIMIT=16
export VIEWER_ENABLE_GEN=0

[[ -d "$CLIP" ]] || { echo "ERROR: clip dir not found: $CLIP" >&2; exit 1; }
"$PYTHON" -c "import viser" 2>/dev/null || { echo "ERROR: viser not importable; pip install -r requirements.txt" >&2; exit 1; }

echo "[run_viewer] clip=$CLIP port=$PORT host=$(hostname)"
echo "[run_viewer] on a remote machine forward the port first:  ssh -L $PORT:localhost:$PORT $USER@$(hostname)"
echo "[run_viewer] then open http://localhost:$PORT/"
CMD=("$PYTHON" -u viewer/serve.py --clip "$CLIP" --port "$PORT")
if [[ "${BACKGROUND:-0}" == "1" ]]; then
  nohup "${CMD[@]}" > "$SCRIPT_DIR/viewer.log" 2>&1 &
  echo "[run_viewer] pid $! (logs: viewer.log; stop with: kill $!)"
else
  exec "${CMD[@]}"
fi
