#!/usr/bin/env bash
# Launch (or resume) the full 100M DDP run in a detached tmux session, with a
# sidecar that pushes checkpoints to R2 so a reclaimed box loses minutes, not
# hours. Survives SSH disconnects.
#
#   bash scripts/launch_train.sh            # start, or auto-resume from newest checkpoint
#   bash scripts/launch_train.sh --attach   # attach to the live session
#   bash scripts/launch_train.sh --stop     # stop the run + sidecar
#
# Tunables (env): GPUS, SESSION, SYNC_INTERVAL (checkpoint push cadence, seconds).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
cd "$(repo_root)"

SESSION="${SESSION:-lithos-train}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
GPUS="${GPUS:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 1)}"

case "${1:-}" in
  --attach) exec tmux attach -t "$SESSION" ;;
  --stop)   tmux kill-session -t "$SESSION" 2>/dev/null && ok "stopped $SESSION" || warn "no session $SESSION"; exit 0 ;;
esac

require_cmd uv; require_cmd tmux
require_env_file .env AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 \
                      LITHOS_STORAGE_BASE_URI WANDB_API_KEY
tmux has-session -t "$SESSION" 2>/dev/null \
  && die "session '$SESSION' already running. Attach: bash scripts/launch_train.sh --attach"

# Resume policy: newest local checkpoint, else newest pushed to R2, else fresh.
resume_arg=""
if local_ckpt="$(latest_checkpoint)"; then
  resume_arg="--resume $local_ckpt"
  log "Resuming from local checkpoint: $local_ckpt"
elif uv run python scripts/sync.py pull "$CKPT_REMOTE/latest" runs/_restored/checkpoints/latest >/dev/null 2>&1; then
  resume_arg="--resume runs/_restored/checkpoints/latest"
  log "Resuming from R2 checkpoint: $CKPT_REMOTE/latest"
else
  log "No prior checkpoint found; starting fresh."
fi

mkdir -p runs
train_cmd="uv run torchrun --standalone --nproc_per_node=$GPUS scripts/train_model.py \
--config $TRAIN_CONFIG $resume_arg 2>&1 | tee -a runs/train.log"

tmux new-session -d -s "$SESSION" -n train "$train_cmd"
tmux new-window  -t "$SESSION" -n cksync "SYNC_INTERVAL=$SYNC_INTERVAL bash scripts/sync_checkpoints.sh 2>&1 | tee -a runs/cksync.log"

ok "Launched '$SESSION' on $GPUS GPU(s); checkpoints sync to R2 every ${SYNC_INTERVAL}s."
echo "    attach: bash scripts/launch_train.sh --attach    (window 0 = train, 1 = cksync)"
echo "    stop:   bash scripts/launch_train.sh --stop"
exit 0
