#!/usr/bin/env bash
# Sidecar: periodically push the newest local checkpoint to R2 at
# $CKPT_REMOTE/latest, so a preempted/terminated box can be resumed with at most
# one interval of lost progress. Started by launch_train.sh; Ctrl-C to stop.
#
#   SYNC_INTERVAL=600 bash scripts/sync_checkpoints.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
cd "$(repo_root)"

SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
require_cmd uv
require_env_file .env AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 LITHOS_STORAGE_BASE_URI

log "checkpoint sync every ${SYNC_INTERVAL}s -> R2:$CKPT_REMOTE/latest  (Ctrl-C to stop)"
last=""
while true; do
  sleep "$SYNC_INTERVAL"
  if ! latest="$(latest_checkpoint)"; then
    log "no checkpoint written yet"
    continue
  fi
  [ "$latest" = "$last" ] && continue   # nothing new since the last push
  log "pushing $latest -> R2:$CKPT_REMOTE/latest"
  if uv run python scripts/sync.py push "$latest" "$CKPT_REMOTE/latest"; then
    last="$latest"
    ok "synced $(basename "$latest")"
  else
    warn "checkpoint push failed; will retry next interval"
  fi
done
