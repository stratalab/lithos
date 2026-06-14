#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu GPU box (NVIDIA driver + CUDA preinstalled,
# e.g. Lambda / CoreWeave). Provisions the environment, pulls the corpus from R2,
# and runs a 2-GPU DDP smoke — then STOPS and prints the launch command.
# Idempotent: safe to re-run if it fails partway.
#
# Do these by hand first (secrets never go in this repo):
#   1) clone the repo and `cd` into it
#   2) scp your .env to the repo root  (R2 creds + WANDB_API_KEY)
#   3) bash scripts/setup_server.sh
#
# Tunables (env): GPUS, MIN_FREE_GB, SMOKE_STEPS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
cd "$(repo_root)"

MIN_FREE_GB="${MIN_FREE_GB:-60}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
GPUS="${GPUS:-$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 0)}"

# -- [1/6] preflight ---------------------------------------------------------
log "[1/6] Preflight"
require_cmd nvidia-smi "expected a GPU box with the NVIDIA driver installed"
[ "$GPUS" -ge 1 ] || die "no GPUs visible to nvidia-smi"
[ "$GPUS" -ge 2 ] || warn "only $GPUS GPU visible; the 100M config targets 2 (set GPUS to override)"
free_gb="$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
[ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] \
  || warn "only ${free_gb}GB free on $(pwd); corpus + checkpoints want >= ${MIN_FREE_GB}GB"
require_env_file .env AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 \
                      LITHOS_STORAGE_BASE_URI WANDB_API_KEY
ok "GPUs: $GPUS | free disk: ${free_gb}GB | .env present"

# -- [2/6] system packages + uv ----------------------------------------------
log "[2/6] System packages + uv"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git build-essential tmux htop curl ca-certificates >/dev/null
else
  warn "apt-get not found; install git/tmux/curl yourself if missing"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
require_cmd uv "uv install failed; see https://docs.astral.sh/uv/"
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# -- [3/6] python deps + CUDA check ------------------------------------------
log "[3/6] Dependencies (uv sync --extra tracking --extra cloud)"
uv sync --extra tracking --extra cloud
$UV python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see CUDA — driver/toolkit mismatch?"
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"{torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
PY
ok "deps installed; torch sees CUDA"

# -- [4/6] pull corpus + tokenizer from R2 -----------------------------------
log "[4/6] Pull corpus + tokenizer from R2"
if [ -f "$CORPUS_LOCAL/corpus_manifest.json" ]; then
  log "corpus already present at $CORPUS_LOCAL (skip)"
else
  $UV python scripts/sync.py pull "$CORPUS_REMOTE" "$CORPUS_LOCAL"
fi
[ -f "$CORPUS_LOCAL/corpus_manifest.json" ] \
  || die "corpus manifest missing after pull — did build_corpus.sh push to R2:$CORPUS_REMOTE?"
[ -d "$TOKENIZER_LOCAL" ] \
  || $UV python scripts/sync.py pull "$TOKENIZER_REMOTE" "$TOKENIZER_LOCAL" \
  || warn "tokenizer pull failed (only needed for sample generation / eval, not training)"
ok "corpus ready: $($UV python -c "import json; m=json.load(open('$CORPUS_LOCAL/corpus_manifest.json')); print(f\"{m['num_tokens']:,} tokens, {len(m['shards'])} shard(s)\")")"

# -- [5/6] 2-GPU DDP smoke ---------------------------------------------------
log "[5/6] ${SMOKE_STEPS}-step DDP smoke on $GPUS GPU(s) (verifies DDP + wandb + checkpoint->R2)"
$UV torchrun --standalone --nproc_per_node="$GPUS" scripts/train_model.py \
  --config "$TRAIN_CONFIG" \
  --override run_name=smoke-2gpu "schedule.max_steps=$SMOKE_STEPS" schedule.warmup_steps=5 \
            checkpoint_interval=10 log_interval=1 'wandb.tags=[smoke]'
smoke_ckpt="$(latest_checkpoint || true)"
[ -n "$smoke_ckpt" ] || die "smoke produced no checkpoint"
$UV python scripts/sync.py push "$smoke_ckpt" "smoke/checkpoint-roundtrip/$(basename "$smoke_ckpt")"
ok "smoke passed: DDP ran across $GPUS GPU(s), wandb logged, checkpoint round-tripped to R2"

# -- [6/6] done --------------------------------------------------------------
echo
ok "Box is READY. Start the full 100M run when you are:"
echo
echo "    bash scripts/launch_train.sh            # runs in tmux, syncs checkpoints to R2"
echo "    bash scripts/launch_train.sh --attach   # watch the live log"
echo
warn "Clean up the smoke artifacts: rm -rf the smoke-2gpu run, and delete R2 smoke/ objects."
exit 0
