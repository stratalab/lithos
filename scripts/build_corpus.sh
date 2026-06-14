#!/usr/bin/env bash
# Build the FineWeb-Edu tokenizer + tokenized corpus and push both to R2.
#
# Run this on a CHEAP CPU box (or your laptop) — it is CPU-bound tokenization;
# do NOT burn $30/hr GPU time on it. It is the prerequisite for setup_server.sh,
# which pulls the finished corpus onto the training box.
#
#   bash scripts/build_corpus.sh           # full sample-10BT build (hours, ~20GB)
#   SMOKE=1 bash scripts/build_corpus.sh   # tiny end-to-end validation (minutes)
#
# Requires a .env with R2 credentials (AWS_* + LITHOS_STORAGE_BASE_URI) and,
# for FineWeb-Edu downloads, network access (optionally HF_TOKEN to avoid rate limits).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"
cd "$(repo_root)"

require_cmd uv "install: curl -LsSf https://astral.sh/uv/install.sh | sh"
require_env_file .env AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL_S3 LITHOS_STORAGE_BASE_URI

# Native libs (HF datasets streaming, the tokenizers Rust threadpool) can abort at
# interpreter shutdown — AFTER the work is done — with "PyGILState_Release ...
# finalizing". Quiet the threadpool, and treat the written artifact (not the exit
# code) as the source of truth so a benign teardown crash can't discard a build
# that completed — which on the multi-hour real build would be very costly.
export TOKENIZERS_PARALLELISM=false

# build_step <artifact> <command...>: run a build command; succeed iff it produced
# <artifact>, tolerating a non-zero exit from a post-success shutdown crash.
build_step() {
  local artifact="$1"; shift
  local rc=0
  "$@" || rc=$?
  [ -f "$artifact" ] || die "build failed (exit $rc): $artifact not produced — see output above."
  [ "$rc" -eq 0 ] || warn "command exited $rc after writing $artifact (benign shutdown crash); continuing."
}

tok_args=()
if [ "${SMOKE:-0}" = "1" ]; then
  warn "SMOKE build: tiny tokenizer + tiny corpus, pushed to a disposable R2 smoke/ prefix."
  CORPUS_CONFIG=configs/data/fineweb-edu-smoke.yaml
  CORPUS_LOCAL=data/fineweb-edu/corpus-smoke-v0.1
  CORPUS_REMOTE=smoke/corpus-fineweb-edu-smoke
  TOKENIZER_REMOTE=smoke/tokenizers/fineweb-edu-32k
  tok_args=(--max-documents 3000)
fi

log "[1/4] Train tokenizer ($TOKENIZER_CONFIG) -> $TOKENIZER_LOCAL"
build_step "$TOKENIZER_LOCAL/tokenizer.json" \
  $UV python scripts/train_tokenizer.py --config "$TOKENIZER_CONFIG" "${tok_args[@]}"

log "[2/4] Push tokenizer -> R2:$TOKENIZER_REMOTE"
$UV python scripts/sync.py push "$TOKENIZER_LOCAL" "$TOKENIZER_REMOTE"

log "[3/4] Build tokenized corpus ($CORPUS_CONFIG) -> $CORPUS_LOCAL"
build_step "$CORPUS_LOCAL/corpus_manifest.json" \
  $UV python scripts/tokenize_corpus.py --config "$CORPUS_CONFIG"

log "[4/4] Push corpus -> R2:$CORPUS_REMOTE"
$UV python scripts/sync.py push "$CORPUS_LOCAL" "$CORPUS_REMOTE"

ok "Corpus built and pushed. The training box pulls these (setup_server.sh):"
echo "      tokenizer : R2:$TOKENIZER_REMOTE  ->  $TOKENIZER_LOCAL"
echo "      corpus    : R2:$CORPUS_REMOTE  ->  $CORPUS_LOCAL"
[ "${SMOKE:-0}" = "1" ] && warn "SMOKE artifacts live under R2 smoke/ — delete them when you're done validating."
exit 0
