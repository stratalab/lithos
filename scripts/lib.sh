#!/usr/bin/env bash
# Shared helpers for the Lithos ops scripts (build_corpus / setup_server /
# launch_train / sync_checkpoints). Source this file; do not execute it.
#
# Secrets never live here or in any committed script: credentials come from a
# git-ignored .env that you scp to the box. These helpers only *check* it.

# --- logging (warn/die -> stderr so stdout stays parseable) ------------------
_c() { printf '\033[%sm' "$1"; }
log()  { printf '%s[lithos]%s %s\n' "$(_c '1;34')" "$(_c 0)" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n'  "$(_c '1;32')" "$(_c 0)" "$*"; }
warn() { printf '%s[warn]%s %s\n'  "$(_c '1;33')" "$(_c 0)" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n'  "$(_c '1;31')" "$(_c 0)" "$*" >&2; exit 1; }

# require_cmd <cmd> [hint]
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1${2:+ — $2}"
}

# Repo root, derived from this file's location (scripts/lib.sh -> repo root).
repo_root() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }

# Verify an env file exists and defines each required KEY, without printing values.
require_env_file() {
  local env_file="$1"; shift
  [ -f "$env_file" ] || die "$env_file not found. scp it from your laptop \
(it is git-ignored — never commit secrets)."
  local missing=() key
  for key in "$@"; do
    grep -qE "^[[:space:]]*(export[[:space:]]+)?${key}=" "$env_file" || missing+=("$key")
  done
  [ "${#missing[@]}" -eq 0 ] || die "$env_file is missing required keys: ${missing[*]}"
}

# Newest checkpoint dir of the most-recently-touched run (path on stdout; rc!=0 if none).
latest_checkpoint() {
  local run ckpt
  run="$(ls -dt runs/*/ 2>/dev/null | head -1 || true)"
  [ -n "$run" ] || return 1
  # A run dir can exist with no checkpoint yet; the pipeline below would still
  # exit 0 with empty output, so guard explicitly to avoid a "--resume <empty>".
  ckpt="$(ls -d "${run}checkpoints/"step_* 2>/dev/null | sort -V | tail -1)"
  [ -n "$ckpt" ] || return 1
  printf '%s\n' "$ckpt"
}

# --- artifact layout: the contract between build (push) and setup (pull) ------
# Local paths mirror the committed configs; remote paths are relative to
# LITHOS_STORAGE_BASE_URI (the R2 bucket set in .env). Override any via env.
: "${TOKENIZER_CONFIG:=configs/tokenizer/fineweb-edu-32k.yaml}"
: "${CORPUS_CONFIG:=configs/data/fineweb-edu-100m.yaml}"
: "${TRAIN_CONFIG:=configs/train/100m.yaml}"
: "${TOKENIZER_LOCAL:=artifacts/tokenizer/fineweb-edu-32k}"
: "${TOKENIZER_REMOTE:=tokenizers/fineweb-edu-32k}"
: "${CORPUS_LOCAL:=data/fineweb-edu/corpus-100m-v0.1}"
: "${CORPUS_REMOTE:=corpus/fineweb-edu-100m-v0.1}"
: "${CKPT_REMOTE:=checkpoints/lithos-100m}"
