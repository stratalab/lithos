#!/usr/bin/env bash
# Bootstrap a rented GPU box (1x H100 80GB) as the labeling endpoint.
#
# On the box:
#   curl -fsSL <raw-url-of-this-script> | bash
# Then from the local machine, tunnel the endpoint (keeps it private):
#   ssh -L 8000:localhost:8000 ubuntu@<box-ip>
# And label:
#   uv run python scripts/label_quality.py --domain physics-eng \
#       --jsonl data/pilot/physics-eng.jsonl --n 300 \
#       --endpoint http://localhost:8000/v1 --model qwen3-32b
#
# Model: Qwen3-32B FP8 (~33GB weights -> plenty of KV headroom on 80GB;
# Apache-2.0; /no_think soft switch used by the labeling client).

set -euo pipefail

MODEL="${LABELER_MODEL:-Qwen/Qwen3-32B-FP8}"

echo "==> uv + vllm"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/vllm-env --python 3.12 2>/dev/null || true
source ~/vllm-env/bin/activate
uv pip install vllm hf_transfer

echo "==> serving $MODEL on :8000 (first run downloads ~33GB)"
export HF_HUB_ENABLE_HF_TRANSFER=1
exec vllm serve "$MODEL" \
  --served-model-name qwen3-32b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8000
