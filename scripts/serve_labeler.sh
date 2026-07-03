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

# --- GPU driver (some rental images ship bare Ubuntu, no NVIDIA driver) -----
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  if ! lspci | grep -qi nvidia; then
    echo "!! no NVIDIA device on the PCI bus — wrong instance type?" >&2
    exit 1
  fi
  echo "==> NVIDIA device present but driver missing — installing server driver"
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-driver-570-server 2>/dev/null \
    || sudo apt-get install -y -qq nvidia-driver-550-server
  sudo modprobe nvidia || true
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "==> driver installed but not loadable without reboot."
    echo "==> rebooting in 5s — RE-RUN THIS SCRIPT after the box comes back."
    sleep 5
    sudo reboot
    exit 0
  fi
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "==> build deps (vLLM JIT-compiles CUDA utils at startup — needs Python.h)"
sudo apt-get install -y -qq python3.12-dev build-essential 2>/dev/null \
  || sudo apt-get install -y -qq python3-dev build-essential

echo "==> uv + vllm"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/vllm-env --python 3.12 2>/dev/null || true
source ~/vllm-env/bin/activate
uv pip install vllm hf_transfer

echo "==> serving $MODEL on :8000 (first run downloads ~33GB)"
export HF_HUB_ENABLE_HF_TRANSFER=1
# flashinfer's sampler JIT-compiles with nvcc, which bare images lack; the
# torch-native fallback is fine for short labeling generations.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$MODEL" \
  --served-model-name qwen3-32b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8000
