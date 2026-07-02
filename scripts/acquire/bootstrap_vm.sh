#!/usr/bin/env bash
# Bootstrap a throwaway acquisition VM (Ubuntu 22.04+) — doc §1.7 / VM→R2 pattern.
#
# Usage (on the VM):
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_ENDPOINT_URL_S3=...
#   export LITHOS_STORAGE_BASE_URI=s3://lithos-data-fineweb-edu
#   export HF_TOKEN=...            # stratalab token (gated datasets)
#   curl -fsSL <raw-url-of-this-script> | bash
#   cd lithos && uv run python scripts/acquire/acquire.py --wave p0 --dry-run
#
# Sizing: the VM's scratch disk must hold the largest single corpus (~600GB);
# a Hetzner dedicated / any cloud box with 1TB NVMe and a fat pipe is ideal.

set -euo pipefail

echo "==> apt deps"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl aria2 p7zip-full

echo "==> uv + python"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> rclone"
command -v rclone >/dev/null || (curl -fsSL https://rclone.org/install.sh | sudo bash)

echo "==> rclone remote 'r2' (Cloudflare R2 via env auth)"
: "${AWS_ENDPOINT_URL_S3:?set AWS_ENDPOINT_URL_S3}"
rclone config create r2 s3 provider=Cloudflare env_auth=true \
  endpoint="${AWS_ENDPOINT_URL_S3}" acl=private --non-interactive >/dev/null

echo "==> lithos repo"
[ -d lithos ] || git clone --depth 1 https://github.com/stratalab/lithos.git
cd lithos
uv sync --no-dev 2>/dev/null || uv sync

echo "==> hf auth"
: "${HF_TOKEN:?set HF_TOKEN}"
uv run hf auth login --token "$HF_TOKEN" >/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> ready. plan the wave:"
echo "    uv run python scripts/acquire/acquire.py --wave p0 --dry-run"
