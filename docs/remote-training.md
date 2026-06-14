# Remote training runbook (2×H100)

Zero-to-training on a fresh GPU box, in three scripts. The model weights are the
thing you own; everything here is just plumbing to get the run started fast and
survive a box being reclaimed.

```
build_corpus.sh   (cheap CPU box)  →  R2  →  setup_server.sh  →  launch_train.sh
   tokenize once, push                       provision + smoke      run + checkpoint sync
```

Secrets never live in the repo. You `scp` a git-ignored `.env` (R2 creds +
`WANDB_API_KEY`) to each box; the scripts only *check* it's present, never read it.

## 0. One-time: build the corpus (CPU box or laptop)

Tokenizing ~10B tokens is CPU-bound — don't do it on a $30/hr GPU. On any box
with `.env` (R2 creds) and network:

```bash
SMOKE=1 bash scripts/build_corpus.sh     # ~minutes: validate the whole chain first
bash scripts/build_corpus.sh             # the real sample-10BT build (hours, ~20GB)
```

This trains the tokenizer, builds the tokenized shards + manifest, and pushes
both to R2 (`tokenizers/fineweb-edu-32k`, `corpus/fineweb-edu-100m-v0.1`).

## 1. On the GPU box: provision + smoke

Managed Ubuntu box (NVIDIA driver + CUDA preinstalled):

```bash
git clone https://github.com/stratalab/lithos.git && cd lithos
# from your laptop, in another shell:
#   scp .env  user@box:~/lithos/.env
bash scripts/setup_server.sh
```

`setup_server.sh` is idempotent and does: preflight (GPUs, disk, `.env`) →
apt + `uv` → `uv sync --extra tracking --extra cloud` → verify `torch.cuda` →
pull the corpus from R2 → a 20-step **2-GPU DDP smoke** that proves DDP, W&B
logging, and checkpoint→R2 all work — then it **stops**.

> Cloning a private repo needs auth: a GitHub PAT over HTTPS, a deploy key, or
> `gh auth login`.

## 2. Launch the real run

```bash
bash scripts/launch_train.sh            # starts in tmux; auto-resumes if a checkpoint exists
bash scripts/launch_train.sh --attach   # watch (window 0 = train, 1 = checkpoint sync)
bash scripts/launch_train.sh --stop     # stop run + sidecar
```

It runs `torchrun` in a detached `tmux` session (survives SSH drops) and starts
a sidecar that pushes the newest checkpoint to `R2:checkpoints/lithos-100m/latest`
every `SYNC_INTERVAL` seconds (default 600). If the box is reclaimed, bring up a
new one, run `setup_server.sh`, then `launch_train.sh` — it pulls the latest
checkpoint from R2 and resumes.

Live metrics: the W&B project (`stratalab/lithos`). Canonical local record:
`runs/<id>/metrics.jsonl`.

## Knobs

| Env | Default | Meaning |
|-----|---------|---------|
| `GPUS` | autodetected | `--nproc_per_node` for torchrun |
| `SYNC_INTERVAL` | `600` | checkpoint→R2 push cadence (s) |
| `SESSION` | `lithos-train` | tmux session name |
| `MIN_FREE_GB` | `60` | disk preflight threshold |
| `CORPUS_REMOTE` / `CORPUS_LOCAL` | see `scripts/lib.sh` | artifact paths |

On 80GB H100s you can set `grad_checkpointing: false` in `configs/train/100m.yaml`
for more throughput (it's `true` to fit seq 2048 on a 12GB local card).
