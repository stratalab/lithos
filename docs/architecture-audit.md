# Architecture audit: Lithos vs. Qwen3 (2026-07-05)

**Question:** does the Lithos model faithfully implement the Qwen3 architecture, and are
the from-scratch hyperparameter choices correct? The family thesis depends on it — the
from-scratch models (100M→500M→1B→3B) must be a true **Qwen3 envelope** so one deployment
recipe drives both them and the Qwen3-4B continued-pretrain hero.

**Method:** component-by-component comparison of `lithos/model/*` against the transformers
`Qwen3` reference (`.venv/.../models/qwen3/`) + the Qwen3 hyperparameters. This is
**code-vs-paper**, complementing **E7** (which proved code-vs-transformers *forward parity*,
bit-exact, on imported weights). E7 can't catch a wrong from-scratch *default*, because the
importer copies Qwen3's values — this audit does.

## Verdict: the implementation is correct and the envelope is faithful

Structurally identical to Qwen3, confirmed line-for-line and by E7's parity:

| Component | Status |
|---|---|
| Attention: GQA + `repeat_kv`, **qk-norm** (RMSNorm over `head_dim`, pre-RoPE), no QKV/o bias, `1/√head_dim` scaling, `head_dim` decoupling | ✅ match |
| RoPE: HF `rotate_half`, cos/sin construction | ✅ match |
| RMSNorm: float32 compute → cast → weight-multiply | ✅ bit-identical to `Qwen3RMSNorm` |
| MLP: SwiGLU `down(silu(gate)·up)` | ✅ match |
| Decoder block: pre-norm residual, `input`/`post_attention` layernorm placement | ✅ match |
| Model: embed → blocks → final norm → head, tying, vocab-pad masked from loss/logits, plain CE on pre-shifted targets | ✅ match |

**Nothing was broken.** But the *shape* was Qwen3 while three defaults were Llama-flavored —
the "quietly-wrong default" class E7 structurally cannot catch.

## Findings and resolution

| # | Finding | Was | Qwen3 | Action |
|---|---|---|---|---|
| 1 | `rms_eps` | `1e-5` (Llama) | `1e-6` | **Default → `1e-6`.** No config pinned it, so all go-forward models align; numerically negligible for the existing shakedown. |
| 2 | `rope_theta` | `10000` (Llama-2) | `1e6` | **Default → `1e6`.** Decision below. Existing configs pin `10000` explicitly, so they're unaffected. |
| 3 | `qk_norm` default | `False` | always on | **Default → `True`.** `False` is now explicitly "export as Llama" (`serve/export.py` picks arch by qk-norm); `True` = the Qwen3 envelope. |
| 4 | Dropout | present (knob) | none | Kept as a knob; commented that nonzero diverges. Family configs use `0.0`. |
| 5 | Residual init | depth-scaled (GPT-2/Llama) | plain `N(0,0.02)` | **Kept** — sound for from-scratch, and init only affects from-scratch training, never the imported hero. |
| 6 | Sliding-window attn | none | supported (off by default) | **Not needed** — the dense Qwen3 configs we target don't enable it; revisit only if a hero variant does. |

### The `rope_theta` decision

Adopt **`1e6`** (Qwen3's value) as the default, for three reasons: it future-proofs the
planned context extension (E10 — RoPE-theta scaling + long-doc anneal; starting at `1e6`
avoids a theta jump later), it matches the hero, and it is empirically fine at 2048 context
(Qwen3-4B uses it and handles short sequences well). The cost — the existing 100M shakedown
used `1e4` — is moot: it's superseded by the 500M flagship, and its configs pin `10000`
explicitly (they *must*, to match the trained base's positional encoding).

## Reproducibility note

The existing **`lithos-100m-v0.1`** was trained with the old defaults (`rms_eps=1e-5`,
`rope_theta=10000`, `qk_norm=true`). Its `resolved_config.yaml` records those, so it loads
and reproduces exactly. **Go-forward from-scratch models** (500M flagship onward) use the
Qwen3-aligned defaults above. The `rope_theta` gap is real (1e4 vs 1e6 — a large change to
positional encoding), so **never post-train the existing 100M with the new default**; its
post-train configs correctly pin `10000`.

*Files touched: `lithos/model/{config,norm,rope}.py` (defaults only). Gates green: ruff +
mypy (73 files) + full suite.*
