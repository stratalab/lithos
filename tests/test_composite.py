"""The composite model layer, end to end (`lithos/serve/composite.py`).

A scripted model + round-trip tokenizer drive a REAL sandbox execution through the
composite, so the whole path — retrieve → prepend → decode (policy fixes the support) →
pause → execute → inject → resume → provenance — is exercised without training.

The load-bearing tests are the decode-policy ones: a guarantee anything downstream can overturn
is not a guarantee. The policy runs FIRST, on the raw logits, and is final because every
later stage (temperature/top-k/top-p) can only *remove* mass. Running it last would be
equally safe and strictly worse — see
`test_banning_the_models_favourite_token_does_not_empty_the_nucleus`.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from lithos.model.generation import generate
from lithos.posttrain.chat_template import TIR_TOKENS
from lithos.serve.composite import (
    Citation,
    CompositeModel,
    DenyTokensPolicy,
    Passage,
    RetrievedContext,
    Retriever,
    ServedModelId,
    StubRetriever,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")), reason="POSIX-only sandbox"
)

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_ALL = _SPECIALS + list(TIR_TOKENS)


class RoundTripTok:
    """Specials/TIR at fixed low ids; each char maps to base+ord(c) so encode/decode
    round-trips and the sandbox runs the decoded code."""

    def __init__(self):
        self._tok2id = {t: i for i, t in enumerate(_ALL)}
        self._id2tok = {i: t for t, i in self._tok2id.items()}
        self._base = len(_ALL)

    def token_to_id(self, token):
        return self._tok2id.get(token)

    def encode(self, text):
        return SimpleNamespace(ids=[self._base + ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i in self._id2tok:
                if not skip_special_tokens:
                    out.append(self._id2tok[i])
            else:
                out.append(chr(i - self._base))
        return "".join(out)

    @property
    def vocab(self):
        return self._base + 256


class ScriptedModel(torch.nn.Module):
    """Forces a fixed token sequence via a large logit at the last position."""

    def __init__(self, script, vocab):
        super().__init__()
        self.script = list(script)
        self.vocab = vocab
        self.calls = 0

    def forward(self, input_ids, kv_caches=None):
        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab), -30.0)
        nxt = self.script[min(self.calls, len(self.script) - 1)]
        logits[:, -1, nxt] = 30.0
        self.calls += 1
        return logits, None


@pytest.fixture
def tok():
    return RoundTripTok()


def _ids(tok, text):
    return tok.encode(text).ids


def _passage(text="Bernoulli: p + 0.5*rho*v^2 = const", tier="restricted", n=1):
    return Passage(
        text=text,
        source_id=f"src:pearson-fluids-ch{n}",
        record_id=f"rec:{n}",
        text_sha256=f"{n:064x}",
        tier=tier,
        score=1.0,
    )


# ── identity: the four-tuple IS the parts list ────────────────────────────────


def test_served_model_id_digest_changes_with_every_component():
    base = ServedModelId("w" * 64, "ds-v1", "policy-v1", "t" * 64)
    seen = {base.digest()}
    for kw in (
        {"weights_sha256": "x" * 64},
        {"datastore_version": "ds-v2"},
        {"decode_policy_version": "policy-v2"},
        {"tool_env_sha": "u" * 64},
    ):
        import dataclasses

        variant = dataclasses.replace(base, **kw)
        assert variant.digest() not in seen, f"digest ignored {next(iter(kw))}"
        seen.add(variant.digest())


def test_tool_env_sha_is_part_of_identity_and_is_stable():
    from lithos.posttrain.sandbox import tool_env_fingerprint, tool_env_sha

    assert tool_env_sha() == tool_env_sha()  # stable within a process
    fp = tool_env_fingerprint()
    assert fp["env"]["PYTHONHASHSEED"] == "0"  # the determinism that makes keys match
    assert "numpy" in fp["packages"] and "scipy" in fp["packages"] and "sympy" in fp["packages"]


def test_checker_import_set_is_a_code_constant_not_a_doc_claim():
    from lithos.posttrain.sandbox import CHECKER_IMPORT_SET

    assert frozenset({"stdlib", "numpy", "scipy", "sympy"}) == CHECKER_IMPORT_SET


def test_retriever_without_a_pinned_datastore_version_is_rejected(tok):
    model = ScriptedModel([0], tok.vocab)
    with pytest.raises(ValueError, match="unevaluable"):
        CompositeModel(model, tok, weights_sha256="w" * 64, retriever=StubRetriever([_passage()]))


# ── the decode policy is the final authority on the support ───────────────────


class _UniformModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, kv_caches=None):
        b, t = input_ids.shape
        return torch.zeros((b, t, self.vocab)), None


def test_decode_policy_overrides_the_model_greedy(tok):
    """The model *wants* token X. The policy forbids X. X must not be emitted."""
    vocab = tok.vocab
    want = 42
    model = ScriptedModel([want], vocab)
    out = generate(model, torch.tensor([[1]]), 1, greedy=True, use_cache=False)
    assert out[0, -1].item() == want  # unconstrained: the model gets its way

    model = ScriptedModel([want], vocab)
    out = generate(
        model,
        torch.tensor([[1]]),
        1,
        greedy=True,
        use_cache=False,
        logits_processor=DenyTokensPolicy({want}),
    )
    assert out[0, -1].item() != want


def test_decode_policy_survives_top_p_and_temperature(tok):
    """top-p/temperature run AFTER the policy and only remove mass, so they cannot
    reintroduce a banned token. A uniform model + top_p=1.0 would otherwise sample
    anything in the vocab."""
    vocab = 64
    deny = set(range(1, vocab))  # everything except token 0
    g = torch.Generator().manual_seed(0)
    out = generate(
        _UniformModel(vocab),
        torch.tensor([[0]]),
        12,
        temperature=2.0,
        top_p=1.0,
        generator=g,
        use_cache=False,
        logits_processor=DenyTokensPolicy(deny),
    )
    assert set(out[0, 1:].tolist()) == {0}


def test_a_policy_that_adds_mass_is_rejected():
    """A processor that can *raise* a logit can reintroduce a forbidden token."""

    def cheat(logits, generated):
        return logits + 1.0

    with pytest.raises(ValueError, match="may only remove probability mass"):
        generate(
            _UniformModel(8),
            torch.tensor([[0]]),
            1,
            greedy=True,
            use_cache=False,
            logits_processor=cheat,
        )


def test_a_policy_that_masks_everything_raises_rather_than_emit_nan():
    with pytest.raises(ValueError, match="unsatisfiable"):
        generate(
            _UniformModel(8),
            torch.tensor([[0]]),
            1,
            greedy=False,
            temperature=1.0,
            use_cache=False,
            logits_processor=DenyTokensPolicy(set(range(8))),
        )


def test_banning_the_models_favourite_token_does_not_empty_the_nucleus(tok):
    """The reason the policy runs FIRST rather than last.

    A confident model puts ~all mass on one token, so top-p collapses the nucleus to
    exactly that token. If the policy ran after top-p it would ban the only survivor and
    the constraint would be unsatisfiable — even though the model had a whole vocabulary
    of allowed alternatives. Running it first lets top-p renormalize over the allowed set.
    """
    vocab = 32
    want = 7
    g = torch.Generator().manual_seed(0)
    out = generate(
        ScriptedModel([want], vocab),
        torch.tensor([[0]]),
        1,
        temperature=1.0,
        top_p=0.95,  # against a +30/-30 model this keeps ONLY `want`
        generator=g,
        use_cache=False,
        logits_processor=DenyTokensPolicy({want}),
    )
    assert out[0, -1].item() != want  # sampled some allowed token; did not raise


def test_policy_version_flows_into_the_served_model_id(tok):
    policy = DenyTokensPolicy({5}, version="decode-policy-v7")
    cm = CompositeModel(ScriptedModel([0], tok.vocab), tok, weights_sha256="w" * 64, policy=policy)
    assert cm.id.decode_policy_version == "decode-policy-v7"


# ── retrieval lives above the token stream ────────────────────────────────────


def test_stub_retriever_admits_restricted_and_rejects_unknown():
    """The datastore is exactly where restricted content belongs: cited, never trained."""
    StubRetriever([_passage(tier="restricted")])  # allowed
    with pytest.raises(ValueError, match="tier='unknown'"):
        StubRetriever([_passage(tier="unknown")])


def test_stub_retriever_satisfies_the_protocol():
    assert isinstance(StubRetriever([_passage()]), Retriever)


def _composite(tok, script, **kw):
    return CompositeModel(
        ScriptedModel(script, tok.vocab), tok, weights_sha256="w" * 64, device="cpu", **kw
    )


def test_no_retrieval_means_zero_context_tokens(tok):
    end = tok.token_to_id("<|end|>")
    cm = _composite(tok, [end])
    res = cm.generate("what is Bernoulli?", max_new=4, use_cache=False)
    assert res.context_tokens == 0
    assert res.citations == []
    assert res.model_id.datastore_version is None


def test_retrieval_costs_context_and_is_cited(tok):
    end = tok.token_to_id("<|end|>")
    cm = _composite(
        tok,
        [end],
        retriever=StubRetriever([_passage()]),
        datastore_version="ds-2026-07-10",
    )
    res = cm.generate("what is Bernoulli?", context_token_budget=512, max_new=4, use_cache=False)

    assert res.context_tokens > 0, "prepended passages must consume the context budget"
    assert len(res.citations) == 1
    c = res.citations[0]
    assert isinstance(c, Citation)
    assert c.source_id == "src:pearson-fluids-ch1" and c.tier == "restricted"
    assert c.tokens > 0
    # a retrieved fact is citable BY CONSTRUCTION -- no attribution method required
    assert res.provenance()["citations"][0]["text_sha256"] == f"{1:064x}"


def test_context_budget_is_enforced_by_dropping_passages(tok):
    """The scarcest resource at 500M is context. Spend no more than the budget."""
    end = tok.token_to_id("<|end|>")
    passages = [_passage(text="x" * 200, n=i) for i in range(1, 5)]
    cm = _composite(tok, [end], retriever=StubRetriever(passages), datastore_version="ds-v1")
    res = cm.generate("q", context_token_budget=250, max_new=2, use_cache=False)
    assert res.context_tokens <= 250
    assert 0 < len(res.citations) < 4, "some passages must have been dropped to fit"
    assert sum(c.tokens for c in res.citations) == res.context_tokens


def test_zero_budget_retrieves_nothing_even_with_a_retriever(tok):
    end = tok.token_to_id("<|end|>")
    cm = _composite(tok, [end], retriever=StubRetriever([_passage()]), datastore_version="ds-v1")
    res = cm.generate("q", context_token_budget=0, max_new=2, use_cache=False)
    assert res.context_tokens == 0 and res.citations == []


# ── the tool loop lives below it, and is provenance-tracked ───────────────────


def test_tool_call_executes_and_lands_in_the_provenance_channel(tok):
    """The judgment to call the tool is trained in; the execution never can be."""
    close = tok.token_to_id("<|/tool|>")
    py = tok.token_to_id("<|python|>")
    end = tok.token_to_id("<|end|>")
    code = "print(6*7)"
    script = [py, *_ids(tok, code), close, end]

    cm = _composite(tok, script)
    res = cm.generate("what is 6*7?", max_new=64, use_cache=False)

    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc.runtime == "python" and tc.code == code
    assert "42" in tc.output  # the sandbox really ran it
    assert res.provenance()["tool_calls"][0]["output"].strip() == "42"


def test_injected_tool_result_is_not_a_reasoning_token(tok):
    """`reasoning_tokens` counts the policy's own tokens. The sandbox's move is not one —
    the same mask that keeps tool results out of the loss keeps them out of this count."""
    close = tok.token_to_id("<|/tool|>")
    py = tok.token_to_id("<|python|>")
    end = tok.token_to_id("<|end|>")
    script = [py, *_ids(tok, "print(1)"), close, end]

    cm = _composite(tok, script)
    res = cm.generate("q", max_new=64, use_cache=False)

    assert res.reasoning_tokens < res.completion_tokens, "the injected result was counted"
    assert res.reasoning_tokens == sum(res.action_mask[res.prompt_tokens :])


def test_decode_policy_applies_inside_the_tool_loop(tok):
    """The policy is the last write on every sampled token, tool-call segments included."""
    py = tok.token_to_id("<|python|>")
    end = tok.token_to_id("<|end|>")
    cm = _composite(tok, [py, end], policy=DenyTokensPolicy({py}))
    res = cm.generate("q", max_new=6, use_cache=False)
    assert py not in res.token_ids[res.prompt_tokens :]
    assert res.tool_calls == []


# ── the composite still looks like a model ────────────────────────────────────


def test_result_accounting_is_self_consistent(tok):
    end = tok.token_to_id("<|end|>")
    cm = _composite(tok, [end], retriever=StubRetriever([_passage()]), datastore_version="ds-v1")
    res = cm.generate("q", context_token_budget=512, max_new=4, use_cache=False)

    assert res.prompt_tokens + res.completion_tokens == len(res.token_ids)
    assert len(res.action_mask) == len(res.token_ids)
    assert res.context_tokens < res.prompt_tokens  # context is part of the prompt, not all of it
    assert res.provenance()["served_model_id"][1] == "ds-v1"


def test_retrieved_context_defaults_are_empty():
    ctx = RetrievedContext()
    assert ctx.passages == () and ctx.tokens_used == 0


# ── the real retriever, inside the composite ──────────────────────────────────


def _real_store(tok, texts, tier="restricted"):
    import hashlib

    from lithos.retrieval import Datastore, HashingEmbedder

    embedder = HashingEmbedder(dim=256)
    docs = [
        {
            "id": f"rec:{i}",
            "text": t,
            "source": f"src:{i}",
            "tier": tier,
            "metadata": {
                "source_id": f"src:{i}",
                "record_id": f"rec:{i}",
                "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
            },
        }
        for i, t in enumerate(texts)
    ]
    store = Datastore.build(
        docs, tok, embedder, tokenizer_name="roundtrip", max_tokens=256, overlap_tokens=16
    )
    return store, embedder


def test_composite_with_a_real_retriever_cites_the_right_source(tok):
    from lithos.retrieval import DocumentRetriever

    end = tok.token_to_id("<|end|>")
    store, embedder = _real_store(
        tok,
        [
            "bernoulli principle relates pressure and velocity in a fluid",
            "the mitochondrion is the powerhouse of the cell",
        ],
    )
    retriever = DocumentRetriever(store, embedder, top_k=1)
    cm = CompositeModel(
        ScriptedModel([end], tok.vocab), tok, weights_sha256="w" * 64, retriever=retriever
    )

    res = cm.generate(
        "what does bernoulli say about fluid pressure?",
        context_token_budget=512,
        max_new=4,
        use_cache=False,
    )
    assert len(res.citations) == 1
    c = res.citations[0]
    assert c.source_id == "src:0", "retrieved the mitochondrion passage"
    assert c.tier == "restricted"  # cited on every use; never in the weights
    assert c.chunk_sha256 and c.tokens > 0
    assert res.context_tokens > 0


def test_a_real_retriever_pins_the_datastore_version_without_being_asked(tok):
    """Nobody should have to restate a derived value, and a hand-written one can lie."""
    from lithos.retrieval import DocumentRetriever

    store, embedder = _real_store(tok, ["alpha beta gamma"])
    cm = CompositeModel(
        ScriptedModel([tok.token_to_id("<|end|>")], tok.vocab),
        tok,
        weights_sha256="w" * 64,
        retriever=DocumentRetriever(store, embedder),
    )
    assert cm.id.datastore_version == store.version
    assert cm.id.datastore_version.startswith("ds:")


def test_editing_one_document_changes_the_served_model_identity(tok):
    """C5, in miniature: a corpus-caused change is visible in the model's identity, so a
    regression can be bisected to the corpus rather than blamed on the weights."""
    from lithos.retrieval import DocumentRetriever

    def _id_for(texts):
        store, embedder = _real_store(tok, texts)
        return CompositeModel(
            ScriptedModel([tok.token_to_id("<|end|>")], tok.vocab),
            tok,
            weights_sha256="w" * 64,
            retriever=DocumentRetriever(store, embedder),
        ).id

    a = _id_for(["alpha beta", "gamma delta"])
    b = _id_for(["alpha beta", "gamma delta"])
    c = _id_for(["alpha beta", "gamma DELTA"])

    assert a.digest() == b.digest()  # same weights, same corpus -> same model
    assert a.digest() != c.digest()  # same weights, one edited doc -> different model
    assert a.weights_sha256 == c.weights_sha256  # and it was NOT the weights that moved
