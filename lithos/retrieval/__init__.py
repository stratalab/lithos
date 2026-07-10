"""Retrieval-in-context (R1, as rev B redefines it).

Lives **above** the token stream: passages are prepended and cited, never interpolated
into the decode loop. Every mechanism with positive evidence lives above the stream; every
one with negative evidence lives below it (`docs/composite-plan.md` §1).

    docs -> chunk -> embed -> Datastore(version, tier gate, decontam assert)
                                 |
                                 v
                          DocumentRetriever  --(passages + citations)-->  CompositeModel
"""

from lithos.retrieval.chunk import Chunk, chunk_document
from lithos.retrieval.embed import Embedder, HashingEmbedder
from lithos.retrieval.index import Datastore, NumpyExactIndex
from lithos.retrieval.retrieve import DocumentRetriever
from lithos.retrieval.types import (
    Passage,
    RetrievedContext,
    Retriever,
    StubRetriever,
    assert_datastore_tier,
)

__all__ = [
    "Chunk",
    "Datastore",
    "DocumentRetriever",
    "Embedder",
    "HashingEmbedder",
    "NumpyExactIndex",
    "Passage",
    "RetrievedContext",
    "Retriever",
    "StubRetriever",
    "assert_datastore_tier",
    "chunk_document",
]
