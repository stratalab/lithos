"""Data: documents -> filters -> dedup -> tokenized shards -> packed loader (Phase 3)."""

from lithos.data.dataloader import PackedDataLoader, PackedDataset
from lithos.data.dedup import ExactDocumentDeduper, ExactLineDeduper
from lithos.data.documents import DocumentSource, iter_documents, normalize
from lithos.data.filters import DocumentFilter, FilterConfig, check_document
from lithos.data.manifest import corpus_manifest
from lithos.data.packing import get_sequence, num_sequences
from lithos.data.pipeline import CorpusBuildConfig, build_corpus
from lithos.data.shard import ShardWriter, dtype_for_vocab, load_shard
from lithos.data.tokenize import DocumentTokenizer, tokenize_documents

__all__ = [
    "CorpusBuildConfig",
    "DocumentFilter",
    "DocumentSource",
    "DocumentTokenizer",
    "ExactDocumentDeduper",
    "ExactLineDeduper",
    "FilterConfig",
    "PackedDataLoader",
    "PackedDataset",
    "ShardWriter",
    "build_corpus",
    "check_document",
    "corpus_manifest",
    "dtype_for_vocab",
    "get_sequence",
    "iter_documents",
    "load_shard",
    "normalize",
    "num_sequences",
    "tokenize_documents",
]
