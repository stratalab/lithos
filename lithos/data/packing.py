"""Pack flat token streams into fixed-length training sequences (PRD §8.2).

v0 uses standard concatenation: documents are joined into the token stream with
``<bos>``/``<eos>`` markers (at tokenize time) and chopped into windows of
``seq_len`` tokens plus one shifted-label token. Cross-document attention "bleed"
is the default; intra-document masking / RoPE position reset is a deferred config
flag (PRD §27) that would require carrying document-boundary offsets here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def num_sequences(num_tokens: int, seq_len: int) -> int:
    """Number of (seq_len + 1)-token windows (stride seq_len) in the stream."""
    if num_tokens <= seq_len:
        return 0
    return (num_tokens - 1) // seq_len


def get_sequence(
    tokens: NDArray[np.integer], index: int, seq_len: int
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Return (x, y) for window ``index``; y is x shifted by one token."""
    start = index * seq_len
    window = np.asarray(tokens[start : start + seq_len + 1])
    return window[:-1], window[1:]
